"""Tests for :mod:`evalshift.runner.orchestrator`.

We exercise the orchestrator end-to-end with:

* A real :class:`EvalShiftConfig` + :class:`Suite` constructed in-memory.
* A real :class:`CacheStore` backed by ``sqlite+aiosqlite:///:memory:``.
* A *fake* :class:`ModelClient` whose ``complete`` method returns
  deterministic responses without touching the network.

These tests are the closest thing to an end-to-end integration check we
can run in unit-test scope. The orchestrator's CLI-level entry point is
covered separately in ``tests/unit/test_run_command.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from evalshift.cache.store import CacheStore
from evalshift.captures.reader import CaptureError
from evalshift.captures.toolset import fingerprint_tools
from evalshift.config.loader import load_config
from evalshift.config.models import (
    Defaults,
    EvalShiftConfig,
    PromptDefinition,
)
from evalshift.evaluators.tool_models import ToolSpec, ToolTrace
from evalshift.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    ToolCompletionResult,
)
from evalshift.runner.checkpoint import (
    iter_calls,
    read_state,
)
from evalshift.runner.orchestrator import (
    CHECKPOINT_EVERY,
    RunAborted,
    RunResult,
    _build_work_list,
    _fingerprint_toolset,
    build_messages,
    history_for_cache_key,
    resolve_example_tools,
    run_orchestrator,
    toolset_base_candidates,
)
from evalshift.suite.loader import load_jsonl
from evalshift.suite.models import ChatMessage, HistoryToolCall, Suite
from tests.unit.suite_examples import suite_example

IN_MEMORY_DB = "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def cache() -> AsyncIterator[CacheStore]:
    s = await CacheStore.open(database_url=IN_MEMORY_DB)
    try:
        yield s
    finally:
        await s.close()


def _config(*, concurrency: int = 4, max_cost: float = 100.0) -> EvalShiftConfig:
    return EvalShiftConfig(
        prompts=[
            PromptDefinition(
                id="greet",
                detection="manual",
                content="Hello {name}",
                variables=["name"],
            ),
        ],
        defaults=Defaults(concurrency=concurrency, max_cost_usd=max_cost),
    )


def _suite(n: int = 4) -> Suite:
    return Suite(
        examples=[suite_example(id=f"ex{i}", inputs={"name": f"User{i}"}) for i in range(n)],
    )


def _writeable_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(config_path, suite_path, runs_base)`` rooted in ``tmp_path``."""
    config_path = tmp_path / "evalshift.yaml"
    config_path.write_text("# placeholder; we pass the loaded config object\n")
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text("# placeholder\n")
    runs_base = tmp_path / "runs"
    return config_path, suite_path, runs_base


def _make_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_text: str = "ok",
    raise_for: dict[str, Exception] | None = None,
) -> dict[str, int]:
    """Replace ``ModelClient.complete`` with a deterministic fake.

    Returns a counter dict so tests can assert how many real calls
    happened (i.e. how many were *not* served from cache).
    """
    counter = {"calls": 0}
    raise_for = raise_for or {}

    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        counter["calls"] += 1
        model = str(kwargs["model"])
        if model in raise_for:
            raise raise_for[model]
        return CompletionResult(
            text=f"{response_text}({kwargs['prompt']})",
            model_id=model,
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.0001,
            latency_ms=42,
        )

    monkeypatch.setattr(ModelClient, "complete", fake_complete)
    return counter


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestOrchestratorHappyPath:
    async def test_single_run_writes_state_and_raw(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=3),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        # 1 prompt x 3 examples x 2 models = 6 calls
        assert result.total_calls == 6
        assert result.completed_calls == 6
        assert result.failed_calls == 0

        # State on disk must be marked completed.
        state = read_state(result.run_dir)
        assert state.status == "completed"
        assert state.completed_evaluations == 6

        # raw.jsonl has one line per call.
        rows = list(iter_calls(result.run_dir))
        assert len(rows) == 6
        roles = {(r.role, r.example_id) for r in rows}
        assert ("source", "ex0") in roles
        assert ("target", "ex2") in roles

    async def test_suite_name_recorded_in_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        """Downstream stages resolve per-suite evaluators from this field."""
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=1),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            suite_name="main_chat",
            yes=True,
            cache=cache,
        )

        assert read_state(result.run_dir).suite_name == "main_chat"

    async def test_suite_name_absent_for_raw_suite_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=1),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert read_state(result.run_dir).suite_name is None

    async def test_canonical_model_id_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=2),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",  # alias
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        rows = list(iter_calls(result.run_dir))
        models = {r.model_id for r in rows}
        # Aliases must resolve to canonical ids in the recorded calls.
        assert models == {"gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"}

    async def test_repeat_run_with_cache_serves_cached_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        counter = _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        kwargs: dict[str, Any] = {
            "config": _config(),
            "config_path": config_path,
            "suite": _suite(n=2),
            "suite_path": suite_path,
            "source_model": "gemini-2.5-flash",
            "target_model": "gemini-2.5-pro",
            "runs_base": runs_base,
            "yes": True,
            "cache": cache,
        }

        # First run: every call is live.
        first = await run_orchestrator(**kwargs)
        assert first.live_calls == 4
        assert first.cached_calls == 0
        assert counter["calls"] == 4

        # Second run with the same config + suite + cache: every call
        # should be a cache hit, and no new live calls happen.
        second = await run_orchestrator(**kwargs)
        assert second.cached_calls == 4
        assert second.live_calls == 0
        assert counter["calls"] == 4  # unchanged from before

    async def test_effective_max_tokens_and_finish_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        # Capture what max_tokens the orchestrator threads into complete(),
        # and confirm the returned finish_reason lands on the Call.
        seen: dict[str, Any] = {}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            seen["max_tokens"] = kwargs.get("max_tokens")
            return CompletionResult(
                text="partial",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                finish_reason="length",
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        # Per-prompt override wins over the run-wide default.
        config = EvalShiftConfig(
            prompts=[
                PromptDefinition(
                    id="greet",
                    detection="manual",
                    content="Hello {name}",
                    variables=["name"],
                    max_tokens=2000,
                ),
            ],
            defaults=Defaults(concurrency=2, max_cost_usd=100.0, max_tokens=8000),
        )
        result = await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=_suite(n=1),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert seen["max_tokens"] == 2000  # prompt override, not the 8000 default
        rows = list(iter_calls(result.run_dir))
        assert rows
        assert all(r.finish_reason == "length" and r.truncated for r in rows)

    async def test_default_max_tokens_used_without_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        seen: dict[str, Any] = {}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            seen["max_tokens"] = kwargs.get("max_tokens")
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        config = _config()  # no per-prompt max_tokens; Defaults default is 4096
        await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=_suite(n=1),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )
        assert seen["max_tokens"] == 4096

    async def test_example_generation_config_reaches_the_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        # A promoted example's recorded generation_config is translated into
        # temperature + extra (response_format) on the dispatched call.
        seen: dict[str, Any] = {}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            seen["temperature"] = kwargs.get("temperature")
            seen["extra"] = kwargs.get("extra")
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        suite = Suite(
            examples=[
                suite_example(
                    id="ex0",
                    inputs={"name": "Alex"},
                    generation_config={
                        "temperature": 0.5,
                        "response_mime_type": "application/json",
                    },
                ),
            ],
        )
        await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )
        assert seen["temperature"] == 0.5
        assert seen["extra"] == {"response_format": {"type": "json_object"}}


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestOrchestratorResume:
    async def test_resume_picks_up_after_partial_progress(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        # First "run" stops after 4 of 6 calls because the fake client
        # raises an in-process exception that the orchestrator currently
        # would re-raise. We simulate a partial run differently: we run
        # a smaller suite first, then expand.
        # For this test, the simplest reliable simulation is to
        # short-circuit the first run, append calls to raw.jsonl
        # manually, then ask the orchestrator to "resume".
        from datetime import datetime

        from evalshift.runner import checkpoint as cp_mod
        from evalshift.runner.models import Call, RunModels, RunState

        config = _config()
        suite = _suite(n=2)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        # Manually create an in-progress run whose first 2 calls (out of
        # 1 prompt x 2 examples x 2 models = 4) are already completed.
        run_dir = runs_base / "r_20260601_dead00"
        config_hash = cp_mod.compute_config_hash(config, str(suite_path))
        state = RunState(
            run_id="r_20260601_dead00",
            status="in_progress",
            config_hash=config_hash,
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
            models=RunModels(
                source="gemini/gemini-2.5-flash",
                target="gemini/gemini-2.5-pro",
            ),
            prompt_ids=["greet"],
            suite_path=str(suite_path),
            total_evaluations=4,
            completed_evaluations=2,
        )
        cp_mod.write_state(run_dir, state)
        for ex_id, role, model in [
            ("ex0", "source", "gemini/gemini-2.5-flash"),
            ("ex0", "target", "gemini/gemini-2.5-pro"),
        ]:
            cp_mod.append_call(
                run_dir,
                Call(
                    run_id="r_20260601_dead00",
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id=model,
                    role=role,  # type: ignore[arg-type]
                    text="from previous run",
                ),
            )

        # Now resume. Only the *remaining* 2 calls should hit the fake
        # client.
        counter = _make_fake_client(monkeypatch)
        result = await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            resume=True,
            yes=True,
            cache=cache,
        )

        assert counter["calls"] == 2  # exactly the missing pair
        assert result.completed_calls == 4
        rows = list(iter_calls(result.run_dir))
        assert len(rows) == 4

    async def test_resume_aborts_when_config_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        from datetime import datetime

        from evalshift.runner import checkpoint as cp_mod
        from evalshift.runner.models import RunModels, RunState

        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        # Pre-existing run started against a different config hash.
        run_dir = runs_base / "r_20260601_dead00"
        cp_mod.write_state(
            run_dir,
            RunState(
                run_id="r_20260601_dead00",
                status="in_progress",
                config_hash="totally_different",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["greet"],
                suite_path=str(suite_path),
                total_evaluations=4,
            ),
        )

        from evalshift.runner.checkpoint import CheckpointError

        with pytest.raises(CheckpointError, match="config or suite has changed"):
            await run_orchestrator(
                config=_config(),
                config_path=config_path,
                suite=_suite(n=2),
                suite_path=suite_path,
                source_model="gemini-2.5-flash",
                target_model="gemini-2.5-pro",
                runs_base=runs_base,
                resume=True,
                yes=True,
                cache=cache,
            )

    async def test_resume_with_no_in_progress_aborts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        with pytest.raises(RunAborted, match="no in-progress run"):
            await run_orchestrator(
                config=_config(),
                config_path=config_path,
                suite=_suite(n=2),
                suite_path=suite_path,
                source_model="gemini-2.5-flash",
                target_model="gemini-2.5-pro",
                runs_base=runs_base,
                resume=True,
                yes=True,
                cache=cache,
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOrchestratorErrors:
    async def test_failed_calls_recorded_with_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        # Fail every target call with an auth error; source calls succeed.
        _make_fake_client(
            monkeypatch,
            raise_for={"gemini/gemini-2.5-pro": AuthError("bad key")},
        )
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=2),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        # The run still completes; failed calls are recorded with an error.
        assert result.completed_calls == 4
        assert result.failed_calls == 2
        assert result.live_calls == 2  # only source calls succeeded
        rows = list(iter_calls(result.run_dir))
        target_rows = [r for r in rows if r.role == "target"]
        assert all(r.error is not None for r in target_rows)
        # Final state still completed (the orchestrator doesn't fail the
        # whole run on a per-call error — Phase 5 evaluators handle errors).
        assert read_state(result.run_dir).status == "completed"


# ---------------------------------------------------------------------------
# Cost gating
# ---------------------------------------------------------------------------


class TestOrchestratorCostGate:
    async def test_yes_flag_skips_confirmation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        # Even with --yes, the run should proceed without input.
        # This test would hang if --yes wasn't honoured.
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=2),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )
        assert result.completed_calls == 4


# ---------------------------------------------------------------------------
# Sanity: the constants the docstring promises
# ---------------------------------------------------------------------------


class TestConstants:
    def test_checkpoint_every_is_50(self) -> None:
        # PDF §4.3 calls for "Checkpoint every 50 calls".
        assert CHECKPOINT_EVERY == 50


# ---------------------------------------------------------------------------
# build_messages — pure helper
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_none_for_single_turn_example(self) -> None:
        example = suite_example(id="ex0", inputs={"name": "Alex"})
        assert build_messages(example, "Hello Alex") is None

    def test_prefix_plus_current_turn_for_multi_turn_example(self) -> None:
        example = suite_example(
            id="ex0",
            inputs={"name": "Alex"},
            history=[
                ChatMessage(role="system", content="be terse"),
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            ],
        )
        messages = build_messages(example, "what's next?")
        assert messages == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what's next?"},
        ]

    def test_emits_openai_shaped_tool_turns(self) -> None:
        example = suite_example(
            id="ex0",
            history=[
                ChatMessage(role="system", content="You are Kaila."),
                ChatMessage(role="user", content="list my projects"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        HistoryToolCall(
                            id="c1",
                            name="get_projects",
                            arguments={"status": "active"},
                        ),
                    ],
                ),
                ChatMessage(role="tool", tool_call_id="c1", content='{"projects": []}'),
            ],
        )

        messages = build_messages(example, "yes")
        assert messages is not None
        assert messages[2] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_projects", "arguments": '{"status": "active"}'},
                },
            ],
        }
        assert messages[3] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": '{"projects": []}',
        }
        assert messages[-1] == {"role": "user", "content": "yes"}

    def test_tool_call_without_a_recorded_id_gets_a_positional_one(self) -> None:
        example = suite_example(
            id="ex0",
            history=[
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[HistoryToolCall(name="get_projects", arguments={})],
                ),
            ],
        )
        messages = build_messages(example, "yes")
        assert messages is not None
        assert messages[0]["tool_calls"][0]["id"] == "call_0"

    def test_cache_key_history_keeps_the_pre_tool_shape_for_plain_turns(self) -> None:
        """A text-only prefix must key exactly as it did before tool fields existed.

        Otherwise adding the fields silently invalidates every cached
        multi-turn response.
        """
        example = suite_example(
            id="ex0",
            history=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            ],
        )
        assert history_for_cache_key(example) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_cache_key_history_includes_recorded_tool_calls(self) -> None:
        example = suite_example(
            id="ex0",
            history=[
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[HistoryToolCall(id="c1", name="get_projects", arguments={})],
                ),
            ],
        )
        keyed = history_for_cache_key(example)
        assert keyed == [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "get_projects", "arguments": {}}],
            },
        ]

    def test_empty_list_history_is_not_none_and_is_single_element(self) -> None:
        example = suite_example(id="ex0", inputs={}, history=[])
        messages = build_messages(example, "current turn")
        assert messages is not None
        assert messages == [{"role": "user", "content": "current turn"}]


# ---------------------------------------------------------------------------
# Dispatch routing — single-turn vs. multi-turn, with/without tools
# ---------------------------------------------------------------------------


class TestOrchestratorDispatchRouting:
    def _suite_with_history(self, *, tools: list[ToolSpec] | None = None) -> Suite:
        extra: dict[str, Any] = {"tools": tools} if tools is not None else {}
        return Suite(
            examples=[
                suite_example(
                    id="ex0",
                    inputs={"name": "Alex"},
                    history=[
                        ChatMessage(role="user", content="earlier turn"),
                        ChatMessage(role="assistant", content="earlier reply"),
                    ],
                    **extra,
                ),
            ],
        )

    async def test_single_turn_example_dispatches_via_complete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        calls: dict[str, int] = {"complete": 0, "complete_messages": 0}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete"] += 1
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        async def fake_complete_messages(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete_messages"] += 1
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        monkeypatch.setattr(ModelClient, "complete_messages", fake_complete_messages)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(n=1),  # single-turn examples
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert calls["complete"] == 2  # source + target
        assert calls["complete_messages"] == 0

    async def test_multi_turn_example_dispatches_via_complete_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        calls: dict[str, int] = {"complete": 0, "complete_messages": 0}
        seen_messages: list[list[dict[str, Any]]] = []

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete"] += 1
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        async def fake_complete_messages(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete_messages"] += 1
            seen_messages.append(list(kwargs["messages"]))
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        monkeypatch.setattr(ModelClient, "complete_messages", fake_complete_messages)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=self._suite_with_history(),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert calls["complete"] == 0
        assert calls["complete_messages"] == 2  # source + target
        expected_prefix = [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier reply"},
        ]
        for messages in seen_messages:
            assert messages[:2] == expected_prefix
            assert messages[2] == {"role": "user", "content": "Hello Alex"}

    async def test_multi_turn_with_tools_dispatches_via_complete_messages_with_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        calls: dict[str, int] = {"complete_with_tools": 0, "complete_messages_with_tools": 0}
        seen_messages: list[list[dict[str, Any]]] = []

        trace = ToolTrace(calls=[], final_text="ok")

        async def fake_complete_with_tools(
            self: ModelClient, **kwargs: Any
        ) -> ToolCompletionResult:
            calls["complete_with_tools"] += 1
            return ToolCompletionResult(
                trace=trace,
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                raw_provider_response={},
            )

        async def fake_complete_messages_with_tools(
            self: ModelClient, **kwargs: Any
        ) -> ToolCompletionResult:
            calls["complete_messages_with_tools"] += 1
            seen_messages.append(list(kwargs["messages"]))
            return ToolCompletionResult(
                trace=trace,
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                raw_provider_response={},
            )

        monkeypatch.setattr(ModelClient, "complete_with_tools", fake_complete_with_tools)
        monkeypatch.setattr(
            ModelClient, "complete_messages_with_tools", fake_complete_messages_with_tools
        )

        tool_spec = ToolSpec(
            name="lookup",
            description="Look something up.",
            input_schema={"type": "object", "properties": {}},
        )
        config = EvalShiftConfig(
            prompts=[
                PromptDefinition(
                    id="agent",
                    detection="manual",
                    content="Hello {name}",
                    variables=["name"],
                ),
            ],
            defaults=Defaults(concurrency=4, max_cost_usd=100.0),
        )

        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=self._suite_with_history(tools=[tool_spec]),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert calls["complete_with_tools"] == 0
        assert calls["complete_messages_with_tools"] == 2  # source + target
        expected_prefix = [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "earlier reply"},
        ]
        for messages in seen_messages:
            assert messages[:2] == expected_prefix
            assert messages[2] == {"role": "user", "content": "Hello Alex"}

    async def test_example_with_empty_tools_dispatches_plain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        """The bug this task fixes: dispatch is sourced from the example, not the prompt.

        The prompt carries no toolset configuration at all -- there is no more
        config-level notion of "agent-style" to fight with -- and the example's
        own toolset is explicitly empty (``tools: []``). This must dispatch via
        the plain ``complete`` path, never ``complete_with_tools``.
        """
        calls: dict[str, int] = {"complete": 0, "complete_with_tools": 0}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete"] += 1
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        async def fake_complete_with_tools(
            self: ModelClient, **kwargs: Any
        ) -> ToolCompletionResult:
            calls["complete_with_tools"] += 1
            return ToolCompletionResult(
                trace=ToolTrace(calls=[], final_text="ok"),
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                raw_provider_response={},
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        monkeypatch.setattr(ModelClient, "complete_with_tools", fake_complete_with_tools)

        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        config = EvalShiftConfig(
            prompts=[
                PromptDefinition(
                    id="agent",
                    detection="manual",
                    content="Hello {name}",
                    variables=["name"],
                ),
            ],
            defaults=Defaults(concurrency=4, max_cost_usd=100.0),
        )
        suite = Suite(examples=[suite_example(id="ex0", inputs={"name": "Alex"}, tools=[])])

        result = await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert calls["complete"] == 2  # source + target
        assert calls["complete_with_tools"] == 0
        rows = list(iter_calls(result.run_dir))
        assert len(rows) == 2
        assert all(r.trace is None for r in rows)

    async def test_mixed_toolsets_within_one_suite_dispatch_differently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        """One suite, two examples, two toolsets -- routed independently, one run.

        Example A carries the empty toolset (asserts "no tools offered"); example
        B carries two tools. Both dispatch under the SAME prompt -- proving the
        toolset comes from each example, not from any prompt-level config (there
        is no such config any more). This is the exact scenario the task exists
        for: one suite, per-call toolsets chosen by the example, not the prompt.
        """
        calls: dict[str, int] = {"complete": 0, "complete_with_tools": 0}
        seen_tool_names: list[list[str]] = []

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls["complete"] += 1
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        async def fake_complete_with_tools(
            self: ModelClient, **kwargs: Any
        ) -> ToolCompletionResult:
            calls["complete_with_tools"] += 1
            seen_tool_names.append(sorted(t.name for t in kwargs["tools"]))
            return ToolCompletionResult(
                trace=ToolTrace(calls=[], final_text="ok"),
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                raw_provider_response={},
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        monkeypatch.setattr(ModelClient, "complete_with_tools", fake_complete_with_tools)

        tool_a = ToolSpec(
            name="search",
            description="Search.",
            input_schema={"type": "object", "properties": {}},
        )
        tool_b = ToolSpec(
            name="notify",
            description="Notify.",
            input_schema={"type": "object", "properties": {}},
        )

        suite = Suite(
            examples=[
                suite_example(id="ex_no_tools", inputs={"name": "Alex"}, tools=[]),
                suite_example(id="ex_two_tools", inputs={"name": "Sam"}, tools=[tool_a, tool_b]),
            ],
        )
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        result = await run_orchestrator(
            config=_config(),  # single "greet" prompt
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
        )

        assert result.failed_calls == 0
        assert calls["complete"] == 2  # ex_no_tools: source + target
        assert calls["complete_with_tools"] == 2  # ex_two_tools: source + target
        assert seen_tool_names == [["notify", "search"], ["notify", "search"]]

        rows = list(iter_calls(result.run_dir))
        no_tools_rows = [r for r in rows if r.example_id == "ex_no_tools"]
        two_tools_rows = [r for r in rows if r.example_id == "ex_two_tools"]
        assert len(no_tools_rows) == 2
        assert len(two_tools_rows) == 2
        assert all(r.trace is None for r in no_tools_rows)
        assert all(r.trace is not None for r in two_tools_rows)

    async def test_checkpoint_resume_keyed_correctly_with_multi_turn_examples(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        """Resume must skip already-completed multi-turn calls by
        (prompt_id, example_id, role) — the same key as single-turn.

        Mirrors ``TestOrchestratorResume.test_resume_picks_up_after_partial_progress``:
        manually seed an in-progress run whose ``source`` call for the
        multi-turn example is already recorded, then resume and confirm
        only the missing ``target`` call is dispatched.
        """
        from datetime import datetime

        from evalshift.runner import checkpoint as cp_mod
        from evalshift.runner.models import Call, RunModels, RunState

        config = _config()
        suite = self._suite_with_history()
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)

        run_dir = runs_base / "r_20260601_dead00"
        config_hash = cp_mod.compute_config_hash(config, str(suite_path))
        state = RunState(
            run_id="r_20260601_dead00",
            status="in_progress",
            config_hash=config_hash,
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
            models=RunModels(
                source="gemini/gemini-2.5-flash",
                target="gemini/gemini-2.5-pro",
            ),
            prompt_ids=["greet"],
            suite_path=str(suite_path),
            total_evaluations=2,
            completed_evaluations=1,
        )
        cp_mod.write_state(run_dir, state)
        cp_mod.append_call(
            run_dir,
            Call(
                run_id="r_20260601_dead00",
                prompt_id="greet",
                example_id="ex0",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="from previous run",
            ),
        )

        seen_messages: list[list[dict[str, Any]]] = []

        async def fake_complete_messages(self: ModelClient, **kwargs: Any) -> CompletionResult:
            seen_messages.append(list(kwargs["messages"]))
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete_messages", fake_complete_messages)

        result = await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            resume=True,
            yes=True,
            cache=cache,
        )

        # Only the missing "target" call should have been dispatched.
        assert len(seen_messages) == 1
        assert result.completed_calls == 2
        rows = list(iter_calls(result.run_dir))
        assert len(rows) == 2
        roles = {r.role for r in rows}
        assert roles == {"source", "target"}


def test_build_work_list_requires_toolset_bases() -> None:
    """M3: ``toolset_bases`` used to default to ``()``, in a path whose whole
    premise is "never default" -- run_orchestrator always computes real
    candidates via ``toolset_base_candidates`` (never empty) before calling
    this. The one caller already passes it explicitly, so making it required
    keyword-only costs nothing and makes an accidentally-omitted argument a
    loud TypeError instead of a silent, wrong empty-candidates run.
    """
    with pytest.raises(TypeError):
        _build_work_list(  # type: ignore[call-arg]
            templates=[],
            suite=Suite(),
            canonical_source="src",
            canonical_target="tgt",
        )


# ---------------------------------------------------------------------------
# resolve_example_tools / toolset_base_candidates — base-path resolution
# ---------------------------------------------------------------------------


class TestResolveExampleTools:
    """Direct tests of ``resolve_example_tools`` / ``toolset_base_candidates``.

    These pin the base-path resolution tiers a real run computes once (via
    ``toolset_base_candidates``) and threads into every example's
    ``resolve_example_tools`` call. The sharp edge: a ``toolset_ref`` sidecar for
    a checked-in example suite is NOT under ``.evalshift/`` (gitignored, can't
    hold committed content), and getting the base wrong makes every shipped
    example unrunnable -- see ``TestOrchestratorShippedExamples`` below for the
    real-fixture proof.
    """

    @staticmethod
    def _tool_a() -> ToolSpec:
        return ToolSpec(
            name="search",
            description="Search.",
            input_schema={"type": "object", "properties": {}},
        )

    @staticmethod
    def _write_sidecar(base: Path, tools: list[ToolSpec]) -> str:
        """Write a real toolset sidecar under ``base/toolsets/`` and return its ref."""
        ref = fingerprint_tools([t.to_anthropic() for t in tools])
        path = base / "toolsets" / f"{ref.removeprefix('sha256:')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tools": [t.to_anthropic() for t in tools]}))
        return ref

    def test_inline_tools_returned_directly_no_base_needed(self) -> None:
        tool = self._tool_a()
        example = suite_example(id="ex0", inputs={}, tools=[tool])
        result = resolve_example_tools(example, toolset_bases=(), toolset_cache={})
        assert result == (tool,)

    def test_empty_inline_tools_returns_empty_tuple(self) -> None:
        example = suite_example(id="ex0", inputs={}, tools=[])
        result = resolve_example_tools(example, toolset_bases=(), toolset_cache={})
        assert result == ()

    def test_ref_resolves_from_suite_directory(self, tmp_path: Path) -> None:
        """Mimics ``examples/agent/``: sidecar colocated with the suite file itself."""
        suite_dir = tmp_path / "agent"
        suite_dir.mkdir()
        tool = self._tool_a()
        ref = self._write_sidecar(suite_dir, [tool])
        example = suite_example(id="ex0", inputs={}, toolset_ref=ref)

        bases = toolset_base_candidates(suite_path=suite_dir / "golden.jsonl")
        result = resolve_example_tools(example, toolset_bases=bases, toolset_cache={})
        assert [t.name for t in result] == ["search"]

    def test_unresolvable_ref_raises_capture_error_naming_every_tried_location(
        self,
        tmp_path: Path,
    ) -> None:
        example = suite_example(id="ex0", inputs={}, toolset_ref="sha256:" + "0" * 64)
        bases = (tmp_path / "a", tmp_path / "b")

        with pytest.raises(CaptureError) as exc_info:
            resolve_example_tools(example, toolset_bases=bases, toolset_cache={})

        assert exc_info.value.kind == "missing"
        message = str(exc_info.value)
        assert str(tmp_path / "a") in message
        assert str(tmp_path / "b") in message

    def test_resolution_cached_across_examples_sharing_one_ref(self, tmp_path: Path) -> None:
        """A sidecar shared by many examples is read from disk at most once per run."""
        tool = self._tool_a()
        ref = self._write_sidecar(tmp_path, [tool])
        toolset_cache: dict[str, list[ToolSpec]] = {}

        first = resolve_example_tools(
            suite_example(id="ex0", inputs={}, toolset_ref=ref),
            toolset_bases=(tmp_path,),
            toolset_cache=toolset_cache,
        )
        assert [t.name for t in first] == ["search"]

        # Delete the sidecar: a second resolution, for a DIFFERENT example
        # sharing the same ref, must still succeed -- proving it came from the
        # cache and never touched disk again.
        (tmp_path / "toolsets" / f"{ref.removeprefix('sha256:')}.json").unlink()
        second = resolve_example_tools(
            suite_example(id="ex1", inputs={}, toolset_ref=ref),
            toolset_bases=(tmp_path,),
            toolset_cache=toolset_cache,
        )
        assert second == first

    def test_inline_and_ref_to_same_tools_fingerprint_identically(self, tmp_path: Path) -> None:
        """The two spellings of one toolset must key the cache identically."""
        tool = self._tool_a()
        ref = self._write_sidecar(tmp_path, [tool])

        inline_example = suite_example(id="ex_inline", inputs={}, tools=[tool])
        ref_example = suite_example(id="ex_ref", inputs={}, toolset_ref=ref)

        inline_tools = resolve_example_tools(inline_example, toolset_bases=(), toolset_cache={})
        ref_tools = resolve_example_tools(
            ref_example,
            toolset_bases=(tmp_path,),
            toolset_cache={},
        )

        assert _fingerprint_toolset(inline_tools) == _fingerprint_toolset(ref_tools)


# ---------------------------------------------------------------------------
# Shipped example suites — real end-to-end proof of the base-path plumbing
# ---------------------------------------------------------------------------


class TestOrchestratorShippedExamples:
    """Every real, checked-in example suite must still run end-to-end.

    ``examples/agent/`` carries a real ``toolset_ref`` backed by a committed
    sidecar that lives OUTSIDE ``.evalshift/`` (gitignored, can't hold
    committed content) -- see ``toolset_base_candidates``. This is the best
    end-to-end proof that base-path plumbing is right: get it wrong and this
    raises ``CaptureError`` instead of dispatching, per the Task 8 brief.

    Network is mocked (``ModelClient.complete`` / ``complete_with_tools``);
    ``runs_base`` is redirected to ``tmp_path`` so this never writes into the
    committed ``examples/`` tree.
    """

    # Repo root: tests/unit/test_orchestrator.py -> tests/unit -> tests -> <root>.
    # A direct path, never a broad rglob, so this can't wander into
    # .claude/worktrees/ (a stale full copy of the repo) -- same guard as
    # test_suite_loader.py's TestLoadJsonlCheckedInExamples.
    _EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

    @staticmethod
    def _install_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0001,
                latency_ms=1,
            )

        async def fake_complete_with_tools(
            self: ModelClient, **kwargs: Any
        ) -> ToolCompletionResult:
            return ToolCompletionResult(
                trace=ToolTrace(calls=[], final_text="ok"),
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0001,
                latency_ms=1,
                raw_provider_response={},
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)
        monkeypatch.setattr(ModelClient, "complete_with_tools", fake_complete_with_tools)

    async def _run_example_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
        example_dir: Path,
    ) -> RunResult:
        self._install_fake_client(monkeypatch)
        config_path = example_dir / "evalshift.yaml"
        suite_path = example_dir / "golden.jsonl"
        config = load_config(config_path)
        suite = load_jsonl(suite_path)
        assert config.defaults.source_model is not None
        assert config.defaults.target_model is not None
        return await run_orchestrator(
            config=config,
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model=config.defaults.source_model,
            target_model=config.defaults.target_model,
            runs_base=tmp_path / "runs",
            yes=True,
            cache=cache,
        )

    async def test_agent_example_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        result = await self._run_example_dir(
            monkeypatch,
            tmp_path,
            cache,
            self._EXAMPLES_DIR / "agent",
        )
        assert result.total_calls > 0
        assert result.completed_calls == result.total_calls
        assert result.failed_calls == 0

    async def test_simple_example_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        result = await self._run_example_dir(
            monkeypatch,
            tmp_path,
            cache,
            self._EXAMPLES_DIR / "simple",
        )
        assert result.total_calls > 0
        assert result.completed_calls == result.total_calls
        assert result.failed_calls == 0


class TestRuntimeTemperatureRejectionReporting:
    async def test_rejected_models_land_in_final_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        _make_fake_client(monkeypatch)
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        client = ModelClient()
        # White-box seed: simulates a mid-run provider rejection without
        # needing a live 400 through the faked complete().
        client._temperature_rejected.add("openai/gpt-5.6-terra")

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
            client=client,
        )

        state = read_state(result.run_dir)
        assert "openai/gpt-5.6-terra" in state.non_deterministic_models

    async def test_merge_deduplicates_against_probe_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        cache: CacheStore,
    ) -> None:
        from evalshift.runner import orchestrator as orch_mod

        _make_fake_client(monkeypatch)
        monkeypatch.setattr(
            orch_mod,
            "detect_non_deterministic_models",
            lambda *, source, target: ["openai/gpt-5.6-terra"],
        )
        config_path, suite_path, runs_base = _writeable_paths(tmp_path)
        client = ModelClient()
        client._temperature_rejected.add("openai/gpt-5.6-terra")

        result = await run_orchestrator(
            config=_config(),
            config_path=config_path,
            suite=_suite(),
            suite_path=suite_path,
            source_model="gemini-2.5-flash",
            target_model="gemini-2.5-pro",
            runs_base=runs_base,
            yes=True,
            cache=cache,
            client=client,
        )

        state = read_state(result.run_dir)
        assert state.non_deterministic_models.count("openai/gpt-5.6-terra") == 1
