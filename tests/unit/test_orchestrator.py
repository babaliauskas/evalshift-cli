"""Tests for :mod:`aimigrate.runner.orchestrator`.

We exercise the orchestrator end-to-end with:

* A real :class:`AIMigrateConfig` + :class:`Suite` constructed in-memory.
* A real :class:`CacheStore` backed by ``sqlite+aiosqlite:///:memory:``.
* A *fake* :class:`ModelClient` whose ``complete`` method returns
  deterministic responses without touching the network.

These tests are the closest thing to an end-to-end integration check we
can run in unit-test scope. The orchestrator's CLI-level entry point is
covered separately in ``tests/unit/test_run_command.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from aimigrate.cache.store import CacheStore
from aimigrate.config.models import (
    AIMigrateConfig,
    Defaults,
    PromptDefinition,
)
from aimigrate.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
)
from aimigrate.runner.checkpoint import (
    iter_calls,
    read_state,
)
from aimigrate.runner.orchestrator import (
    CHECKPOINT_EVERY,
    RunAborted,
    run_orchestrator,
)
from aimigrate.suite.models import Suite, SuiteExample

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


def _config(*, concurrency: int = 4, max_cost: float = 100.0) -> AIMigrateConfig:
    return AIMigrateConfig(
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
        examples=[SuiteExample(id=f"ex{i}", inputs={"name": f"User{i}"}) for i in range(n)],
    )


def _writeable_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(config_path, suite_path, runs_base)`` rooted in ``tmp_path``."""
    config_path = tmp_path / "aimigrate.yaml"
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

        from aimigrate.runner import checkpoint as cp_mod
        from aimigrate.runner.models import Call, RunModels, RunState

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

        from aimigrate.runner import checkpoint as cp_mod
        from aimigrate.runner.models import RunModels, RunState

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

        from aimigrate.runner.checkpoint import CheckpointError

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
