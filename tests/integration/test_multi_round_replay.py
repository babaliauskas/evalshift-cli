"""End-to-end: a teacher-forced multi-round replay through ``run_orchestrator``.

The unit tests in ``tests/unit/test_orchestrator.py`` pin the loop's mechanics
against a hand-rolled fake client. This test proves the same thing one layer
out, against artefacts rather than mocks:

1. A golden suite is written to disk as JSONL and loaded back through
   ``load_jsonl``, so ``tool_result_fixtures`` survive a real serialisation
   round-trip — the same file ``capture sync`` writes.
2. ``run_orchestrator`` replays it with the shared ``ReplayClient`` test double
   (``tests/integration/replay_client.py``), whose fixtures are pinned per
   round: the same rendered prompt is dispatched three times, and only the
   round number tells the three apart. That is the load-bearing check on the
   client's new round matching, and it is also how the target model is made to
   diverge in round 2 while agreeing in round 1.
3. The recorded ``raw.jsonl`` is read back and asserted on: one row per role
   (never one per round), a merged three-round trace, summed tokens, and the
   *last* round's text.
4. A subclassed client records the exact ``messages`` list every dispatch
   received, so the assertions can show what a candidate actually sees in
   round 2 and round 3 — the recorded assistant ``tool_calls`` with their
   positional ``call_r{j}_{i}`` ids and the rendered tool results — rather
   than only that the run completed.

A single-shot example rides along in the same suite to prove the loop is opt-in:
it must still make exactly one call, through the plain-prompt entry point, with
the same arguments it made before the loop existed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evalshift_cli.config.loader import load_config
from evalshift_cli.models.client import ToolCompletionResult
from evalshift_cli.runner.checkpoint import iter_calls
from evalshift_cli.runner.orchestrator import run_orchestrator
from evalshift_cli.suite.loader import load_jsonl
from evalshift_cli.suite.models import (
    ExpectedToolCall,
    SuiteExample,
    ToolResultFixture,
)
from tests.integration.replay_client import ReplayClient

pytestmark = pytest.mark.integration

SOURCE_MODEL = "gemini/gemini-2.5-flash"
TARGET_MODEL = "gemini/gemini-2.5-pro"

_LOOP_QUESTION = "refund order 42"
_SOLO_QUESTION = "what are your hours"

_CONFIG = """
version: 1

prompts:
  - id: agent
    detection: manual
    content: "Answer: {question}"
    variables: [question]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  concurrency: 2
  cache: false

evaluators:
  structural:
    - type: regex
      pattern: "Refunded"
"""

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_orders",
        "description": "Look up an order.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
    {
        "name": "issue_refund",
        "description": "Refund an order.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
    {
        "name": "notify_user",
        "description": "Send the user a message.",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
]


class RecordingReplayClient(ReplayClient):
    """``ReplayClient`` that also keeps the exact arguments of every dispatch."""

    def __init__(self, fixtures_path: Path) -> None:
        super().__init__(fixtures_path)
        self.dispatches: list[dict[str, Any]] = []

    async def complete_with_tools(self, **kwargs: Any) -> ToolCompletionResult:
        self.dispatches.append(
            {"model": kwargs["model"], "prompt": kwargs["prompt"], "messages": None},
        )
        return await super().complete_with_tools(**kwargs)

    async def complete_messages_with_tools(self, **kwargs: Any) -> ToolCompletionResult:
        self.dispatches.append(
            {
                "model": kwargs["model"],
                "prompt": None,
                "messages": [dict(m) for m in kwargs["messages"]],
            },
        )
        return await super().complete_messages_with_tools(**kwargs)


def _write_suite(tmp_path: Path) -> Path:
    """A two-example golden suite: one teacher-forced loop, one single-shot."""
    loop = SuiteExample(
        id="ex_loop",
        inputs={"question": _LOOP_QUESTION},
        tools=_TOOLS,
        expected_tool_rounds=[
            [ExpectedToolCall(tool_name="search_orders", arguments={"order_id": "42"})],
            [ExpectedToolCall(tool_name="issue_refund", arguments={"order_id": "42"})],
        ],
        tool_result_fixtures=[
            [
                ToolResultFixture(
                    tool_name="search_orders",
                    result={"order_id": "42", "total": 19.99},
                ),
            ],
            [ToolResultFixture(tool_name="issue_refund", result="refund accepted")],
        ],
    )
    solo = SuiteExample(
        id="ex_solo",
        inputs={"question": _SOLO_QUESTION},
        tools=_TOOLS,
    )
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(e.model_dump_json() for e in (loop, solo)) + "\n",
        encoding="utf-8",
    )
    return path


def _tools_fixture(
    *,
    model: str,
    match: str,
    calls: list[dict[str, Any]],
    final_text: str | None,
    round_index: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model": model,
        "match": match,
        "kind": "tools",
        "result": {
            "calls": calls,
            "final_text": final_text,
            "input_tokens": 10,
            "output_tokens": 4,
            "cost_usd": 0.25,
            "latency_ms": 7,
        },
    }
    if round_index is not None:
        record["round"] = round_index
    return record


def _write_fixtures(tmp_path: Path) -> Path:
    """Per-round replay fixtures; the target diverges in round 2 only."""
    search = [{"tool_name": "search_orders", "arguments": {"order_id": "42"}}]
    refund = [{"tool_name": "issue_refund", "arguments": {"order_id": "42"}}]
    notify = [{"tool_name": "notify_user", "arguments": {"text": "we'll look into it"}}]
    records = [
        # Round 1 — both models agree.
        _tools_fixture(
            model=SOURCE_MODEL, match=_LOOP_QUESTION, calls=search, final_text=None, round_index=0
        ),
        _tools_fixture(
            model=TARGET_MODEL, match=_LOOP_QUESTION, calls=search, final_text=None, round_index=0
        ),
        # Round 2 — the regression: the target notifies instead of refunding.
        _tools_fixture(
            model=SOURCE_MODEL, match=_LOOP_QUESTION, calls=refund, final_text=None, round_index=1
        ),
        _tools_fixture(
            model=TARGET_MODEL, match=_LOOP_QUESTION, calls=notify, final_text=None, round_index=1
        ),
        # Round 3 — the answer round: no calls, just the final text.
        _tools_fixture(
            model=SOURCE_MODEL,
            match=_LOOP_QUESTION,
            calls=[],
            final_text="Refunded order 42.",
            round_index=2,
        ),
        _tools_fixture(
            model=TARGET_MODEL,
            match=_LOOP_QUESTION,
            calls=[],
            final_text="I let them know we're looking into it.",
            round_index=2,
        ),
        # The single-shot example: no "round" key at all, exactly as every
        # fixture written before rounds existed.
        _tools_fixture(
            model=SOURCE_MODEL, match=_SOLO_QUESTION, calls=[], final_text="We're open 9-5."
        ),
        _tools_fixture(
            model=TARGET_MODEL, match=_SOLO_QUESTION, calls=[], final_text="We're open 9-5."
        ),
    ]
    path = tmp_path / "fixtures.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_teacher_forced_replay_runs_one_call_per_round(tmp_path: Path) -> None:
    config_path = tmp_path / "evalshift.yaml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    suite_path = _write_suite(tmp_path)
    fixtures_path = _write_fixtures(tmp_path)

    suite = load_jsonl(suite_path)
    loop_example = next(e for e in suite.examples if e.id == "ex_loop")
    solo_example = next(e for e in suite.examples if e.id == "ex_solo")
    # Two covered tool rounds plus the answer round after them.
    assert loop_example.rounds_to_replay() == 3
    assert solo_example.rounds_to_replay() == 1

    client = RecordingReplayClient(fixtures_path)
    result = asyncio.run(
        run_orchestrator(
            config=load_config(config_path),
            config_path=config_path,
            suite=suite,
            suite_path=suite_path,
            source_model=SOURCE_MODEL,
            target_model=TARGET_MODEL,
            runs_base=tmp_path / "runs",
            yes=True,
            client=client,
        ),
    )

    # ---- raw.jsonl: one row per (example, role), never one per round -------
    rows = list(iter_calls(result.run_dir))
    assert len(rows) == 4
    by_key = {(r.example_id, r.role): r for r in rows}
    assert set(by_key) == {
        ("ex_loop", "source"),
        ("ex_loop", "target"),
        ("ex_solo", "source"),
        ("ex_solo", "target"),
    }

    loop_source = by_key[("ex_loop", "source")]
    loop_target = by_key[("ex_loop", "target")]
    for row in (loop_source, loop_target):
        assert row.error is None
        assert row.trace is not None
        assert row.trace.round_count == 3
        # 3 rounds x the per-round fixture numbers.
        assert row.input_tokens == 30
        assert row.output_tokens == 12
        assert row.cost_usd == pytest.approx(0.75)
        assert row.latency_ms == 21

    # ---- the merged trace: round tags, and the last round's text ----------
    assert loop_source.trace is not None
    assert [(c.round_index, c.tool_name) for c in loop_source.trace.calls] == [
        (0, "search_orders"),
        (1, "issue_refund"),
    ]
    assert [c.sequence_index for c in loop_source.trace.calls] == [0, 1]
    assert loop_source.text == "Refunded order 42."

    assert loop_target.trace is not None
    assert [(c.round_index, c.tool_name) for c in loop_target.trace.calls] == [
        (0, "search_orders"),
        (1, "notify_user"),
    ]
    assert loop_target.text == "I let them know we're looking into it."

    # Round 1 agrees, round 2 diverges — the whole point of per-round replay.
    source_rounds = loop_source.trace.rounds()
    target_rounds = loop_target.trace.rounds()
    assert [c.tool_name for c in source_rounds[0].calls] == [
        c.tool_name for c in target_rounds[0].calls
    ]
    assert [c.tool_name for c in source_rounds[1].calls] == ["issue_refund"]
    assert [c.tool_name for c in target_rounds[1].calls] == ["notify_user"]
    assert source_rounds[2].calls == [] and target_rounds[2].calls == []

    # ---- what the candidate actually saw, round by round ------------------
    loop_dispatches = [
        d
        for d in client.dispatches
        if d["model"] == SOURCE_MODEL
        and (d["prompt"] == f"Answer: {_LOOP_QUESTION}" or d["messages"] is not None)
    ]
    assert len(loop_dispatches) == 3

    # Round 1: the plain-prompt path, byte-identical to a single-shot dispatch.
    assert loop_dispatches[0]["messages"] is None
    assert loop_dispatches[0]["prompt"] == f"Answer: {_LOOP_QUESTION}"

    # Round 2: the prompt plus round 1's RECORDED call and its recorded result.
    assert loop_dispatches[1]["messages"] == [
        {"role": "user", "content": f"Answer: {_LOOP_QUESTION}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_r0_0",
                    "type": "function",
                    "function": {
                        "name": "search_orders",
                        "arguments": '{"order_id": "42"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_r0_0",
            "content": '{"order_id": "42", "total": 19.99}',
        },
    ]

    # Round 3: rounds 1 AND 2, both from the recording — never the candidate's
    # own round-2 call (the target called notify_user and still sees
    # issue_refund here, which is what makes the two sides comparable).
    round3 = loop_dispatches[2]["messages"]
    assert round3[:3] == loop_dispatches[1]["messages"]
    assert round3[3] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_r1_0",
                "type": "function",
                "function": {"name": "issue_refund", "arguments": '{"order_id": "42"}'},
            },
        ],
    }
    # A str result is fed back verbatim, not double-encoded as JSON.
    assert round3[4] == {
        "role": "tool",
        "tool_call_id": "call_r1_0",
        "content": "refund accepted",
    }
    assert len(round3) == 5

    target_round3 = next(
        d["messages"]
        for d in client.dispatches
        if d["model"] == TARGET_MODEL and d["messages"] is not None and len(d["messages"]) == 5
    )
    assert target_round3 == round3

    # ---- the single-shot example is untouched by any of this --------------
    solo_dispatches = [d for d in client.dispatches if d["prompt"] == f"Answer: {_SOLO_QUESTION}"]
    assert len(solo_dispatches) == 2  # source + target, one call each
    assert all(d["messages"] is None for d in solo_dispatches)
    assert not [
        d
        for d in client.dispatches
        if d["messages"] is not None and _SOLO_QUESTION in json.dumps(d["messages"])
    ]
    for role in ("source", "target"):
        solo_row = by_key[("ex_solo", role)]
        assert solo_row.trace is not None
        assert solo_row.trace.round_count == 1
        assert solo_row.input_tokens == 10  # one round's worth
        assert solo_row.text == "We're open 9-5."
