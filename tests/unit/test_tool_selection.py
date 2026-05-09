"""Tests for :class:`evalshift.evaluators.tool_selection.ToolSelectionEvaluator`."""

from __future__ import annotations

import pytest

from evalshift.config.models import ToolSelectionEvaluatorConfig
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.evaluators.tool_selection import (
    ToolSelectionEvaluator,
    _jaccard,
    _sequence_match,
)
from evalshift.suite.models import ExpectedToolCall, SuiteExample


def _trace(*tool_names: str) -> ToolTrace:
    return ToolTrace(
        calls=[
            ToolCall(tool_name=n, arguments={}, sequence_index=i) for i, n in enumerate(tool_names)
        ],
    )


def _example(
    *,
    expected: list[str] | None = None,
    expected_no_tools: bool = False,
    tags: list[str] | None = None,
) -> SuiteExample:
    return SuiteExample(
        id="ex1",
        inputs={},
        tags=tags or [],
        expected_tools=([ExpectedToolCall(tool_name=n) for n in expected] if expected else None),
        expected_no_tools=expected_no_tools,
    )


def _evaluator(
    mode: str = "expected", *, severity_floor: str | None = None
) -> ToolSelectionEvaluator:
    cfg = ToolSelectionEvaluatorConfig(
        name="tool_selection",
        mode=mode,  # type: ignore[arg-type]
        severity_floor=severity_floor,  # type: ignore[arg-type]
    )
    return ToolSelectionEvaluator(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSequenceMatch:
    def test_perfect_in_order(self) -> None:
        assert _sequence_match(["a", "b"], ["a", "b"]) == 1.0

    def test_partial_in_order(self) -> None:
        assert _sequence_match(["a", "b"], ["a"]) == 0.5

    def test_extra_intermediate_calls_ok(self) -> None:
        # Expected ["a","b"] is satisfied even with noise between.
        assert _sequence_match(["a", "b"], ["a", "x", "b"]) == 1.0

    def test_no_expected(self) -> None:
        assert _sequence_match([], []) == 1.0


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert _jaccard({"a"}, {"a"}) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_overlap(self) -> None:
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert _jaccard(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# Mode: expected
# ---------------------------------------------------------------------------


class TestExpectedMode:
    async def test_perfect_match(self) -> None:
        ev = _evaluator()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=["search_orders"]),
            source_trace=_trace("search_orders"),
            target_trace=_trace("search_orders"),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0
        assert record.delta == 0.0

    async def test_target_misses_expected_tool(self) -> None:
        ev = _evaluator()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=["notify_security_team"]),
            source_trace=_trace("notify_security_team"),
            target_trace=_trace("send_email"),  # wrong tool
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0
        assert record.delta == -1.0

    async def test_skipped_when_no_expected_tools(self) -> None:
        ev = _evaluator()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=None),
            source_trace=_trace("a"),
            target_trace=_trace("b"),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0
        assert record.metadata.get("skipped")

    async def test_severity_floor_recorded(self) -> None:
        ev = _evaluator(severity_floor="high")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=["a"]),
            source_trace=_trace("a"),
            target_trace=_trace("a"),
        )
        assert record.metadata.get("severity_floor") == "high"


# ---------------------------------------------------------------------------
# Mode: expected_no_tools
# ---------------------------------------------------------------------------


class TestExpectedNoTools:
    async def test_target_compliant(self) -> None:
        ev = _evaluator()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected_no_tools=True),
            source_trace=_trace(),
            target_trace=_trace(),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0

    async def test_target_calls_anyway(self) -> None:
        ev = _evaluator()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected_no_tools=True),
            source_trace=_trace(),
            target_trace=_trace("send_email"),
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0


# ---------------------------------------------------------------------------
# Mode: exact / set / first
# ---------------------------------------------------------------------------


class TestOtherModes:
    async def test_exact_mismatch(self) -> None:
        ev = _evaluator(mode="exact")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace("a", "b"),
            target_trace=_trace("a", "c"),
        )
        assert record.target_score == 0.0

    async def test_exact_match(self) -> None:
        ev = _evaluator(mode="exact")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace("a", "b"),
            target_trace=_trace("a", "b"),
        )
        assert record.target_score == 1.0

    async def test_set_jaccard(self) -> None:
        ev = _evaluator(mode="set")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace("a", "b"),
            target_trace=_trace("b", "c"),
        )
        assert record.target_score == pytest.approx(1 / 3)

    async def test_first_only_first_matters(self) -> None:
        ev = _evaluator(mode="first")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace("a", "b"),
            target_trace=_trace("a", "c"),
        )
        assert record.target_score == 1.0  # first matches → ok regardless of rest

    async def test_first_mismatch(self) -> None:
        ev = _evaluator(mode="first")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace("a"),
            target_trace=_trace("b"),
        )
        assert record.target_score == 0.0
