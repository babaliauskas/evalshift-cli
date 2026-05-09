"""Tests for :class:`ToolTraceStructureEvaluator`."""

from __future__ import annotations

import pytest

from evalshift.config.models import ToolTraceStructureEvaluatorConfig
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.evaluators.tool_trace_structure import ToolTraceStructureEvaluator
from evalshift.suite.models import SuiteExample


def _trace(
    *names: str,
    parents: list[str | None] | None = None,
    raised_refusal: bool = False,
) -> ToolTrace:
    parents = parents or [None] * len(names)
    return ToolTrace(
        calls=[
            ToolCall(
                tool_name=n,
                arguments={},
                sequence_index=i,
                parent_call_id=p,
            )
            for i, (n, p) in enumerate(zip(names, parents, strict=True))
        ],
        raised_refusal=raised_refusal,
    )


def _ev(
    *,
    check_call_count: bool = True,
    check_parallelism: bool = True,
    check_refusals: bool = True,
    call_count_tolerance: int = 1,
) -> ToolTraceStructureEvaluator:
    cfg = ToolTraceStructureEvaluatorConfig(
        name="tool_trace_structure",
        check_call_count=check_call_count,
        check_parallelism=check_parallelism,
        check_refusals=check_refusals,
        call_count_tolerance=call_count_tolerance,
    )
    return ToolTraceStructureEvaluator(cfg)


class TestCallCount:
    async def test_within_tolerance(self) -> None:
        ev = _ev(call_count_tolerance=1)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a", "b"),
            target_trace=_trace("a", "b", "c"),  # +1 within tolerance
        )
        assert record.metadata["sub_scores"]["call_count"] == 1.0

    async def test_outside_tolerance_decays(self) -> None:
        ev = _ev(call_count_tolerance=0)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a"),
            target_trace=_trace("a", "b", "c", "d"),
        )
        assert record.metadata["sub_scores"]["call_count"] < 1.0


class TestParallelism:
    async def test_both_parallel_match(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a", "b"),  # 2 top-level → parallel
            target_trace=_trace("a", "b"),
        )
        assert record.metadata["sub_scores"]["parallelism"] == 1.0

    async def test_mismatch(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a", "b"),  # parallel
            target_trace=_trace("a", "b", parents=[None, "a"]),  # chained
        )
        assert record.metadata["sub_scores"]["parallelism"] == 0.0


class TestRefusalAlignment:
    async def test_aligned(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace(raised_refusal=True),
            target_trace=_trace(raised_refusal=True),
        )
        assert record.metadata["sub_scores"]["refusal_alignment"] == 1.0

    async def test_mismatch_forces_severity_floor(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a"),
            target_trace=_trace(raised_refusal=True),
        )
        assert record.metadata["sub_scores"]["refusal_alignment"] == 0.0
        assert record.metadata.get("severity_floor") == "high"


class TestExpectedCount:
    async def test_match(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex", expected_tool_count=2),
            source_trace=_trace("a"),
            target_trace=_trace("a", "b"),  # matches expected_tool_count=2
        )
        assert record.metadata["sub_scores"]["expected_count"] == 1.0

    async def test_mismatch(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex", expected_tool_count=2),
            source_trace=_trace("a"),
            target_trace=_trace("a", "b", "c"),
        )
        assert record.metadata["sub_scores"]["expected_count"] == 0.0


class TestAllChecksDisabled:
    async def test_returns_one(self) -> None:
        ev = _ev(
            check_call_count=False,
            check_parallelism=False,
            check_refusals=False,
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a"),
            target_trace=_trace("b", "c", "d"),
        )
        assert record.target_score == 1.0


class TestCombinedScore:
    async def test_subscore_average(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=SuiteExample(id="ex"),
            source_trace=_trace("a"),  # 1 call, no refusal, no parallel
            target_trace=_trace("a", "b"),  # 2 calls (within tolerance), parallel
        )
        # call_count=1.0 (within tol), parallelism=0 (source single, target parallel),
        # refusal_alignment=1.0
        # mean = 2/3 ≈ 0.667
        assert record.target_score == pytest.approx(2 / 3)
