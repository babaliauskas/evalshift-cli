"""Tests for :class:`evalshift.evaluators.tool_arguments.ToolArgumentsEvaluator`."""

from __future__ import annotations

from typing import Any

import pytest

from evalshift.config.models import ToolArgumentsEvaluatorConfig
from evalshift.evaluators.tool_arguments import (
    ToolArgumentsEvaluator,
    _is_subset,
    _match_calls,
)
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.suite.models import SuiteExample


def _trace(*calls: tuple[str, dict[str, Any]]) -> ToolTrace:
    return ToolTrace(
        calls=[
            ToolCall(tool_name=name, arguments=args, sequence_index=i)
            for i, (name, args) in enumerate(calls)
        ],
    )


def _ev(
    *,
    strategies: dict[str, str] | None = None,
    numeric_tolerance: float = 0.05,
    embeddings_fn: Any = None,
) -> ToolArgumentsEvaluator:
    cfg = ToolArgumentsEvaluatorConfig(
        name="tool_arguments",
        strategies=strategies or {},  # type: ignore[arg-type]
        numeric_tolerance=numeric_tolerance,
    )
    return ToolArgumentsEvaluator(cfg, embeddings_fn=embeddings_fn)


def _example() -> SuiteExample:
    return SuiteExample(id="ex1", inputs={})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsSubset:
    def test_dict_subset(self) -> None:
        assert _is_subset({"a": 1}, {"a": 1, "b": 2})

    def test_dict_not_subset(self) -> None:
        assert not _is_subset({"a": 1, "c": 3}, {"a": 1, "b": 2})

    def test_nested_dict(self) -> None:
        assert _is_subset({"x": {"y": 1}}, {"x": {"y": 1, "z": 2}})

    def test_list_subset_unordered(self) -> None:
        assert _is_subset([1, 2], [3, 1, 2])

    def test_scalar_equality(self) -> None:
        assert _is_subset("foo", "foo")
        assert not _is_subset("foo", "bar")


class TestMatchCalls:
    def test_one_to_one(self) -> None:
        s = _trace(("a", {}), ("b", {}))
        t = _trace(("a", {}), ("b", {}))
        matched = _match_calls(s, t)
        assert len(matched) == 2
        assert [m[0].tool_name for m in matched] == ["a", "b"]

    def test_unmatched_dropped(self) -> None:
        s = _trace(("a", {}), ("b", {}))
        t = _trace(("a", {}))  # no b in target
        matched = _match_calls(s, t)
        assert len(matched) == 1

    def test_repeated_tool_pairs_by_index(self) -> None:
        s = _trace(("a", {"q": 1}), ("a", {"q": 2}))
        t = _trace(("a", {"q": 1}), ("a", {"q": 2}))
        matched = _match_calls(s, t)
        # Greedy nearest-index → src[0]→tgt[0], src[1]→tgt[1]
        assert matched[0][0].arguments == {"q": 1}
        assert matched[0][1].arguments == {"q": 1}
        assert matched[1][0].arguments == {"q": 2}
        assert matched[1][1].arguments == {"q": 2}


# ---------------------------------------------------------------------------
# Score scenarios
# ---------------------------------------------------------------------------


class TestExactStrategy:
    async def test_perfect_match(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "ACME"})),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0

    async def test_value_drift(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "Initech"})),
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0


class TestSubsetStrategy:
    async def test_target_has_extras_passes(self) -> None:
        ev = _ev(strategies={"q": "subset"})
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "ACME"})),
        )
        assert record.target_score == 1.0


class TestNumericStrategy:
    async def test_within_tolerance(self) -> None:
        ev = _ev(strategies={"qty": "numeric"}, numeric_tolerance=0.1)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 100})),
            target_trace=_trace(("ship", {"qty": 105})),
        )
        # 5% relative error within 10% tolerance → high score.
        assert record.target_score > 0.4

    async def test_outside_tolerance(self) -> None:
        ev = _ev(strategies={"qty": "numeric"}, numeric_tolerance=0.05)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 10})),
            target_trace=_trace(("ship", {"qty": 100})),  # huge drift
        )
        assert record.target_score == 0.0


class TestSemanticStrategy:
    async def test_falls_back_to_exact_when_no_embeddings(self) -> None:
        ev = _ev(strategies={"text": "semantic"}, embeddings_fn=None)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"text": "hello"})),
            target_trace=_trace(("ping", {"text": "hello"})),
        )
        assert record.target_score == 1.0

    async def test_uses_embeddings_fn(self) -> None:
        async def fake_emb(a: str, b: str) -> float:
            return 0.92

        ev = _ev(strategies={"text": "semantic"}, embeddings_fn=fake_emb)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"text": "hello"})),
            target_trace=_trace(("ping", {"text": "hi"})),
        )
        assert record.target_score == pytest.approx(0.92)


class TestNoMatch:
    async def test_target_calls_different_tool(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("a", {})),
            target_trace=_trace(("b", {})),
        )
        # No matched calls; target made calls so it's a regression.
        assert record.target_score == 0.0


class TestParseError:
    async def test_parse_error_args_score_zero(self) -> None:
        ev = _ev()
        # arguments dict with sentinel _parse_error → all real keys gone.
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"_raw": "{bad", "_parse_error": True})),
            target_trace=_trace(("ping", {"_raw": "{bad", "_parse_error": True})),
        )
        # Both halves had only sentinel keys → fallback 0.0.
        assert record.target_score == 0.0
