"""Evaluator: did the target model call the tools we expected?

Four modes (see :class:`ToolSelectionEvaluatorConfig`):

* ``exact`` — sequence equality on tool names.
* ``set`` — Jaccard similarity on the set of tool names.
* ``first`` — only the first tool call is checked.
* ``expected`` (default) — match against ``example.expected_tools`` in
  order. This is the most useful mode for migrations: we compare each
  model against ground truth, so a regression is unambiguous.

Both source and target sides get scored against the same yardstick, so
``delta = target - source`` carries the regression signal.
"""

from __future__ import annotations

from typing import Any

from evalshift.config.models import ToolSelectionEvaluatorConfig
from evalshift.evaluators.base import EvalRecord, PairedScore
from evalshift.evaluators.tool_models import ToolTrace
from evalshift.suite.models import SuiteExample


class ToolSelectionEvaluator:
    """Score whether the right tools were called."""

    def __init__(self, config: ToolSelectionEvaluatorConfig) -> None:
        self.config = config
        self.name = config.name

    async def score_pair(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        """Score a (source, target) trace pair for one example."""
        if example.expected_no_tools:
            return self._score_no_tools(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )

        mode = self.config.mode
        if mode == "expected":
            return self._score_expected(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        if mode == "exact":
            return self._score_exact(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        if mode == "set":
            return self._score_set(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        # mode == "first"
        return self._score_first(
            run_id=run_id,
            prompt_id=prompt_id,
            example=example,
            source_trace=source_trace,
            target_trace=target_trace,
        )

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _score_no_tools(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        target_score = 1.0 if target_trace.call_count == 0 else 0.0
        source_score = 1.0 if source_trace.call_count == 0 else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=source_score,
                target_score=target_score,
                metadata={
                    "mode": "expected_no_tools",
                    "source_calls": source_trace.call_count,
                    "target_calls": target_trace.call_count,
                },
            ),
        )

    def _score_expected(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        if not example.expected_tools:
            # No ground truth: skip cleanly. Neutral 1.0/1.0 keeps the
            # delta at 0; metadata flag tells the analysis layer this
            # row was effectively skipped.
            return self._record(
                run_id=run_id,
                prompt_id=prompt_id,
                example_id=example.id,
                paired=PairedScore(
                    source_score=1.0,
                    target_score=1.0,
                    metadata={"skipped": "no expected_tools in example"},
                ),
            )

        expected_names = [e.tool_name for e in example.expected_tools]
        source_score = _sequence_match(expected_names, source_trace.tool_names)
        target_score = _sequence_match(expected_names, target_trace.tool_names)
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=source_score,
                target_score=target_score,
                metadata={
                    "mode": "expected",
                    "expected_names": expected_names,
                    "source_names": source_trace.tool_names,
                    "target_names": target_trace.tool_names,
                },
            ),
        )

    def _score_exact(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        # Source-vs-itself = 1.0 by definition; target must match source's sequence.
        target_score = 1.0 if target_trace.tool_names == source_trace.tool_names else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "exact",
                    "source_names": source_trace.tool_names,
                    "target_names": target_trace.tool_names,
                },
            ),
        )

    def _score_set(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        source_set = source_trace.tool_name_set
        target_set = target_trace.tool_name_set
        target_score = _jaccard(source_set, target_set)
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "set",
                    "source_set": sorted(source_set),
                    "target_set": sorted(target_set),
                },
            ),
        )

    def _score_first(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        src_first = source_trace.tool_names[0] if source_trace.calls else None
        tgt_first = target_trace.tool_names[0] if target_trace.calls else None
        # Source is always its own baseline at 1.0.
        target_score = 1.0 if src_first is not None and tgt_first == src_first else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "first",
                    "source_first": src_first,
                    "target_first": tgt_first,
                },
            ),
        )

    def _record(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example_id: str,
        paired: PairedScore,
    ) -> EvalRecord:
        meta: dict[str, Any] = dict(paired.metadata)
        if self.config.severity_floor:
            meta["severity_floor"] = self.config.severity_floor
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example_id,
            evaluator_name=self.name,
            source_score=paired.source_score,
            target_score=paired.target_score,
            delta=paired.delta,
            explanation=paired.explanation,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sequence_match(expected: list[str], actual: list[str]) -> float:
    """How many ``expected`` names appear in ``actual`` in order, normalised."""
    if not expected:
        return 1.0
    i = j = matched = 0
    while i < len(expected) and j < len(actual):
        if expected[i] == actual[j]:
            matched += 1
            i += 1
            j += 1
        else:
            j += 1
    return matched / len(expected)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity. Returns 1.0 if both sets empty, else intersection/union."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


__all__ = ["ToolSelectionEvaluator"]
