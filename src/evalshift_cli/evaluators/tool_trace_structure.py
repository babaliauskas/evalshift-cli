"""Evaluator: structural properties of tool traces (call count, parallelism, refusal).

Sub-scores combined into a single target_score:

* ``call_count`` — linear decay past ``call_count_tolerance``.
* ``parallelism`` — boolean match of ``has_parallel_calls()``, compared
  round by round on a teacher-forced multi-round replay: fanning out is a
  property of one response, so two traces that fan out in *different* rounds
  do not match. See :func:`_parallelism_agrees`.
* ``refusal_alignment`` — boolean match. A refusal regression
  (target refused but source didn't, or vice versa) sets
  ``severity_floor="high"`` in the record metadata so the analysis
  layer can surface it prominently.
* ``expected_count_alignment`` — boolean match against
  ``example.expected_tool_count``, when set.
"""

from __future__ import annotations

from typing import Any

from evalshift_cli.config.models import ToolTraceStructureEvaluatorConfig
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import REFUSAL_REGRESSION, TOOL_TRACE_STRUCTURE_DRIFT
from evalshift_cli.evaluators.tool_models import ToolTrace
from evalshift_cli.evaluators.tool_rounds import is_multi_round, padded_rounds, replayed_rounds
from evalshift_cli.suite.models import SuiteExample

#: Stable evaluator-type slug stamped onto every record. The analysis
#: layer selects policy rows on this, never on the user-chosen name.
KIND = "tool_trace_structure"


class ToolTraceStructureEvaluator:
    """Score structural properties of (source, target) tool traces."""

    kind = KIND

    def __init__(self, config: ToolTraceStructureEvaluatorConfig) -> None:
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
        sub_scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        if self.config.check_call_count:
            sub_scores["call_count"] = self._score_call_count(source_trace, target_trace)
            details["call_count"] = {
                "source": source_trace.call_count,
                "target": target_trace.call_count,
                "tolerance": self.config.call_count_tolerance,
            }

        if self.config.check_parallelism:
            same_parallel, per_round = _parallelism_agrees(source_trace, target_trace)
            sub_scores["parallelism"] = 1.0 if same_parallel else 0.0
            details["parallelism"] = {
                "source": source_trace.has_parallel_calls(),
                "target": target_trace.has_parallel_calls(),
            }
            if per_round is not None:
                details["parallelism"]["rounds"] = per_round

        if self.config.check_refusals:
            same_refusal = source_trace.raised_refusal == target_trace.raised_refusal
            sub_scores["refusal_alignment"] = 1.0 if same_refusal else 0.0
            details["refusal_alignment"] = {
                "source_refused": source_trace.raised_refusal,
                "target_refused": target_trace.raised_refusal,
            }
            if not same_refusal:
                # Refusal regressions are user-facing — force severity floor.
                details["severity_floor"] = "high"

        if example.expected_tool_count is not None:
            sub_scores["expected_count"] = (
                1.0 if target_trace.call_count == example.expected_tool_count else 0.0
            )
            details["expected_count"] = {
                "expected": example.expected_tool_count,
                "actual": target_trace.call_count,
            }

        if is_multi_round(source_trace, target_trace):
            # Disclosed on the record because the analysis layer never sees
            # the suite: a pair scored over four rounds and a pair scored over
            # one are not the same measurement, and only this says which.
            details["rounds_replayed"] = {
                "source": source_trace.round_count,
                "target": target_trace.round_count,
            }

        # When all sub-checks are disabled, treat the pair as trivially
        # equivalent — the user explicitly opted out of every check.
        target_score = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 1.0
        meta: dict[str, Any] = {
            "sub_scores": sub_scores,
            "details": details,
        }
        if "severity_floor" in details:
            meta["severity_floor"] = details["severity_floor"]
        if target_score < 1.0:
            meta["failure_categories"] = [
                REFUSAL_REGRESSION
                if sub_scores.get("refusal_alignment") == 0.0
                else TOOL_TRACE_STRUCTURE_DRIFT,
            ]
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            evaluator_name=self.name,
            kind=KIND,
            source_score=1.0,
            target_score=target_score,
            delta=target_score - 1.0,
            metadata=meta,
        )

    def _score_call_count(self, source: ToolTrace, target: ToolTrace) -> float:
        delta = abs(source.call_count - target.call_count)
        if delta <= self.config.call_count_tolerance:
            return 1.0
        excess = delta - self.config.call_count_tolerance
        return max(0.0, 1.0 - excess / max(source.call_count, 1))


def _parallelism_agrees(
    source: ToolTrace,
    target: ToolTrace,
) -> tuple[bool, list[dict[str, Any]] | None]:
    """Whether the two sides fanned out in the same rounds.

    Parallelism describes one model response, so a multi-round replay is
    compared response by response: two traces that each fan out once, in
    different rounds, sequenced their work differently and the whole-trace
    booleans hide it. A round only one side replayed counts as sequential —
    it made no parallel calls, because it made none at all.

    Args:
        source: The source model's trace.
        target: The target model's trace.

    Returns:
        ``(agrees, per_round)``. ``per_round`` is ``None`` for a single-round
        pair, whose record must stay exactly what it was before rounds
        existed; otherwise it is one ``{"round", "source", "target"}`` entry
        per replayed round, which is what explains a 0.0 whose two top-level
        booleans read the same.
    """
    if not is_multi_round(source, target):
        return source.has_parallel_calls() == target.has_parallel_calls(), None
    count = replayed_rounds(source, target)
    source_rounds = padded_rounds(source, count)
    target_rounds = padded_rounds(target, count)
    detail = [
        {
            "round": index,
            "source": source_rounds[index].has_parallel_calls(),
            "target": target_rounds[index].has_parallel_calls(),
        }
        for index in range(count)
    ]
    return all(entry["source"] == entry["target"] for entry in detail), detail


__all__ = ["ToolTraceStructureEvaluator"]
