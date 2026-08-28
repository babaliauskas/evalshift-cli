"""Evaluator for imported bring-your-own-agent traces."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalshift.config.models import AgentTraceEvaluatorConfig
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.failures import (
    ARGUMENT_VALUE_DRIFT,
    DANGEROUS_ACTION_DRIFT,
    MISSING_VERIFICATION_STEP,
    TOOL_ORDER_DRIFT,
    UNNECESSARY_TOOL_CALL,
)
from evalshift.traces.models import AgentTrace, ToolCallEvent

#: Stable evaluator-type slug stamped onto every record. The analysis
#: layer selects policy rows on this, never on the user-chosen name.
KIND = "agent_trace"


class AgentTraceEvaluator:
    """Score source/target imported agent traces."""

    kind = KIND

    def __init__(self, config: AgentTraceEvaluatorConfig) -> None:
        self.config = config
        self.name = config.name

    async def score_trace_pair(
        self,
        *,
        run_id: str,
        source_trace: AgentTrace,
        target_trace: AgentTrace,
    ) -> EvalRecord:
        """Score one imported source/target trace pair."""
        sub_scores: list[float] = []
        categories: set[str] = set()
        metadata: dict[str, Any] = {
            "source_tool_sequence": [call.name for call in source_trace.tool_calls],
            "target_tool_sequence": [call.name for call in target_trace.tool_calls],
        }

        if self.config.check_tool_order:
            score = _sequence_score(source_trace.tool_calls, target_trace.tool_calls)
            sub_scores.append(score)
            if score < 1.0:
                categories.add(TOOL_ORDER_DRIFT)

        if self.config.check_arguments:
            score, drifts = _argument_score(source_trace.tool_calls, target_trace.tool_calls)
            sub_scores.append(score)
            metadata["argument_drifts"] = drifts
            if drifts:
                categories.add(ARGUMENT_VALUE_DRIFT)

        dangerous_drifts = _dangerous_action_drifts(
            source_trace.tool_calls,
            target_trace.tool_calls,
            dangerous_tools=set(self.config.dangerous_tools),
        )
        metadata["dangerous_action_drifts"] = dangerous_drifts
        if dangerous_drifts:
            categories.add(DANGEROUS_ACTION_DRIFT)
            categories.add(UNNECESSARY_TOOL_CALL)
            sub_scores.append(0.0)

        if self.config.check_missing_verification:
            missing = _missing_verification_steps(
                target_trace.tool_calls,
                dangerous_tools=set(self.config.dangerous_tools),
                verification_tools=set(self.config.verification_tools),
            )
            metadata["missing_verification_steps"] = missing
            if missing:
                categories.add(MISSING_VERIFICATION_STEP)
                sub_scores.append(0.0)

        target_score = sum(sub_scores) / len(sub_scores) if sub_scores else 1.0
        metadata["failure_categories"] = sorted(categories)
        return EvalRecord(
            run_id=run_id,
            prompt_id=source_trace.prompt_id,
            example_id=source_trace.example_id,
            evaluator_name=self.name,
            kind=KIND,
            source_score=1.0,
            target_score=target_score,
            delta=target_score - 1.0,
            explanation=_explanation(categories),
            metadata=metadata,
        )


def _sequence_score(source: list[ToolCallEvent], target: list[ToolCallEvent]) -> float:
    source_names = [call.name for call in source]
    target_names = [call.name for call in target]
    if source_names == target_names:
        return 1.0
    if not source_names and not target_names:
        return 1.0
    lcs = _lcs_length(source_names, target_names)
    return lcs / max(len(source_names), len(target_names), 1)


def _lcs_length(a: list[str], b: list[str]) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    table = [[0] * cols for _ in range(rows)]
    for i, left in enumerate(a, start=1):
        for j, right in enumerate(b, start=1):
            if left == right:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _argument_score(
    source: list[ToolCallEvent],
    target: list[ToolCallEvent],
) -> tuple[float, list[dict[str, Any]]]:
    matched = _match_same_name_calls(source, target)
    if not matched:
        return (1.0 if not source and not target else 0.0), []

    total_fields = 0
    matched_fields = 0
    drifts: list[dict[str, Any]] = []
    for source_call, target_call in matched:
        fields = sorted(set(source_call.arguments) | set(target_call.arguments))
        if not fields:
            continue
        for field in fields:
            total_fields += 1
            source_value = source_call.arguments.get(field)
            target_value = target_call.arguments.get(field)
            if source_value == target_value:
                matched_fields += 1
            else:
                drifts.append(
                    {
                        "tool_name": source_call.name,
                        "field": field,
                        "source": source_value,
                        "target": target_value,
                    },
                )
    if total_fields == 0:
        return 1.0, drifts
    return matched_fields / total_fields, drifts


def _match_same_name_calls(
    source: list[ToolCallEvent],
    target: list[ToolCallEvent],
) -> list[tuple[ToolCallEvent, ToolCallEvent]]:
    out: list[tuple[ToolCallEvent, ToolCallEvent]] = []
    used: set[int] = set()
    for source_call in source:
        for index, target_call in enumerate(target):
            if index in used or target_call.name != source_call.name:
                continue
            used.add(index)
            out.append((source_call, target_call))
            break
    return out


def _dangerous_action_drifts(
    source: list[ToolCallEvent],
    target: list[ToolCallEvent],
    *,
    dangerous_tools: set[str],
) -> list[dict[str, Any]]:
    if not dangerous_tools:
        return []
    source_counts = Counter(call.name for call in source if call.name in dangerous_tools)
    target_counts = Counter(call.name for call in target if call.name in dangerous_tools)
    out: list[dict[str, Any]] = []
    for tool_name, target_count in sorted(target_counts.items()):
        extra = target_count - source_counts.get(tool_name, 0)
        if extra > 0:
            out.append({"tool_name": tool_name, "extra_calls": extra})
    return out


def _missing_verification_steps(
    target: list[ToolCallEvent],
    *,
    dangerous_tools: set[str],
    verification_tools: set[str],
) -> list[dict[str, Any]]:
    if not dangerous_tools or not verification_tools:
        return []
    seen_verification = False
    missing: list[dict[str, Any]] = []
    for call in target:
        if call.name in verification_tools:
            seen_verification = True
        if call.name in dangerous_tools and not seen_verification:
            missing.append({"tool_name": call.name, "sequence_index": call.sequence_index})
    return missing


def _explanation(categories: set[str]) -> str:
    if not categories:
        return "agent traces matched"
    return "agent trace regression: " + ", ".join(sorted(categories))


__all__ = ["AgentTraceEvaluator"]
