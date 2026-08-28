"""Diff imported source/target agent traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evalshift.evaluators.failures import ARGUMENT_VALUE_DRIFT, TOOL_ORDER_DRIFT
from evalshift.traces.models import AgentTrace, ToolCallEvent


@dataclass(frozen=True, slots=True)
class TraceDiffItem:
    """One difference between source and target traces."""

    kind: str
    category: str
    source_index: int | None = None
    target_index: int | None = None
    source_name: str | None = None
    target_name: str | None = None
    field: str | None = None
    source_value: Any = None
    target_value: Any = None


@dataclass(frozen=True, slots=True)
class TraceDiff:
    """Complete diff for one source/target trace pair."""

    prompt_id: str
    example_id: str
    items: list[TraceDiffItem] = field(default_factory=list)


def diff_traces(source: AgentTrace, target: AgentTrace) -> TraceDiff:
    """Return a compact tool-call oriented diff for imported traces."""
    items: list[TraceDiffItem] = []
    source_calls = source.tool_calls
    target_calls = target.tool_calls
    source_counts = Counter(call.name for call in source_calls)
    target_counts = Counter(call.name for call in target_calls)

    for name, source_count in sorted(source_counts.items()):
        missing = source_count - target_counts.get(name, 0)
        if missing > 0:
            for call in _last_n([c for c in source_calls if c.name == name], missing):
                items.append(
                    TraceDiffItem(
                        kind="missing",
                        category=TOOL_ORDER_DRIFT,
                        source_index=call.sequence_index,
                        source_name=call.name,
                    ),
                )

    for name, target_count in sorted(target_counts.items()):
        extra = target_count - source_counts.get(name, 0)
        if extra > 0:
            for call in _last_n([c for c in target_calls if c.name == name], extra):
                items.append(
                    TraceDiffItem(
                        kind="extra",
                        category=TOOL_ORDER_DRIFT,
                        target_index=call.sequence_index,
                        target_name=call.name,
                    ),
                )

    for source_call, target_call in _match_same_name_calls(source_calls, target_calls):
        if source_call.sequence_index != target_call.sequence_index:
            items.append(
                TraceDiffItem(
                    kind="reordered",
                    category=TOOL_ORDER_DRIFT,
                    source_index=source_call.sequence_index,
                    target_index=target_call.sequence_index,
                    source_name=source_call.name,
                    target_name=target_call.name,
                ),
            )
        for field_name in sorted(set(source_call.arguments) | set(target_call.arguments)):
            source_value = source_call.arguments.get(field_name)
            target_value = target_call.arguments.get(field_name)
            if source_value != target_value:
                items.append(
                    TraceDiffItem(
                        kind="argument_drift",
                        category=ARGUMENT_VALUE_DRIFT,
                        source_index=source_call.sequence_index,
                        target_index=target_call.sequence_index,
                        source_name=source_call.name,
                        target_name=target_call.name,
                        field=field_name,
                        source_value=source_value,
                        target_value=target_value,
                    ),
                )

    return TraceDiff(prompt_id=source.prompt_id, example_id=source.example_id, items=items)


def _match_same_name_calls(
    source: list[ToolCallEvent],
    target: list[ToolCallEvent],
) -> list[tuple[ToolCallEvent, ToolCallEvent]]:
    matched: list[tuple[ToolCallEvent, ToolCallEvent]] = []
    used: set[int] = set()
    for source_call in source:
        for index, target_call in enumerate(target):
            if index in used or target_call.name != source_call.name:
                continue
            used.add(index)
            matched.append((source_call, target_call))
            break
    return matched


def _last_n(items: list[ToolCallEvent], n: int) -> list[ToolCallEvent]:
    if n <= 0:
        return []
    return items[-n:]


__all__ = ["TraceDiff", "TraceDiffItem", "diff_traces"]
