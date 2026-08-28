"""Tests for imported agent trace diffing."""

from __future__ import annotations

from evalshift.evaluators.failures import ARGUMENT_VALUE_DRIFT, TOOL_ORDER_DRIFT
from evalshift.traces.diff import diff_traces
from evalshift.traces.models import AgentTrace


def _tool(name: str, index: int, args: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "type": "tool_call",
        "sequence_index": index,
        "timestamp": "2026-06-09T12:00:00Z",
        "metadata": {},
        "name": name,
        "arguments": args or {},
        "call_id": f"call_{name}_{index}",
    }


def _trace(events: list[dict[str, object]], *, role: str) -> AgentTrace:
    return AgentTrace.model_validate(
        {
            "run_id": "r_trace",
            "prompt_id": "p",
            "example_id": "ex1",
            "role": role,
            "events": events,
        },
    )


def test_diff_marks_missing_extra_and_argument_drift() -> None:
    source = _trace(
        [
            _tool("check_refund_policy", 0),
            _tool("issue_refund", 1, {"ticket_id": "T-1032"}),
        ],
        role="source",
    )
    target = _trace(
        [
            _tool("issue_refund", 0, {"ticket_id": "T-1023"}),
            _tool("send_email", 1),
        ],
        role="target",
    )

    diff = diff_traces(source, target)

    assert any(
        item.kind == "missing" and item.source_name == "check_refund_policy" for item in diff.items
    )
    assert any(item.kind == "extra" and item.target_name == "send_email" for item in diff.items)
    argument_items = [item for item in diff.items if item.kind == "argument_drift"]
    assert argument_items[0].category == ARGUMENT_VALUE_DRIFT
    assert argument_items[0].field == "ticket_id"


def test_diff_marks_reordered_matching_tool_calls() -> None:
    source = _trace([_tool("a", 0), _tool("b", 1)], role="source")
    target = _trace([_tool("b", 0), _tool("a", 1)], role="target")

    diff = diff_traces(source, target)

    assert any(
        item.kind == "reordered" and item.category == TOOL_ORDER_DRIFT for item in diff.items
    )
