"""Tests for imported agent trace models and JSONL loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalshift.traces.loader import (
    TraceLoadError,
    index_traces,
    load_traces_jsonl,
    pairs_for_prompt_examples,
    write_traces_jsonl,
)
from evalshift.traces.models import AgentTrace, ToolCallEvent


def _trace(*, role: str = "source", example_id: str = "ex1") -> dict[str, object]:
    return {
        "run_id": "r_20260609_trace1",
        "prompt_id": "p",
        "example_id": example_id,
        "role": role,
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 2,
                "timestamp": "2026-06-09T12:00:02Z",
                "metadata": {"phase": "act"},
                "name": "issue_refund",
                "arguments": {"ticket_id": "T-1032"},
                "call_id": "call_refund",
            },
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-09T12:00:00Z",
                "metadata": {},
                "model_id": "src-model",
                "input": {"query": "refund"},
                "output": "thinking",
                "input_tokens": 10,
                "output_tokens": 4,
                "cost_usd": 0.01,
                "latency_ms": 100,
            },
            {
                "type": "tool_result",
                "sequence_index": 3,
                "timestamp": "2026-06-09T12:00:03Z",
                "metadata": {},
                "name": "issue_refund",
                "call_id": "call_refund",
                "result": {"ok": True},
                "error": None,
            },
        ],
    }


def test_agent_trace_sorts_events_by_sequence_index() -> None:
    trace = AgentTrace.model_validate(_trace())

    assert [event.sequence_index for event in trace.events] == [0, 2, 3]
    assert isinstance(trace.events[1], ToolCallEvent)
    assert trace.tool_calls[0].name == "issue_refund"


def test_agent_trace_rejects_duplicate_sequence_index() -> None:
    payload = _trace()
    events = payload["events"]
    assert isinstance(events, list)
    events[1]["sequence_index"] = 2

    with pytest.raises(ValueError, match="duplicate sequence_index"):
        AgentTrace.model_validate(payload)


def test_agent_trace_rejects_tool_result_without_prior_call_id() -> None:
    payload = _trace()
    events = payload["events"]
    assert isinstance(events, list)
    events[2]["call_id"] = "missing_call"

    with pytest.raises(ValueError, match="unknown tool_result call_id"):
        AgentTrace.model_validate(payload)


def test_load_traces_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_trace()) + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(TraceLoadError, match=r"traces\.jsonl:2"):
        load_traces_jsonl(path)


def test_write_and_load_traces_jsonl_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    trace = AgentTrace.model_validate(_trace())

    write_traces_jsonl(path, [trace])

    loaded = load_traces_jsonl(path)
    assert loaded == [trace]


def test_index_and_pair_traces_by_prompt_example_and_role() -> None:
    source = AgentTrace.model_validate(_trace(role="source"))
    target = AgentTrace.model_validate(_trace(role="target"))
    indexed = index_traces([target, source])

    assert indexed[("p", "ex1", "source")] == source
    assert indexed[("p", "ex1", "target")] == target

    pairs = pairs_for_prompt_examples([source, target], prompt_examples=[("p", "ex1")])
    assert len(pairs) == 1
    assert pairs[0].source == source
    assert pairs[0].target == target


def test_pairing_omits_incomplete_pairs() -> None:
    source = AgentTrace.model_validate(_trace(role="source"))

    assert pairs_for_prompt_examples([source], prompt_examples=[("p", "ex1")]) == []


# --- Phase B3.2: trace timestamps must survive the server's UTC rule ----------


def _with_timestamp(value: str) -> dict[str, object]:
    trace = _trace()
    events = trace["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, dict)
        event["timestamp"] = value
    return trace


def test_naive_event_timestamp_is_rejected() -> None:
    """A timestamp with no offset names no instant, and the bundle contract forbids it.

    Rejecting beats guessing: assuming UTC would silently relabel a trace
    recorded in another zone, and the whole run would carry the lie.
    """
    with pytest.raises(ValueError, match="timestamp"):
        AgentTrace.model_validate(_with_timestamp("2026-06-09T12:00:00"))


def test_naive_event_timestamp_names_the_file_and_line(tmp_path: Path) -> None:
    """The error must point at the capture, which is why this is caught at load."""
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(_with_timestamp("2026-06-09T12:00:00")) + "\n", encoding="utf-8")

    with pytest.raises(TraceLoadError, match=r"traces\.jsonl:1"):
        load_traces_jsonl(path)


def test_offset_event_timestamp_is_normalized_to_utc() -> None:
    """An offset timestamp is unambiguous, so it is converted rather than refused."""
    trace = AgentTrace.model_validate(_with_timestamp("2026-06-09T14:00:00+02:00"))

    dumped = trace.model_dump(mode="json")
    events = dumped["events"]
    assert [event["timestamp"] for event in events] == ["2026-06-09T12:00:00Z"] * len(events)


def test_utc_event_timestamps_serialize_the_way_the_bundle_schema_demands() -> None:
    """The server's `pattern` accepts `Z` and `+00:00` only; pydantic emits `Z`."""
    trace = AgentTrace.model_validate(_trace())

    dumped = trace.model_dump(mode="json")
    for event in dumped["events"]:
        assert str(event["timestamp"]).endswith("Z")
