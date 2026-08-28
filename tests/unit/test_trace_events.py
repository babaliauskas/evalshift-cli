"""Adapters from the CLI's internal traces onto the hosted wire format."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.hosted.trace_events import (
    MAX_RESULT_BYTES,
    MAX_STREAM_BYTES,
    from_agent_trace,
    from_tool_trace,
)
from evalshift.traces.models import AgentTrace


def _ts(second: int) -> datetime:
    return datetime(2026, 8, 14, 12, 0, second, tzinfo=UTC)


def _agent_trace(events: list[dict[str, object]]) -> AgentTrace:
    return AgentTrace.model_validate(
        {
            "run_id": "r_test",
            "prompt_id": "replay",
            "example_id": "cap_71dff1e7",
            "role": "source",
            "events": events,
        }
    )


def test_tool_only_trace_becomes_a_tool_call_event() -> None:
    trace = ToolTrace(
        calls=[
            ToolCall(
                tool_name="get_daily_briefing",
                arguments={},
                call_id="call_abc",
                sequence_index=0,
            )
        ],
        final_text=None,
    )

    stream = from_tool_trace(trace, side="source")

    assert stream == {
        "side": "source",
        "truncated": False,
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 0,
                "round": 0,
                "timestamp": None,
                "name": "get_daily_briefing",
                "arguments": {},
                "call_id": "call_abc",
                "parent_call_id": None,
            }
        ],
    }


def test_final_text_becomes_a_final_output_event_after_the_calls() -> None:
    trace = ToolTrace(
        calls=[ToolCall(tool_name="get_projects", arguments={}, sequence_index=0)],
        final_text="Here are your projects.",
    )

    stream = from_tool_trace(trace, side="target")
    assert stream is not None
    events = stream["events"]

    assert [e["type"] for e in events] == ["tool_call", "final_output"]
    assert events[1]["sequence_index"] == 1
    assert events[1]["text"] == "Here are your projects."


def test_text_only_trace_becomes_a_single_final_output_event() -> None:
    trace = ToolTrace(calls=[], final_text="It is lively during the fair.")

    stream = from_tool_trace(trace, side="source")

    assert stream is not None
    assert [e["type"] for e in stream["events"]] == ["final_output"]


def test_refusal_becomes_an_error_event() -> None:
    trace = ToolTrace(calls=[], raised_refusal=True, refusal_text="I can't help with that.")

    stream = from_tool_trace(trace, side="target")
    assert stream is not None
    events = stream["events"]

    assert events[-1]["type"] == "error"
    assert events[-1]["category"] == "refusal"
    assert events[-1]["message"] == "I can't help with that."


def test_none_trace_yields_no_stream() -> None:
    assert from_tool_trace(None, side="source") is None


def test_empty_trace_yields_no_stream() -> None:
    assert from_tool_trace(ToolTrace(), side="source") is None


def test_rounds_increment_on_each_model_call_after_the_first() -> None:
    trace = _agent_trace(
        [
            {"type": "model_call", "sequence_index": 0, "timestamp": _ts(0), "model_id": "m"},
            {"type": "tool_call", "sequence_index": 1, "timestamp": _ts(1), "name": "get_projects"},
            {
                "type": "tool_result",
                "sequence_index": 2,
                "timestamp": _ts(2),
                "name": "get_projects",
                "result": {"ok": True},
            },
            {"type": "model_call", "sequence_index": 3, "timestamp": _ts(3), "model_id": "m"},
            {"type": "final_output", "sequence_index": 4, "timestamp": _ts(4), "text": "done"},
        ]
    )

    stream = from_agent_trace(trace)

    assert stream is not None
    assert [(e["type"], e["round"]) for e in stream["events"]] == [
        ("model_call", 0),
        ("tool_call", 0),
        ("tool_result", 0),
        ("model_call", 1),
        ("final_output", 1),
    ]


def test_model_call_input_and_output_payloads_are_dropped() -> None:
    trace = _agent_trace(
        [
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": _ts(0),
                "model_id": "gemini/gemini-3.1-flash-lite-preview",
                "input": [{"role": "user", "content": "x" * 50_000}],
                "output": {"content": "y" * 10_000},
                "input_tokens": 26196,
                "output_tokens": 21,
                "cost_usd": 0.0065805,
                "latency_ms": 1152,
            }
        ]
    )

    stream = from_agent_trace(trace)
    assert stream is not None
    event = stream["events"][0]

    assert "input" not in event
    assert "output" not in event
    assert event["input_tokens"] == 26196
    assert event["cost_usd"] == 0.0065805


def test_side_comes_from_the_trace_role() -> None:
    trace = _agent_trace(
        [{"type": "final_output", "sequence_index": 0, "timestamp": _ts(0), "text": "hi"}]
    )
    stream = from_agent_trace(trace)
    assert stream is not None
    assert stream["side"] == "source"


def test_timestamps_are_serialized_as_iso_strings() -> None:
    trace = _agent_trace(
        [{"type": "final_output", "sequence_index": 0, "timestamp": _ts(7), "text": "hi"}]
    )
    stream = from_agent_trace(trace)
    assert stream is not None
    assert stream["events"][0]["timestamp"].startswith("2026-08-14T12:00:07")


def test_agent_trace_with_no_events_yields_no_stream() -> None:
    assert from_agent_trace(_agent_trace([])) is None


def test_oversized_tool_result_is_truncated_not_dropped() -> None:
    trace = _agent_trace(
        [
            {
                "type": "tool_result",
                "sequence_index": 0,
                "timestamp": _ts(0),
                "name": "search_web",
                "result": {"documents": ["x" * (MAX_RESULT_BYTES * 2)]},
            }
        ]
    )

    stream = from_agent_trace(trace)
    assert stream is not None
    event = stream["events"][0]

    assert event["truncated"] is True
    assert event["name"] == "search_web"
    assert len(json.dumps(event["result"])) <= MAX_RESULT_BYTES + 256


def test_result_within_the_cap_is_untouched_and_not_flagged() -> None:
    trace = _agent_trace(
        [
            {
                "type": "tool_result",
                "sequence_index": 0,
                "timestamp": _ts(0),
                "name": "get_projects",
                "result": {"projects": [1, 2, 3]},
            }
        ]
    )

    stream = from_agent_trace(trace)
    assert stream is not None
    event = stream["events"][0]

    assert event["truncated"] is False
    assert event["result"] == {"projects": [1, 2, 3]}


def test_oversized_stream_keeps_leading_events_and_flags_itself() -> None:
    big = "x" * (MAX_RESULT_BYTES // 2)
    trace = _agent_trace(
        [
            {
                "type": "tool_result",
                "sequence_index": i,
                "timestamp": _ts(0),
                "name": f"tool_{i}",
                "result": {"blob": big},
            }
            for i in range(200)
        ]
    )

    stream = from_agent_trace(trace)

    assert stream is not None
    assert stream["truncated"] is True
    assert 0 < len(stream["events"]) < 200
    assert stream["events"][0]["name"] == "tool_0"
    assert len(json.dumps(stream["events"])) <= MAX_STREAM_BYTES
