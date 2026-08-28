"""Map the CLI's internal traces onto the hosted bundle's event format.

The hosted run-detail page renders what a model *did*, not just what it said.
Two internal representations describe that, and neither goes on the wire:

* :class:`~evalshift.evaluators.tool_models.ToolTrace` — produced by replay. A
  flat call list plus optional final text, no rounds and no tool results,
  because a single-shot replay never executes a tool.
* :class:`~evalshift.traces.models.AgentTrace` — imported bring-your-own-agent
  timelines. A discriminated event stream with rounds, results and timestamps.

Both map here onto one wire shape so the server, the database and the client
speak a single language. Neither internal model changes.
"""

from __future__ import annotations

import json
from typing import Any

from evalshift.evaluators.tool_models import ToolTrace
from evalshift.traces.models import AgentTrace

__all__ = ["MAX_RESULT_BYTES", "MAX_STREAM_BYTES", "from_agent_trace", "from_tool_trace"]

#: Largest serialized ``tool_result.result`` carried verbatim. Measured against
#: real captures the biggest single result was 8.0KB, so this is 2x headroom.
#: Raise it here if a workload returns larger payloads — nothing else depends
#: on the value.
MAX_RESULT_BYTES = 16_384

#: Largest serialized event list for one side of one example.
MAX_STREAM_BYTES = 262_144


def _cap_result(result: Any) -> tuple[Any, bool]:
    """Shorten an oversized tool result. Returns ``(result, truncated)``.

    The value is replaced by a marker rather than dropped, so the pane shows
    that something came back and how much of it was elided.
    """
    encoded = json.dumps(result, default=str)
    if len(encoded) <= MAX_RESULT_BYTES:
        return result, False
    return (
        {
            "_truncated": True,
            "_original_bytes": len(encoded),
            "preview": encoded[:MAX_RESULT_BYTES],
        },
        True,
    )


def _cap_stream(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Drop trailing events once the stream exceeds its cap.

    Leading events are kept: a trace reads forward, so the first calls are the
    ones that explain what the model did.
    """
    if len(json.dumps(events, default=str)) <= MAX_STREAM_BYTES:
        return events, False
    kept: list[dict[str, Any]] = []
    size = 2  # the enclosing "[]"
    for event in events:
        encoded = len(json.dumps(event, default=str)) + 1
        if size + encoded > MAX_STREAM_BYTES:
            break
        kept.append(event)
        size += encoded
    return kept, True


def _event(
    event_type: str, *, sequence_index: int, round_index: int, **fields: Any
) -> dict[str, Any]:
    """One wire event with the fields every event carries.

    ``timestamp`` is always present and may be ``None``: replay reconstructs
    events from a ``ToolTrace``, which records order but not time.
    """
    return {
        "type": event_type,
        "sequence_index": sequence_index,
        "round": round_index,
        "timestamp": None,
        **fields,
    }


def from_tool_trace(trace: ToolTrace | None, *, side: str) -> dict[str, Any] | None:
    """Wire stream for one replayed model side, or ``None`` if there is nothing to show.

    Every event is ``round: 0``. Replay is single-shot — one
    ``litellm.acompletion`` call with no tool results fed back — so it cannot
    produce a second round. When teacher-forced multi-round replay lands, this
    is the function that starts emitting higher rounds; the wire format already
    allows them.

    No ``model_call`` event is emitted. Its metrics would duplicate the example
    row's ``cost_usd_*`` and ``latency_ms_*`` columns.

    Args:
        trace: The call's recorded trace. ``None`` for plain text prompts.
        side: ``"source"`` or ``"target"``.

    Returns:
        The stream, or ``None`` when the trace is absent or carries nothing —
        no calls, no final text and no refusal.
    """
    if trace is None:
        return None

    events: list[dict[str, Any]] = []
    for call in trace.calls:
        events.append(
            _event(
                "tool_call",
                sequence_index=len(events),
                round_index=0,
                name=call.tool_name,
                arguments=call.arguments,
                call_id=call.call_id,
                parent_call_id=call.parent_call_id,
            )
        )

    if trace.final_text:
        events.append(
            _event(
                "final_output",
                sequence_index=len(events),
                round_index=0,
                text=trace.final_text,
            )
        )

    if trace.raised_refusal:
        events.append(
            _event(
                "error",
                sequence_index=len(events),
                round_index=0,
                message=trace.refusal_text or "model refused",
                category="refusal",
            )
        )

    if not events:
        return None
    capped, truncated = _cap_stream(events)
    return {"side": side, "events": capped, "truncated": truncated}


# Payload keys copied straight through, keyed by event type. ``model_call``
# deliberately omits ``input`` and ``output``: on the measured captures those
# two fields are ~95% of the bytes (47-172KB per capture against 0.3-8KB of
# tool results) and duplicate what ``Example.input`` already carries.
_PASSTHROUGH_FIELDS: dict[str, tuple[str, ...]] = {
    "model_call": ("model_id", "input_tokens", "output_tokens", "cost_usd", "latency_ms"),
    "tool_call": ("name", "arguments", "call_id", "parent_call_id"),
    "tool_result": ("name", "call_id", "result", "error"),
    "final_output": ("text",),
    "error": ("message", "category"),
    "retrieval": ("source", "query", "documents"),
    "guardrail": ("name", "verdict", "reason"),
}


def from_agent_trace(trace: AgentTrace) -> dict[str, Any] | None:
    """Wire stream for one imported bring-your-own-agent trace.

    A round begins at each ``model_call``. The counter increments on every
    ``model_call`` after the first, so events preceding any model call stay in
    round 0.

    Args:
        trace: The imported trace for one prompt/example/side.

    Returns:
        The stream, or ``None`` when the trace has no events.
    """
    events: list[dict[str, Any]] = []
    round_index = 0
    seen_model_call = False

    for source_event in trace.events:
        dumped = source_event.model_dump(mode="json")
        event_type = str(dumped["type"])
        if event_type == "model_call":
            if seen_model_call:
                round_index += 1
            seen_model_call = True

        fields = {
            key: dumped[key] for key in _PASSTHROUGH_FIELDS.get(event_type, ()) if key in dumped
        }
        if event_type == "tool_result":
            fields["result"], fields["truncated"] = _cap_result(fields.get("result"))
        events.append(
            {
                "type": event_type,
                "sequence_index": len(events),
                "round": round_index,
                "timestamp": dumped.get("timestamp"),
                **fields,
            }
        )

    if not events:
        return None
    capped, truncated = _cap_stream(events)
    return {"side": trace.role, "events": capped, "truncated": truncated}
