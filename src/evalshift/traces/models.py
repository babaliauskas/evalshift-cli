"""Strict models for imported bring-your-own-agent traces."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

TraceRole = Literal["source", "target"]


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcTimestamp = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""An event timestamp, offset-aware and stored in UTC.

The bundle contract requires a zero offset (`BUNDLE_SPEC.md` §Validation, enforced by
`app/runs/bundle.py`), and these events are copied into the bundle verbatim by
`evalshift.hosted.trace_events.from_agent_trace`. So the two cases are settled here, where
the error can still name the capture file and line:

* an offset timestamp is unambiguous and is converted;
* a naive one is refused, because assuming UTC would silently relabel a trace recorded
  in another zone and the whole run would then carry that as fact.
"""


class _StrictModel(BaseModel):
    """Base model for trace artifacts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _BaseEvent(_StrictModel):
    """Fields shared by all imported trace events."""

    type: str
    sequence_index: int = Field(ge=0)
    timestamp: UtcTimestamp
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCallEvent(_BaseEvent):
    """A model invocation inside an agent timeline."""

    type: Literal["model_call"]
    model_id: str = Field(min_length=1)
    input: Any = None
    output: Any = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    # Content-addressed pointer to the toolset sidecar (``sha256:<hex>``, see
    # ``evalshift.captures.toolset.fingerprint_tools``) plus the cheap,
    # display-only tool-name list. Optional *here* so the parser accepts a
    # capture that predates per-call toolset capture -- the requirement that
    # they be present is enforced at promotion, not at parse time, where a
    # useful error can name the capture instead of a generic parse failure.
    toolset_ref: str | None = None
    tools_offered: list[str] | None = None


class ToolCallEvent(_BaseEvent):
    """A tool invocation emitted by an agent."""

    type: Literal["tool_call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    parent_call_id: str | None = None


class ToolResultEvent(_BaseEvent):
    """The result returned for a previous tool call."""

    type: Literal["tool_result"]
    name: str = Field(min_length=1)
    call_id: str | None = None
    result: Any = None
    error: str | None = None


class RetrievalEvent(_BaseEvent):
    """A retrieval step in an agent timeline."""

    type: Literal["retrieval"]
    source: str = Field(min_length=1)
    query: str = ""
    documents: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailEvent(_BaseEvent):
    """A guardrail or policy check."""

    type: Literal["guardrail"]
    name: str = Field(min_length=1)
    verdict: Literal["pass", "fail", "warn", "skipped"]
    reason: str | None = None


class FinalOutputEvent(_BaseEvent):
    """The final response visible to the user."""

    type: Literal["final_output"]
    text: str = ""


class ErrorEvent(_BaseEvent):
    """An agent/runtime error event."""

    type: Literal["error"]
    message: str = Field(min_length=1)
    category: str | None = None


TraceEvent = Annotated[
    ModelCallEvent
    | ToolCallEvent
    | ToolResultEvent
    | RetrievalEvent
    | GuardrailEvent
    | FinalOutputEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
TraceEventAdapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class AgentTrace(_StrictModel):
    """Imported trace for one prompt/example/model side."""

    run_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    role: TraceRole
    events: list[TraceEvent] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCallEvent]:
        """Return tool-call events in sequence order."""
        return [event for event in self.events if isinstance(event, ToolCallEvent)]

    @property
    def final_outputs(self) -> list[FinalOutputEvent]:
        """Return final-output events in sequence order."""
        return [event for event in self.events if isinstance(event, FinalOutputEvent)]

    @model_validator(mode="after")
    def _normalize_and_validate_events(self) -> Self:
        indices = [event.sequence_index for event in self.events]
        if len(indices) != len(set(indices)):
            duplicates = sorted({index for index in indices if indices.count(index) > 1})
            raise ValueError(f"duplicate sequence_index in trace: {duplicates}")

        self.events.sort(key=lambda event: event.sequence_index)

        seen_call_ids: set[str] = set()
        for event in self.events:
            if isinstance(event, ToolCallEvent) and event.call_id is not None:
                seen_call_ids.add(event.call_id)
            if (
                isinstance(event, ToolResultEvent)
                and event.call_id is not None
                and event.call_id not in seen_call_ids
            ):
                raise ValueError(f"unknown tool_result call_id: {event.call_id}")
        return self


__all__ = [
    "AgentTrace",
    "ErrorEvent",
    "FinalOutputEvent",
    "GuardrailEvent",
    "ModelCallEvent",
    "RetrievalEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TraceEvent",
    "TraceEventAdapter",
    "TraceRole",
    "UtcTimestamp",
]
