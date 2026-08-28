"""Agent trace models, loading, and diffing."""

from __future__ import annotations

from evalshift.traces.diff import TraceDiff, TraceDiffItem, diff_traces
from evalshift.traces.loader import (
    TRACES_FILENAME,
    TraceLoadError,
    TracePair,
    index_traces,
    load_traces_jsonl,
    pairs_for_prompt_examples,
    write_traces_jsonl,
)
from evalshift.traces.models import (
    AgentTrace,
    ErrorEvent,
    FinalOutputEvent,
    GuardrailEvent,
    ModelCallEvent,
    RetrievalEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)

__all__ = [
    "TRACES_FILENAME",
    "AgentTrace",
    "ErrorEvent",
    "FinalOutputEvent",
    "GuardrailEvent",
    "ModelCallEvent",
    "RetrievalEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "TraceDiff",
    "TraceDiffItem",
    "TraceEvent",
    "TraceLoadError",
    "TracePair",
    "diff_traces",
    "index_traces",
    "load_traces_jsonl",
    "pairs_for_prompt_examples",
    "write_traces_jsonl",
]
