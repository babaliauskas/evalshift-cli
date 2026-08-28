"""JSONL loading and indexing helpers for imported agent traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalshift.traces.models import AgentTrace, TraceRole

TRACES_FILENAME: str = "traces.jsonl"


class TraceLoadError(ValueError):
    """Raised when an imported trace JSONL file cannot be loaded."""


@dataclass(frozen=True, slots=True)
class TracePair:
    """Source and target traces for one prompt/example pair."""

    prompt_id: str
    example_id: str
    source: AgentTrace
    target: AgentTrace


TraceKey = tuple[str, str, TraceRole]


def load_traces_jsonl(path: Path) -> list[AgentTrace]:
    """Load agent traces from JSONL with path/line-numbered errors."""
    traces: list[AgentTrace] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TraceLoadError(f"{path}: {exc}") from exc

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceLoadError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        try:
            traces.append(AgentTrace.model_validate(payload))
        except ValidationError as exc:
            raise TraceLoadError(f"{path}:{line_no}: invalid trace: {exc}") from exc
        except ValueError as exc:
            raise TraceLoadError(f"{path}:{line_no}: invalid trace: {exc}") from exc
    return traces


def write_traces_jsonl(path: Path, traces: list[AgentTrace]) -> None:
    """Write normalized agent traces as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(traces, key=lambda t: (t.prompt_id, t.example_id, t.role))
    with path.open("w", encoding="utf-8") as fh:
        for trace in ordered:
            fh.write(trace.model_dump_json())
            fh.write("\n")


def index_traces(traces: list[AgentTrace]) -> dict[TraceKey, AgentTrace]:
    """Index traces by ``(prompt_id, example_id, role)``."""
    out: dict[TraceKey, AgentTrace] = {}
    for trace in traces:
        key = (trace.prompt_id, trace.example_id, trace.role)
        if key in out:
            raise TraceLoadError(
                f"duplicate trace for prompt={trace.prompt_id!r} "
                f"example={trace.example_id!r} role={trace.role!r}",
            )
        out[key] = trace
    return out


def pairs_for_prompt_examples(
    traces: list[AgentTrace],
    *,
    prompt_examples: list[tuple[str, str]],
) -> list[TracePair]:
    """Return complete source/target trace pairs for the requested keys."""
    indexed = index_traces(traces)
    pairs: list[TracePair] = []
    for prompt_id, example_id in prompt_examples:
        source = indexed.get((prompt_id, example_id, "source"))
        target = indexed.get((prompt_id, example_id, "target"))
        if source is None or target is None:
            continue
        pairs.append(
            TracePair(
                prompt_id=prompt_id,
                example_id=example_id,
                source=source,
                target=target,
            ),
        )
    return pairs


__all__ = [
    "TRACES_FILENAME",
    "TraceKey",
    "TraceLoadError",
    "TracePair",
    "index_traces",
    "load_traces_jsonl",
    "pairs_for_prompt_examples",
    "write_traces_jsonl",
]
