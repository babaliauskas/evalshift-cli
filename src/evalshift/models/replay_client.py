"""A :class:`ModelClient` that replays canned responses from a fixtures file.

Used by ``evalshift run --offline`` so the showcase examples (and demos
without API keys) can run end-to-end with deterministic, reviewable
results. The orchestrator instantiates a :class:`ReplayClient` and passes
it through ``client=`` to ``run_orchestrator``; everything else in the
pipeline is unchanged.

Fixture format (JSONL, one record per line):

    {
      "model": "<canonical model id>",
      "match": "<substring that must appear in the rendered prompt>",
      "kind": "text" | "tools",
      "result": { ... }
    }

For ``kind: "text"``, ``result`` carries the fields of
:class:`CompletionResult` (``text``, ``input_tokens``, ``output_tokens``,
``cost_usd``, ``latency_ms`` — all optional except ``text``).

For ``kind: "tools"``, ``result`` carries:

    {
      "calls": [{"tool_name": "...", "arguments": {...}}, ...],
      "final_text": "..." | null,
      "raised_refusal": false,
      "refusal_text": null,
      "input_tokens": 0, "output_tokens": 0,
      "cost_usd": 0.0, "latency_ms": 0
    }

Matching: a fixture is selected when ``model`` equals the canonical id
the orchestrator dispatched and ``match`` is a substring of the rendered
prompt. Exactly one fixture must match — zero or two raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalshift.evaluators.tool_models import ToolCall, ToolSpec, ToolTrace
from evalshift.models.client import (
    CompletionResult,
    ModelClient,
    ModelClientError,
    ToolCompletionResult,
)


class ReplayError(ModelClientError):
    """Raised when a fixture lookup fails (no match, ambiguous match, bad shape)."""


@dataclass(frozen=True, slots=True)
class _Fixture:
    model: str
    match: str
    kind: str  # "text" | "tools"
    result: dict[str, Any]
    line_no: int


class ReplayClient(ModelClient):
    """A :class:`ModelClient` that returns canned responses from a JSONL file.

    Args:
        fixtures_path: Path to a JSONL fixtures file. Each line is one
            fixture record (see module docstring for shape).
    """

    def __init__(self, fixtures_path: Path | str) -> None:
        # Skip ``super().__init__`` retry policy — replay never retries.
        self._fixtures: list[_Fixture] = _load_fixtures(Path(fixtures_path))
        self._fixtures_path = Path(fixtures_path)

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        fix = self._find(model=model, prompt=prompt, kind="text")
        r = fix.result
        return CompletionResult(
            text=str(r.get("text", "")),
            model_id=model,
            input_tokens=int(r.get("input_tokens", 0)),
            output_tokens=int(r.get("output_tokens", 0)),
            cost_usd=float(r.get("cost_usd", 0.0)),
            latency_ms=int(r.get("latency_ms", 0)),
        )

    async def complete_with_tools(
        self,
        *,
        model: str,
        prompt: str,
        tools: list[ToolSpec],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ToolCompletionResult:
        fix = self._find(model=model, prompt=prompt, kind="tools")
        r = fix.result
        trace = _build_trace(r, fixture_line=fix.line_no)
        return ToolCompletionResult(
            trace=trace,
            model_id=model,
            input_tokens=int(r.get("input_tokens", 0)),
            output_tokens=int(r.get("output_tokens", 0)),
            cost_usd=float(r.get("cost_usd", 0.0)),
            latency_ms=int(r.get("latency_ms", 0)),
            raw_provider_response={},
        )

    def _find(self, *, model: str, prompt: str, kind: str) -> _Fixture:
        candidates = [
            f for f in self._fixtures if f.model == model and f.kind == kind and f.match in prompt
        ]
        if len(candidates) == 1:
            return candidates[0]
        preview = prompt[:160].replace("\n", " ")
        if not candidates:
            raise ReplayError(
                f"no {kind} fixture matched (model={model!r}, prompt~={preview!r}) "
                f"in {self._fixtures_path}",
            )
        lines = ", ".join(str(c.line_no) for c in candidates)
        raise ReplayError(
            f"{len(candidates)} {kind} fixtures matched (model={model!r}, "
            f"prompt~={preview!r}) at lines {lines} in {self._fixtures_path} "
            f"— make 'match' substrings unique",
        )


def _load_fixtures(path: Path) -> list[_Fixture]:
    if not path.exists():
        raise ReplayError(f"fixtures file not found: {path}")
    out: list[_Fixture] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            try:
                fix = _Fixture(
                    model=str(rec["model"]),
                    match=str(rec["match"]),
                    kind=str(rec["kind"]),
                    result=dict(rec["result"]),
                    line_no=line_no,
                )
            except KeyError as exc:
                raise ReplayError(
                    f"{path}:{line_no}: fixture missing required field {exc.args[0]!r}",
                ) from exc
            if fix.kind not in {"text", "tools"}:
                raise ReplayError(
                    f"{path}:{line_no}: kind must be 'text' or 'tools', got {fix.kind!r}",
                )
            out.append(fix)
    if not out:
        raise ReplayError(f"fixtures file is empty: {path}")
    return out


def _build_trace(result: dict[str, Any], *, fixture_line: int) -> ToolTrace:
    raw_calls = result.get("calls") or []
    calls: list[ToolCall] = []
    for idx, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            raise ReplayError(
                f"fixture line {fixture_line}: calls[{idx}] must be an object",
            )
        calls.append(
            ToolCall(
                tool_name=str(raw["tool_name"]),
                arguments=dict(raw.get("arguments") or {}),
                call_id=raw.get("call_id"),
                parent_call_id=raw.get("parent_call_id"),
                sequence_index=int(raw.get("sequence_index", idx)),
            ),
        )
    return ToolTrace(
        calls=calls,
        final_text=result.get("final_text"),
        raised_refusal=bool(result.get("raised_refusal", False)),
        refusal_text=result.get("refusal_text"),
    )


__all__ = ["ReplayClient", "ReplayError"]
