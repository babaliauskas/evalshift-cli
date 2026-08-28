"""A :class:`ModelClient` test double that replays canned responses from a fixtures file.

Used by ``tests/integration/test_conversation_pipeline.py`` to exercise the
full capture -> sync -> run -> evaluate -> analyze -> report pipeline
deterministically, without a real model call. There is no CLI ``--offline``
flag -- ``evalshift run`` / ``evalshift all`` always call a real model -- so
this class is injected directly through ``client=`` to ``run_orchestrator``
(``src/evalshift/runner/orchestrator.py``); everything else in the pipeline
is unchanged.

Fixture format (JSONL, one record per line):

    {
      "model": "<canonical model id>",
      "match": "<substring that must appear in the rendered prompt>",
      "kind": "text" | "tools",
      "result": { ... }
    }

For ``kind: "text"``, ``result`` carries the fields of
:class:`CompletionResult` (``text``, ``input_tokens``, ``output_tokens``,
``cost_usd``, ``latency_ms``, ``finish_reason`` — all optional except
``text``). Set ``"finish_reason": "length"`` to replay a truncated call.

For ``kind: "tools"``, ``result`` carries:

    {
      "calls": [{"tool_name": "...", "arguments": {...}}, ...],
      "final_text": "..." | null,
      "raised_refusal": false,
      "refusal_text": null,
      "input_tokens": 0, "output_tokens": 0,
      "cost_usd": 0.0, "latency_ms": 0,
      "finish_reason": null
    }

Matching: a fixture is selected when ``model`` equals the canonical id
the orchestrator dispatched and ``match`` is a substring of the rendered
prompt. Exactly one fixture must match — zero or two raise.

Message-mode calls (``complete_messages`` / ``complete_messages_with_tools``,
used by the orchestrator for multi-turn examples that carry a ``history``
prefix) match ``match`` against the content of the *last* ``user``-role
message only, never the whole flattened transcript — under teacher forcing
a later turn's history repeats earlier turns' text verbatim, so matching
against the full transcript would let a turn-1 fixture ambiguously match a
turn-2 call. Note this still doesn't fully disambiguate: identical
follow-up turns across different conversations (e.g. two conversations
both continuing with "yes") collide on the same last-user-message text,
and the usual exactly-one-match error surfaces that collision so the
fixture author can pick a more specific ``match`` string.
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
        # Base init is required for client state the orchestrator reads
        # (``temperature_rejected_models``); the retry policy it also sets
        # is inert here — replay never retries.
        super().__init__()
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
            finish_reason=_opt_str(r.get("finish_reason")),
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
            finish_reason=_opt_str(r.get("finish_reason")),
        )

    async def complete_messages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        target = _last_user_content(messages)
        fix = self._find(model=model, prompt=target, kind="text")
        r = fix.result
        return CompletionResult(
            text=str(r.get("text", "")),
            model_id=model,
            input_tokens=int(r.get("input_tokens", 0)),
            output_tokens=int(r.get("output_tokens", 0)),
            cost_usd=float(r.get("cost_usd", 0.0)),
            latency_ms=int(r.get("latency_ms", 0)),
            finish_reason=_opt_str(r.get("finish_reason")),
        )

    async def complete_messages_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ToolCompletionResult:
        target = _last_user_content(messages)
        fix = self._find(model=model, prompt=target, kind="tools")
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
            finish_reason=_opt_str(r.get("finish_reason")),
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


def _opt_str(value: Any) -> str | None:
    """Coerce an optional fixture field to ``str | None`` (``None`` passthrough)."""
    return None if value is None else str(value)


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last ``user``-role message in ``messages``.

    This is the fixture match target for message-mode calls — see the
    module docstring for the teacher-forcing rationale. Returns an empty
    string when there is no ``user`` message (defensive; should not
    happen in practice), which naturally falls through to the zero-match
    error path in :meth:`ReplayClient._find`.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


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
