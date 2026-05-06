"""Provider-agnostic parsing of tool-use responses into :class:`ToolTrace`.

LiteLLM normalises most provider responses to OpenAI's shape, but
Anthropic-native responses still have a ``choices[0].message.content``
list of typed blocks (``text`` / ``tool_use`` / ``refusal``). This
module dispatches on the model id, picks the right parser, and produces
a uniform :class:`ToolTrace`.

Errors at this layer are :class:`ToolParseError` and carry the raw
response so the smoke-test script can dump it for fixture capture.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from aimigrate.evaluators.tool_models import ToolCall, ToolTrace

Provider = Literal["anthropic", "openai", "gemini"]


class ToolParseError(Exception):
    """Raised when a provider response can't be parsed into a :class:`ToolTrace`.

    Attributes:
        provider: ``"anthropic"`` | ``"openai"`` | ``"gemini"``.
        reason: Short human-readable explanation of what failed.
        raw: The raw response payload, for fixture capture / debugging.
    """

    def __init__(self, provider: str, reason: str, raw: Any) -> None:
        super().__init__(f"{provider}: {reason}")
        self.provider = provider
        self.reason = reason
        self.raw = raw


def detect_provider(model_id: str) -> Provider:
    """Map a LiteLLM-style model id to a provider name.

    Mirrors the prefix-inference logic in
    :func:`aimigrate.models.registry._infer_provider_and_canonical` so
    the parser, registry, and report all agree on what counts as
    ``"anthropic"`` vs. ``"openai"`` vs. ``"gemini"``.

    Raises:
        ToolParseError: If the model id can't be mapped to one of the
            three supported providers. v0.2 doesn't ship parsers for
            anything else; users get a clear error rather than silent
            mis-routing.
    """
    lowered = model_id.lower()
    if model_id.startswith("anthropic/") or "claude" in lowered:
        return "anthropic"
    if model_id.startswith("openai/") or lowered.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    if model_id.startswith("gemini/") or "gemini" in lowered:
        return "gemini"
    raise ToolParseError(
        provider="unknown",
        reason=f"cannot detect provider for model id: {model_id!r}",
        raw=model_id,
    )


def parse_response_to_trace(
    response: dict[str, Any],
    *,
    provider: str,
    model_id: str,
) -> ToolTrace:
    """Top-level dispatcher: route to the right provider parser.

    Args:
        response: The raw LiteLLM response dict (i.e. the result of
            ``await litellm.acompletion(...)``, ``model_dump()``-style).
        provider: One of ``"anthropic"``, ``"openai"``, ``"gemini"``.
        model_id: The canonical model id; included in error messages.

    Returns:
        A normalised :class:`ToolTrace`.

    Raises:
        ToolParseError: If the response shape isn't recognised or the
            provider name isn't supported.
    """
    if provider == "anthropic":
        return _parse_anthropic(response, model_id)
    if provider == "openai":
        return _parse_openai(response, model_id)
    if provider == "gemini":
        return _parse_gemini(response, model_id)
    raise ToolParseError(
        provider=provider,
        reason=f"unsupported provider name: {provider!r}",
        raw=response,
    )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _parse_anthropic(response: dict[str, Any], model_id: str) -> ToolTrace:
    """Parse a LiteLLM Anthropic response into a :class:`ToolTrace`.

    LiteLLM sometimes normalises Anthropic's content list into OpenAI's
    ``tool_calls`` shape (depending on version + flags). We handle both:
    if ``message.tool_calls`` is present we delegate to the OpenAI
    parser; otherwise we walk Anthropic's native ``message.content``
    block list (``text`` / ``tool_use`` / ``refusal``).
    """
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolParseError(
            provider="anthropic",
            reason=f"unexpected response shape: {exc}",
            raw=response,
        ) from exc

    if isinstance(message, dict) and message.get("tool_calls"):
        return _parse_openai_shape(message, model_id, provider_for_errors="anthropic")

    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        # Plain text response from Anthropic — text-only, no tools.
        return ToolTrace(final_text=content if content else None)
    if content is None:
        return ToolTrace()
    if not isinstance(content, list):
        raise ToolParseError(
            provider="anthropic",
            reason=f"expected list or str for content, got {type(content).__name__}",
            raw=response,
        )

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    raised_refusal = False
    refusal_text: str | None = None

    for idx, block in enumerate(content):
        if not isinstance(block, dict):
            raise ToolParseError(
                provider="anthropic",
                reason=f"content block at index {idx} is not a dict",
                raw=response,
            )
        block_type = block.get("type")
        if block_type == "tool_use":
            calls.append(
                ToolCall(
                    tool_name=block.get("name", ""),
                    arguments=dict(block.get("input", {})),
                    call_id=block.get("id"),
                    parent_call_id=None,
                    sequence_index=idx,
                ),
            )
        elif block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "refusal":
            raised_refusal = True
            refusal_text = block.get("text") or refusal_text
        # Unknown block types are ignored (defensive against future Anthropic additions).

    return ToolTrace(
        calls=_renumber(calls),
        final_text="\n".join(p for p in text_parts if p) if text_parts else None,
        raised_refusal=raised_refusal,
        refusal_text=refusal_text,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _parse_openai(response: dict[str, Any], model_id: str) -> ToolTrace:
    """Parse a LiteLLM OpenAI response into a :class:`ToolTrace`."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolParseError(
            provider="openai",
            reason=f"unexpected response shape: {exc}",
            raw=response,
        ) from exc
    if not isinstance(message, dict):
        raise ToolParseError(
            provider="openai",
            reason=f"expected message dict, got {type(message).__name__}",
            raw=response,
        )
    return _parse_openai_shape(message, model_id, provider_for_errors="openai")


def _parse_openai_shape(
    message: dict[str, Any],
    model_id: str,
    *,
    provider_for_errors: str,
) -> ToolTrace:
    """Shared logic for OpenAI-shaped responses.

    Reused by both ``_parse_openai`` and ``_parse_anthropic`` (when
    LiteLLM normalises Anthropic to OpenAI shape) and ``_parse_gemini``.
    """
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ToolParseError(
            provider=provider_for_errors,
            reason=f"expected list for tool_calls, got {type(tool_calls).__name__}",
            raw=message,
        )

    final_text = message.get("content")
    if isinstance(final_text, str) and not final_text:
        final_text = None

    calls: list[ToolCall] = []
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            raise ToolParseError(
                provider=provider_for_errors,
                reason=f"tool_calls[{idx}] is not a dict",
                raw=message,
            )
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            raise ToolParseError(
                provider=provider_for_errors,
                reason=f"tool_calls[{idx}].function is not a dict",
                raw=message,
            )
        tool_name = fn.get("name", "")
        if not tool_name:
            raise ToolParseError(
                provider=provider_for_errors,
                reason=f"tool_calls[{idx}] missing function.name",
                raw=message,
            )

        # OpenAI / LiteLLM may serialise arguments as a JSON string.
        # We decode defensively; malformed JSON is preserved with a
        # ``_parse_error`` flag for downstream evaluators to handle.
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw, "_parse_error": True}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {"_raw": args_raw, "_parse_error": True}

        calls.append(
            ToolCall(
                tool_name=tool_name,
                arguments=args,
                call_id=tc.get("id"),
                parent_call_id=None,
                sequence_index=idx,
            ),
        )

    refusal = message.get("refusal")
    raised_refusal = bool(refusal) if not tool_calls else False
    refusal_text = refusal if isinstance(refusal, str) else None

    return ToolTrace(
        calls=calls,
        final_text=final_text if isinstance(final_text, str) else None,
        raised_refusal=raised_refusal,
        refusal_text=refusal_text,
    )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _parse_gemini(response: dict[str, Any], model_id: str) -> ToolTrace:
    """Parse a LiteLLM Gemini response.

    LiteLLM normalises Gemini's ``functionCall`` parts into OpenAI's
    ``tool_calls`` shape, so we delegate. If LiteLLM ever changes this,
    add a native parser here and dispatch on a marker.
    """
    return _parse_openai(response, model_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renumber(calls: list[ToolCall]) -> list[ToolCall]:
    """Re-assign 0-indexed ``sequence_index`` to a list of calls.

    Used after we've potentially skipped non-tool content blocks in an
    Anthropic response — we want the indices to be dense in the trace,
    not sparse with gaps where text blocks were.
    """
    return [
        ToolCall(
            tool_name=c.tool_name,
            arguments=c.arguments,
            call_id=c.call_id,
            parent_call_id=c.parent_call_id,
            sequence_index=i,
        )
        for i, c in enumerate(calls)
    ]


__all__ = [
    "Provider",
    "ToolParseError",
    "detect_provider",
    "parse_response_to_trace",
]
