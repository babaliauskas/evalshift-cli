"""Async wrapper over LiteLLM's completion API.

This module is the single chokepoint for outbound LLM calls. Every other
part of AIMigrate that needs to talk to a model goes through
:class:`ModelClient`, which gives us:

* **Uniform error mapping** — provider-specific exceptions are normalised
  into :class:`RateLimitError`, :class:`AuthError`, and
  :class:`ModelError` so callers don't need a `try`/`except` per provider.
* **Bounded retry with jittered backoff** — at the request level rather
  than relying on LiteLLM's built-in retry, because we want to expose the
  *original* error type after exhausting retries.
* **Cost + token bookkeeping** — every successful call returns the
  measured wall time, token counts, and dollar cost in a single dataclass
  so downstream code (cache, orchestrator, reports) doesn't recompute.

The cache is intentionally **not** wired in here. The orchestrator
(Phase 4) is the right layer to coordinate cache reads/writes around the
client; keeping the client cache-agnostic makes ``aimigrate test-call``
trivial and tests faster.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, cast

import litellm

from aimigrate.evaluators.tool_models import ToolSpec, ToolTrace
from aimigrate.evaluators.tool_parser import (
    ToolParseError,
    detect_provider,
    parse_response_to_trace,
)
from aimigrate.models.registry import resolve_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class ModelClientError(Exception):
    """Base class for every error raised by :class:`ModelClient`."""


class RateLimitError(ModelClientError):
    """Provider returned a 429-like rate-limit response."""


class AuthError(ModelClientError):
    """Provider rejected our credentials (missing or invalid API key)."""


class ModelError(ModelClientError):
    """Catch-all for any other provider-side or transport failure."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCompletionResult:
    """Outcome of a successful tool-aware LLM call.

    Mirrors :class:`CompletionResult` for cost/token/latency bookkeeping
    but carries a normalised :class:`ToolTrace` instead of raw text. The
    final assistant text (when the model produced any) is available on
    ``trace.final_text``.

    Attributes:
        trace: Provider-agnostic tool trace.
        model_id: The canonical model id used.
        input_tokens / output_tokens: From the provider response.
        cost_usd: Dollar cost reported by ``litellm.completion_cost``.
        latency_ms: Wall-clock time of the call.
        raw_provider_response: The raw response dict, for fixture
            capture and debugging. Not persisted by the orchestrator.
    """

    trace: ToolTrace
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    raw_provider_response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """One successful LLM call's worth of data.

    Attributes:
        text: The model's response text.
        model_id: The canonical model id used (post-alias resolution).
        input_tokens: Prompt-side token count.
        output_tokens: Completion-side token count.
        cost_usd: Dollar cost reported by ``litellm.completion_cost``.
            May be ``0.0`` if LiteLLM's pricing table doesn't know the
            model — we never block on that.
        latency_ms: Wall-clock time of the live call (excludes retries
            that the underlying provider transparently absorbed).
    """

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff with full jitter.

    The defaults follow the AWS "full jitter" recipe: pick a random delay
    between 0 and ``min(cap, base * 2**attempt)`` seconds. This is the
    common-sense default for talking to rate-limited APIs.
    """

    max_attempts: int = 3
    base_seconds: float = 1.0
    cap_seconds: float = 30.0

    def delay(self, attempt: int) -> float:
        """Compute the sleep before retry attempt ``attempt`` (1-indexed)."""
        upper = min(self.cap_seconds, self.base_seconds * (2 ** (attempt - 1)))
        return random.uniform(0, upper)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ModelClient:
    """Thin async wrapper around :func:`litellm.acompletion`.

    Args:
        retry_policy: Override the default retry policy. Pass
            ``RetryPolicy(max_attempts=1)`` to disable retries.
    """

    def __init__(self, *, retry_policy: RetryPolicy | None = None) -> None:
        self._retry = retry_policy or RetryPolicy()

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Make a single completion call.

        Args:
            model: A canonical id or alias from
                :mod:`aimigrate.models.registry`. Resolved to the
                canonical id before dispatch so reports see consistent
                names.
            prompt: The fully-rendered prompt (already substituted).
            temperature: Sampling temperature. ``None`` uses the model's
                registered default (which is 0 for every entry today —
                determinism matters for evaluation).
            max_tokens: Completion length cap. ``None`` uses the
                registered default.
            extra: Provider-specific kwargs forwarded to
                ``litellm.acompletion`` (e.g. ``response_format`` for the
                judge to demand JSON).

        Returns:
            A :class:`CompletionResult` with the response text plus
            cost/token bookkeeping.

        Raises:
            RateLimitError: 429 from the provider.
            AuthError: bad/missing credentials.
            ModelError: any other failure.
        """
        meta = resolve_model(model)
        canonical = meta.id
        temp = meta.default_temperature if temperature is None else temperature
        mt = meta.default_max_tokens if max_tokens is None else max_tokens
        kwargs: dict[str, Any] = {
            "model": canonical,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": mt,
        }
        if extra:
            kwargs.update(extra)

        last_exc: Exception | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            start = time.perf_counter()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                mapped = _map_exception(exc)
                # Auth errors are deterministic; don't waste retries on them.
                if isinstance(mapped, AuthError):
                    raise mapped from exc
                last_exc = mapped
                if attempt >= self._retry.max_attempts:
                    raise mapped from exc
                delay = self._retry.delay(attempt)
                log.warning(
                    "model %s attempt %d failed (%s); retrying in %.2fs",
                    canonical,
                    attempt,
                    mapped.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            return _build_result(canonical, response, latency_ms)

        # Loop exits cleanly only via return; if we ever reach here, raise.
        raise ModelError(f"exhausted retries for model {canonical}") from last_exc

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
        """Call ``model`` with ``tools`` attached and parse the response.

        The same retry / error-mapping policy as :meth:`complete`
        applies; on success the response is funnelled through
        :func:`parse_response_to_trace` for provider-agnostic
        normalisation.

        Args:
            model: Canonical id or alias from
                :mod:`aimigrate.models.registry`.
            prompt: User-side prompt text.
            tools: Tool specs to expose to the model. Serialised
                per-provider (``to_anthropic`` for Anthropic models,
                ``to_openai`` for everything else — Gemini accepts the
                OpenAI shape via LiteLLM).
            temperature: Sampling temperature. ``None`` uses the model's
                registered default.
            max_tokens: Completion length cap. ``None`` uses the
                registered default.
            extra: Provider-specific kwargs forwarded to
                ``litellm.acompletion``.

        Returns:
            A :class:`ToolCompletionResult` with the parsed
            :class:`ToolTrace` and full cost/token/latency bookkeeping.

        Raises:
            RateLimitError / AuthError / ModelError: Same semantics as
                :meth:`complete`.
            ModelError: If the response can't be parsed into a
                :class:`ToolTrace` (wraps :class:`ToolParseError`).
        """
        meta = resolve_model(model)
        canonical = meta.id
        temp = meta.default_temperature if temperature is None else temperature
        mt = meta.default_max_tokens if max_tokens is None else max_tokens

        provider = detect_provider(canonical)
        tools_payload = [
            t.to_anthropic() if provider == "anthropic" else t.to_openai() for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": canonical,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": mt,
            "tools": tools_payload,
        }
        if extra:
            kwargs.update(extra)

        last_exc: Exception | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            start = time.perf_counter()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                mapped = _map_exception(exc)
                if isinstance(mapped, AuthError):
                    raise mapped from exc
                last_exc = mapped
                if attempt >= self._retry.max_attempts:
                    raise mapped from exc
                delay = self._retry.delay(attempt)
                log.warning(
                    "model %s (tools) attempt %d failed (%s); retrying in %.2fs",
                    canonical,
                    attempt,
                    mapped.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            return _build_tool_result(
                canonical=canonical,
                response=response,
                latency_ms=latency_ms,
                provider=provider,
            )

        raise ModelError(f"exhausted retries for model {canonical}") from last_exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_result(
    canonical: str,
    response: Any,
    latency_ms: int,
) -> CompletionResult:
    text = _extract_text(response)
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cost_usd = _safe_cost(response)
    return CompletionResult(
        text=text,
        model_id=canonical,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def _build_tool_result(
    *,
    canonical: str,
    response: Any,
    latency_ms: int,
    provider: str,
) -> ToolCompletionResult:
    """Parse a tools-aware response into a :class:`ToolCompletionResult`.

    LiteLLM responses can be SDK objects or dicts depending on version.
    We coerce to a dict view (``model_dump`` if available, otherwise
    ``dict()`` / direct cast) before handing to the parser, so the
    parser can stay strictly dict-shaped.
    """
    raw = _response_as_dict(response)
    try:
        trace = parse_response_to_trace(raw, provider=provider, model_id=canonical)
    except ToolParseError as exc:
        raise ModelError(f"failed to parse tool response: {exc}") from exc

    usage = raw.get("usage") or {}
    return ToolCompletionResult(
        trace=trace,
        model_id=canonical,
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cost_usd=_safe_cost(response),
        latency_ms=latency_ms,
        raw_provider_response=raw,
    )


def _response_as_dict(response: Any) -> dict[str, Any]:
    """Coerce a LiteLLM response object or dict into a dict view."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return cast(dict[str, Any], response.model_dump())
    if hasattr(response, "dict"):
        return cast(dict[str, Any], response.dict())
    # Last-resort: build a minimal dict from the choices/usage attrs.
    return {
        "choices": getattr(response, "choices", []),
        "usage": getattr(response, "usage", {}),
    }


def _extract_text(response: Any) -> str:
    """Pull the assistant message text out of a LiteLLM response object."""
    try:
        choice = response.choices[0]
        message = getattr(choice, "message", None) or choice["message"]
        content = getattr(message, "content", None)
        if content is None:
            content = message.get("content") if isinstance(message, dict) else None
        if content is None:
            return ""
        return cast(str, content)
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ModelError(f"could not extract text from response: {exc}") from exc


def _safe_cost(response: Any) -> float:
    """Return ``litellm.completion_cost`` or 0 if the model isn't priced.

    LiteLLM raises if it doesn't know the price; we treat unknown costs
    as $0 rather than failing the whole call (the user already got a
    response — billing them an exception would be ridiculous).
    """
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception as exc:
        log.debug("cost lookup failed (treating as $0): %s", exc)
        return 0.0


def _map_exception(exc: BaseException) -> ModelClientError:
    """Map a LiteLLM/provider exception to AIMigrate's error hierarchy."""
    name = type(exc).__name__
    msg = str(exc)
    if name in {"RateLimitError", "Timeout", "APITimeoutError"}:
        return RateLimitError(msg or name)
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return AuthError(msg or name)
    return ModelError(msg or name)


__all__ = [
    "AuthError",
    "CompletionResult",
    "ModelClient",
    "ModelClientError",
    "ModelError",
    "RateLimitError",
    "RetryPolicy",
    "ToolCompletionResult",
]
