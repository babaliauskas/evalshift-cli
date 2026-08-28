"""Async wrapper over LiteLLM's completion API.

This module is the single chokepoint for outbound LLM calls. Every other
part of EvalShift that needs to talk to a model goes through
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
client; keeping the client cache-agnostic makes ``evalshift test-call``
trivial and tests faster.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import litellm

from evalshift.evaluators.tool_models import ToolSpec, ToolTrace
from evalshift.evaluators.tool_parser import (
    ToolParseError,
    detect_provider,
    parse_response_to_trace,
)
from evalshift.models.registry import resolve_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LiteLLM logging
# ---------------------------------------------------------------------------


class _DedupeWarningsFilter(logging.Filter):
    """Let each distinct LiteLLM warning through once per process.

    LiteLLM logs some warnings per call rather than per condition — its
    Gemini 3 ``temperature`` deprecation notice produced ~290 identical lines
    in a single run, burying everything else.

    Deduping rather than silencing keeps the signal: a deprecation nobody has
    seen yet still appears, exactly once. Only ``WARNING`` records are
    considered; ERROR and below pass through untouched so nothing indicating
    a real failure is ever dropped.
    """

    def __init__(self) -> None:
        super().__init__()
        # Distinct warning strings only — LiteLLM emits a handful, so this
        # stays small for the life of the process.
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` for a repeat WARNING, ``True`` for anything else."""
        if record.levelno != logging.WARNING:
            return True
        message = record.getMessage()
        if message in self._seen:
            return False
        self._seen.add(message)
        return True


class _LateBoundStderr:
    """Write-through proxy that resolves ``sys.stderr`` at write time.

    LiteLLM's ``StreamHandler`` captures the ``sys.stderr`` *object* when the
    library is imported, so every later redirection of ``sys.stderr`` is
    bypassed — including the one ``rich.live.Live`` installs to keep a live
    region coherent. A warning arriving mid-frame wrote straight over the
    ``evalshift all`` pipeline block, and Rich then redrew the block below the
    damage: one run, two pipeline blocks on screen.

    Resolving the stream per write hands those lines back to Rich, which
    clears the live region, prints the warning above it, and redraws. Outside
    a live region (and when stderr is not a terminal, where Rich installs no
    redirect at all) the write lands on ``sys.stderr`` exactly as before.
    """

    def write(self, data: str) -> int:
        """Write ``data`` to whatever ``sys.stderr`` is bound to right now."""
        return sys.stderr.write(data)

    def flush(self) -> None:
        """Flush whatever ``sys.stderr`` is bound to right now."""
        sys.stderr.flush()

    def isatty(self) -> bool:
        """Report the current ``sys.stderr``'s terminal-ness."""
        return sys.stderr.isatty()


def _console_stream_handlers(logger: logging.Logger) -> list[logging.StreamHandler[Any]]:
    """The logger's handlers that write to the console, and only those.

    A ``FileHandler`` is a ``StreamHandler`` too, and a log file must keep
    receiving its records — so only handlers currently pointed at
    ``sys.stderr``, the interpreter's original ``sys.__stderr__``, or our own
    :class:`_LateBoundStderr` proxy count as console handlers.
    """
    console_streams = {sys.stderr, sys.__stderr__}
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and (isinstance(handler.stream, _LateBoundStderr) or handler.stream in console_streams)
    ]


def _late_bind_stderr_handlers(litellm_log: logging.Logger) -> None:
    """Repoint LiteLLM's console handlers at :class:`_LateBoundStderr`."""
    for handler in _console_stream_handlers(litellm_log):
        if not isinstance(handler.stream, _LateBoundStderr):
            handler.setStream(_LateBoundStderr())


class _DeferredWarningsHandler(logging.Handler):
    """Root handler backing :func:`deferred_console_warnings`.

    WARNING records go into the shared buffer; anything above WARNING is
    written straight to the *current* ``sys.stderr`` — an error explains a
    failure and must never be held back until the pipeline finishes. Records
    below WARNING never reach :meth:`emit` (the handler's level filters them).
    """

    def __init__(self, buffer: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.WARNING)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer a WARNING; write anything more severe through immediately."""
        if record.levelno < logging.ERROR:
            self._buffer.append(record)
        else:
            sys.stderr.write(record.getMessage() + "\n")


@contextmanager
def deferred_console_warnings() -> Iterator[list[logging.LogRecord]]:
    """Buffer console WARNING records for the duration of the block.

    ``evalshift all`` renders a pipeline block and a verdict; a warning
    arriving mid-run — LiteLLM's per-call deprecation notices, the insights
    generator's retry notes — used to print wherever the emitting call
    happened to be, splitting the output into fragments. Under this context
    manager those records collect in the yielded list instead, so the caller
    can print them as one section once the block is complete.

    Mechanics: LiteLLM's console handlers are detached (file handlers stay),
    letting its records propagate to the root logger like everyone else's,
    where a :class:`_DeferredWarningsHandler` buffers them. The same handler
    catches warnings from our own loggers, which previously reached stderr
    via ``logging.lastResort``. ERROR and above are never deferred — they
    are written to stderr the moment they happen. Everything is restored on
    exit, exception or not.
    """
    litellm_log = logging.getLogger("LiteLLM")
    root = logging.getLogger()
    buffer: list[logging.LogRecord] = []
    handler = _DeferredWarningsHandler(buffer)
    detached = _console_stream_handlers(litellm_log)
    for h in detached:
        litellm_log.removeHandler(h)
    root.addHandler(handler)
    try:
        yield buffer
    finally:
        root.removeHandler(handler)
        for h in detached:
            litellm_log.addHandler(h)


def _configure_litellm() -> None:
    """Tame LiteLLM's logging: dedupe its warnings, late-bind its stream.

    Called at import time, after ``import litellm`` has installed the
    library's own handler. Idempotent, because this module can legitimately
    be imported from several entry points: stacked filters would each keep
    their own ``_seen`` set, letting a warning through once per copy, and
    nested proxies would each add a layer of indirection for nothing.
    """
    litellm_log = logging.getLogger("LiteLLM")
    _late_bind_stderr_handlers(litellm_log)
    if any(isinstance(f, _DedupeWarningsFilter) for f in litellm_log.filters):
        return
    litellm_log.addFilter(_DedupeWarningsFilter())


_configure_litellm()


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
        finish_reason: The provider's normalised stop reason
            (``choices[0].finish_reason``). ``"length"`` means the output
            was truncated by the ``max_tokens`` cap.
    """

    trace: ToolTrace
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    raw_provider_response: dict[str, Any]
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """True when the provider cut the output off at the token cap."""
        return self.finish_reason == "length"


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
        finish_reason: The provider's normalised stop reason
            (``choices[0].finish_reason``). ``"length"`` means the output
            was truncated by the ``max_tokens`` cap.
    """

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        """True when the provider cut the output off at the token cap."""
        return self.finish_reason == "length"


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
        # Canonical ids of models that 400ed the VALUE of ``temperature``
        # (reasoning-tier models accept only their default). Once listed, a
        # model's calls omit the parameter entirely — one failed call per
        # model per process. See _is_temperature_value_rejection.
        self._temperature_rejected: set[str] = set()

    @property
    def temperature_rejected_models(self) -> frozenset[str]:
        """Canonical ids that rejected every non-default ``temperature`` value.

        Populated at dispatch time, from the provider's own 400. The
        orchestrator and the evaluate command merge this into the run
        state's ``non_deterministic_models`` so the report's banner covers
        runtime discoveries as well as the run-start capability probe.
        """
        return frozenset(self._temperature_rejected)

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

        Thin delegation to :meth:`complete_messages` wrapping ``prompt``
        in a single user-role message — see that method for the full
        retry / cost / error-mapping contract.

        Args:
            model: A canonical id or alias from
                :mod:`evalshift.models.registry`. Resolved to the
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
        return await self.complete_messages(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
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
        """Make a single completion call from a pre-built messages list.

        Unlike :meth:`complete`, callers control the full messages array
        (e.g. a multi-turn history prefix followed by the current-turn
        user message). ``messages`` is forwarded to
        ``litellm.acompletion`` verbatim — LiteLLM maps role names
        per-provider (e.g. a ``system`` message becomes Gemini's
        ``systemInstruction``), so no transformation happens here.

        Args:
            model: A canonical id or alias from
                :mod:`evalshift.models.registry`. Resolved to the
                canonical id before dispatch so reports see consistent
                names.
            messages: The full messages list to send, in LiteLLM/OpenAI
                message shape (``{"role": ..., "content": ...}``).
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
            "messages": messages,
            "temperature": temp,
            "max_tokens": mt,
            # drop_params only saves models LiteLLM's configs special-case
            # (o-series names). Other reasoning-tier models reject
            # temperature != 1 at the API with a 400; that case is handled
            # at dispatch — see _is_temperature_value_rejection and the
            # adaptation in _dispatch_with_retry.
            "drop_params": True,
        }
        if extra:
            kwargs.update(extra)

        response, latency_ms = await self._dispatch_with_retry(canonical, kwargs, log_suffix="")
        return _build_result(canonical, response, latency_ms)

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

        Thin delegation to :meth:`complete_messages_with_tools` wrapping
        ``prompt`` in a single user-role message.

        Args:
            model: Canonical id or alias from
                :mod:`evalshift.models.registry`.
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
        return await self.complete_messages_with_tools(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
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
        """Call ``model`` with ``tools`` attached from a pre-built messages list.

        Unlike :meth:`complete_with_tools`, callers control the full
        messages array (e.g. a multi-turn history prefix followed by the
        current-turn user message). ``messages`` is forwarded to
        ``litellm.acompletion`` verbatim. The same retry / error-mapping
        policy as :meth:`complete_messages` applies; on success the
        response is funnelled through :func:`parse_response_to_trace` for
        provider-agnostic normalisation.

        Args:
            model: Canonical id or alias from
                :mod:`evalshift.models.registry`.
            messages: The full messages list to send, in LiteLLM/OpenAI
                message shape (``{"role": ..., "content": ...}``).
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
            "messages": messages,
            "temperature": temp,
            "max_tokens": mt,
            "tools": tools_payload,
            # See complete_messages: drop_params plus the dispatch-time
            # temperature adaptation keep reasoning-tier models usable.
            "drop_params": True,
        }
        if extra:
            kwargs.update(extra)

        response, latency_ms = await self._dispatch_with_retry(
            canonical, kwargs, log_suffix=" (tools)"
        )
        return _build_tool_result(
            canonical=canonical,
            response=response,
            latency_ms=latency_ms,
            provider=provider,
        )

    async def _dispatch_with_retry(
        self,
        canonical: str,
        kwargs: dict[str, Any],
        *,
        log_suffix: str,
    ) -> tuple[Any, int]:
        """Shared retry/dispatch loop for the ``complete*`` methods.

        Calls ``litellm.acompletion(**kwargs)`` under the client's retry
        policy, mapping provider exceptions to EvalShift's error
        hierarchy. Auth errors short-circuit (no retry, since they're
        deterministic); everything else retries with jittered backoff up
        to ``max_attempts``.

        Args:
            canonical: The resolved canonical model id, for log messages.
            kwargs: Full kwargs to forward to ``litellm.acompletion``.
            log_suffix: Appended to the retry warning message to
                distinguish tool-aware calls (``" (tools)"``) from plain
                ones (``""``) — kept purely for log-message parity with
                the pre-refactor code.

        Returns:
            A ``(response, latency_ms)`` tuple for the successful call.

        One adaptation is layered on top of the retry policy: a provider
        400 that rejects the VALUE of ``temperature`` (reasoning-tier
        models accept only their default) pops the parameter and
        redispatches immediately, without consuming a retry attempt. The
        model id is memoized on the client so later calls omit the
        parameter before dispatch. The adaptation fires at most once per
        call — it requires ``temperature`` in the kwargs and removes it.

        Raises:
            RateLimitError / AuthError / ModelError: mapped provider
                failure once retries are exhausted (or immediately for
                auth errors).
        """
        if canonical in self._temperature_rejected:
            kwargs.pop("temperature", None)
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                if "temperature" in kwargs and _is_temperature_value_rejection(exc):
                    kwargs.pop("temperature")
                    if canonical not in self._temperature_rejected:
                        self._temperature_rejected.add(canonical)
                        log.warning(
                            "model %s rejects non-default temperature values; "
                            "resending without temperature — sampling for this "
                            "model is not controlled and outputs are "
                            "non-deterministic",
                            canonical,
                        )
                    attempt -= 1  # adaptation, not a retry
                    continue
                mapped = _map_exception(exc)
                # Auth errors are deterministic; don't waste retries on them.
                if isinstance(mapped, AuthError):
                    raise mapped from exc
                if attempt >= self._retry.max_attempts:
                    raise mapped from exc
                delay = self._retry.delay(attempt)
                log.warning(
                    "model %s%s attempt %d failed (%s); retrying in %.2fs",
                    canonical,
                    log_suffix,
                    attempt,
                    mapped.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            return response, latency_ms


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
    finish_reason = _extract_finish_reason(response)
    if finish_reason == "length":
        log.warning(
            "model %s output truncated at the token cap (finish_reason=length); "
            "raise defaults.max_tokens or the prompt's max_tokens",
            canonical,
        )
    if text == "" and output_tokens > 0 and finish_reason == "stop":
        log.warning(
            "model %s returned no text despite %d output tokens — likely a "
            "reasoning/thinking-only response; consider raising max_tokens or "
            "disabling thinking",
            canonical,
            output_tokens,
        )
    return CompletionResult(
        text=text,
        model_id=canonical,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
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
    finish_reason = _extract_finish_reason(raw)
    if finish_reason == "length":
        log.warning(
            "model %s tool output truncated at the token cap (finish_reason=length); "
            "raise defaults.max_tokens or the prompt's max_tokens",
            canonical,
        )
    return ToolCompletionResult(
        trace=trace,
        model_id=canonical,
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        cost_usd=_safe_cost(response),
        latency_ms=latency_ms,
        raw_provider_response=raw,
        finish_reason=finish_reason,
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


def _extract_finish_reason(response: Any) -> str | None:
    """Pull ``choices[0].finish_reason`` from a LiteLLM response.

    LiteLLM normalises provider stop reasons (Anthropic ``stop_reason``,
    OpenAI/Gemini ``finish_reason``) onto ``choices[0].finish_reason``;
    ``"length"`` signals the output was truncated at the token cap. Handles
    both SDK-object and dict-shaped responses (the tool path hands us a dict
    view). This is best-effort metadata — return ``None`` rather than
    raising if the field is absent, since a missing stop reason must not
    fail the call.
    """
    try:
        choices = response["choices"] if isinstance(response, dict) else response.choices
        choice = choices[0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if isinstance(choice, dict):
        return cast("str | None", choice.get("finish_reason"))
    return cast("str | None", getattr(choice, "finish_reason", None))


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


def _is_temperature_value_rejection(exc: BaseException) -> bool:
    """Report whether ``exc`` is a provider 400 rejecting ``temperature``'s value.

    Matched by type NAME like :func:`_map_exception`, so litellm's
    ``BadRequestError`` is caught without importing its class. The caller
    must additionally check that the outgoing kwargs actually carried
    ``temperature`` — a temperature-flavoured 400 on a call that never sent
    the parameter is somebody else's bug and must surface.
    """
    return type(exc).__name__ == "BadRequestError" and "temperature" in str(exc).lower()


def _map_exception(exc: BaseException) -> ModelClientError:
    """Map a LiteLLM/provider exception to EvalShift's error hierarchy."""
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
