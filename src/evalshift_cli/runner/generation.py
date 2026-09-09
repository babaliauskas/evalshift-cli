"""Translate a recorded generation_config into ModelClient dispatch kwargs.

The dict arrives verbatim from a capture (recorded by the evalshift-sdk under the
model_call event's ``metadata["generation_config"]`` and promoted onto
``SuiteExample.generation_config``). Only well-known keys are honoured; everything
else is ignored with a warning — a bad or foreign recorded config must never
block a suite run, but it must not vanish silently either (that silence is
external-review finding #8).

Tool-choice constraints arrive in whichever provider's spelling production used
(``tool_choice`` as an OpenAI string/object, ``tool_choice`` as an Anthropic
object, or Gemini's ``tool_config``). All three normalise here to the single
OpenAI-style form, because LiteLLM's provider configs already translate that
form into Anthropic's ``tool_choice`` object and Gemini's ``toolConfig`` — see
``TestLiteLLMTranslatesToolChoice`` in ``tests/unit/test_model_client.py``,
which pins that behaviour.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Keys this translator consumes; anything else is warned about and skipped.
_HANDLED_KEYS = frozenset(
    {
        "temperature",
        "response_format",
        "response_mime_type",
        "response_schema",
        "tool_choice",
        "parallel_tool_calls",
        "tool_config",
    }
)

#: Anthropic ``tool_choice.type`` → the OpenAI string that means the same thing.
#: ``"tool"`` is absent deliberately: it needs the accompanying ``name`` and so
#: becomes an object, not a string.
_ANTHROPIC_MODES: dict[str, str] = {"auto": "auto", "any": "required", "none": "none"}

#: Gemini ``function_calling_config.mode`` → the same OpenAI string.
_GEMINI_MODES: dict[str, str] = {"AUTO": "auto", "ANY": "required", "NONE": "none"}

#: Ignored-key sets already warned about, so a 500-example suite whose captures
#: all record the same unhandled key logs once rather than 500 times. Keyed by
#: the sorted key tuple, so a genuinely different config still gets its warning.
_WARNED_IGNORED_KEYS: set[tuple[str, ...]] = set()


def _normalise_tool_choice(
    config: dict[str, Any],
) -> tuple[str | dict[str, Any] | None, bool | None]:
    """``(openai_style_tool_choice, parallel_tool_calls)`` from a recorded config.

    Accepts all three recorded spellings and emits the OpenAI-style one:

    * OpenAI — ``"auto" | "none" | "required"`` or
      ``{"type": "function", "function": {"name": ...}}``: passed through.
    * Anthropic — ``{"type": "auto"|"any"|"tool"|"none", "name"?,
      "disable_parallel_tool_use"?}``: ``any`` → ``"required"``, ``tool`` →
      the named-function object, and ``disable_parallel_tool_use`` is inverted
      into ``parallel_tool_calls``.
    * Gemini — ``tool_config: {"function_calling_config": {"mode": ...,
      "allowed_function_names"?: [...]}}``. Exactly one allowed name becomes a
      named-function choice; several degrade to ``"required"`` (OpenAI-style
      has no allow-list). Gemini has no parallel-tool flag, so none is derived.

    ``tool_choice`` wins when both spellings are present — they assert the same
    thing about the same call, and a capture only ever records one provider.
    Unrecognised shapes are warned about and skipped, never raised.
    """
    choice: str | dict[str, Any] | None = None
    parallel: bool | None = None

    raw = config.get("tool_choice")
    if raw is not None:
        if isinstance(raw, str) and raw in {"auto", "none", "required"}:
            choice = raw
        elif isinstance(raw, dict):
            kind = raw.get("type")
            name = raw.get("name")
            if kind == "function" and isinstance(raw.get("function"), dict):
                fn_name = raw["function"].get("name")
                if isinstance(fn_name, str) and fn_name:
                    choice = {"type": "function", "function": {"name": fn_name}}
            elif kind == "tool" and isinstance(name, str) and name:
                choice = {"type": "function", "function": {"name": name}}
            elif isinstance(kind, str) and kind in _ANTHROPIC_MODES:
                choice = _ANTHROPIC_MODES[kind]
            disable_parallel = raw.get("disable_parallel_tool_use")
            if isinstance(disable_parallel, bool):
                parallel = not disable_parallel
        if choice is None:
            log.warning("generation_config tool_choice not understood, ignoring it: %r", raw)

    raw_config = config.get("tool_config")
    if choice is None and raw_config is not None:
        fcc = None
        if isinstance(raw_config, dict):
            # Snake case is the google-genai Python spelling; camelCase is the
            # REST one. A capture can carry either.
            fcc = raw_config.get("function_calling_config", raw_config.get("functionCallingConfig"))
        if isinstance(fcc, dict):
            mode = fcc.get("mode")
            allowed = fcc.get("allowed_function_names", fcc.get("allowedFunctionNames"))
            if isinstance(mode, str) and mode.upper() in _GEMINI_MODES:
                choice = _GEMINI_MODES[mode.upper()]
                if (
                    choice == "required"
                    and isinstance(allowed, list)
                    and len(allowed) == 1
                    and isinstance(allowed[0], str)
                    and allowed[0]
                ):
                    choice = {"type": "function", "function": {"name": allowed[0]}}
        if choice is None:
            log.warning("generation_config tool_config not understood, ignoring it: %r", raw_config)

    if "parallel_tool_calls" in config:
        raw_parallel = config["parallel_tool_calls"]
        if isinstance(raw_parallel, bool):
            # An explicit top-level flag beats the one inferred from an
            # Anthropic tool_choice object.
            parallel = raw_parallel
        else:
            log.warning(
                "generation_config parallel_tool_calls is not a bool, ignoring it: %r",
                raw_parallel,
            )

    return choice, parallel


def translate_generation_config(
    config: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any] | None]:
    """``(temperature_override, extra_kwargs)`` for ModelClient from a recorded config.

    * ``temperature`` (int/float, not bool) → temperature override.
    * a litellm-shaped ``response_format`` (dict) → passed through as-is.
    * else ``response_mime_type == "application/json"`` → ``response_format``:
      ``json_schema`` when a dict ``response_schema`` is present, else ``json_object``.
    * ``tool_choice`` / ``tool_config`` / ``parallel_tool_calls`` → OpenAI-style
      ``tool_choice`` and ``parallel_tool_calls`` (see
      :func:`_normalise_tool_choice`).
    * anything else is ignored (warned) — never an error.
    """
    if not isinstance(config, dict) or not config:
        return None, None

    temperature: float | None = None
    raw_temp = config.get("temperature")
    if isinstance(raw_temp, (int, float)) and not isinstance(raw_temp, bool):
        temperature = float(raw_temp)

    extra: dict[str, Any] = {}
    response_format = config.get("response_format")
    if isinstance(response_format, dict):
        extra["response_format"] = response_format
    elif config.get("response_mime_type") == "application/json":
        schema = config.get("response_schema")
        if isinstance(schema, dict):
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "captured_schema", "schema": schema},
            }
        else:
            extra["response_format"] = {"type": "json_object"}

    tool_choice, parallel_tool_calls = _normalise_tool_choice(config)
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = parallel_tool_calls

    ignored = tuple(sorted(k for k in config if k not in _HANDLED_KEYS))
    if ignored and ignored not in _WARNED_IGNORED_KEYS:
        _WARNED_IGNORED_KEYS.add(ignored)
        log.warning(
            "generation_config keys ignored at dispatch (the target will not be asked "
            "to honour them): %s",
            list(ignored),
        )
    return temperature, extra or None


__all__ = ["translate_generation_config"]
