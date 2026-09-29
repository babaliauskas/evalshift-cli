"""DeepSeek API behaviour EvalShift has to account for, in one place.

DeepSeek's current API models (``deepseek-flash``, ``deepseek-v4-pro``) run in
*thinking mode* unless the request turns it off
(https://api-docs.deepseek.com/guides/thinking_mode/). Thinking mode changes
two things the replay path depends on:

* ``temperature`` is accepted and ignored — "setting them will not trigger an
  error but will also have no effect". LiteLLM still lists ``temperature`` as
  supported, so :func:`~evalshift_cli.models.capabilities.honors_temperature`
  asks :func:`thinking_by_default` before trusting LiteLLM's answer, and the
  report's non-determinism banner covers DeepSeek arms.
* A request carrying ``tools`` must pass back the ``reasoning_content`` of
  every earlier assistant turn, or the API answers 400. A teacher-forced round
  is rebuilt from the *recording* — often another model's — so there is no
  DeepSeek reasoning to pass. :func:`backfill_reasoning_content` supplies the
  single-space placeholder the API accepts. LiteLLM does the same, but only
  when the caller passes ``thinking`` explicitly, which EvalShift never does
  (a capture cannot record it); doing it here also keeps the fix independent
  of the installed LiteLLM version.

EvalShift never switches thinking off: the application under test runs with
DeepSeek's default, and replaying a different configuration would measure
the wrong thing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import litellm

from evalshift_cli.models.registry import resolve_model

#: The minimum ``reasoning_content`` DeepSeek accepts on an assistant turn —
#: an empty reasoning chain, the same value LiteLLM injects.
REASONING_PLACEHOLDER: Final = " "


def thinking_by_default(model_id: str) -> bool:
    """Report whether ``model_id`` is a DeepSeek API model that thinks by default.

    Args:
        model_id: Any user-supplied model id or alias; resolved first, so a
            bare id recorded by a capture works.

    Returns:
        ``True`` only for the ``deepseek`` provider when LiteLLM positively
        reports reasoning support for the canonical id. Every other provider,
        DeepSeek weights on another host, and every uncertain LiteLLM answer
        return ``False`` — the same "uncertainty reads as honoured" rule as
        :mod:`evalshift_cli.models.capabilities`.
    """
    meta = resolve_model(model_id)
    if meta.provider != "deepseek":
        return False
    try:
        return bool(litellm.supports_reasoning(model=meta.id))
    except Exception:
        return False


def backfill_reasoning_content(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with :data:`REASONING_PLACEHOLDER` on bare assistant turns.

    Assistant turns that already carry a non-empty ``reasoning_content`` keep
    it; every other role passes through. The input is not mutated.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
            out.append({**msg, "reasoning_content": REASONING_PLACEHOLDER})
        else:
            out.append(dict(msg))
    return out


__all__ = ["REASONING_PLACEHOLDER", "backfill_reasoning_content", "thinking_by_default"]
