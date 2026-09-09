"""Runtime capability probes for models EvalShift is about to call.

Separate from :mod:`evalshift_cli.models.registry` on purpose. The registry is
static data we ship; this module asks **LiteLLM** a question at call time.

The question that matters today is whether a model still honours
``temperature``. EvalShift sends ``temperature=0.0`` on every call and sets
``drop_params=True``, so when a provider withdraws the parameter LiteLLM drops
it silently rather than erroring: sampling reverts to the provider default and
the paired comparison quietly loses its control variable. Google has announced
exactly this for Gemini 3+.

The same mechanism swallows every *other* parameter a replay carries.
A capture that pinned ``response_format`` or ``tool_choice`` replays against a
target that never accepted them as a call that succeeds with the constraint
missing — the arm then measures the model change *plus* an absent constraint.
:func:`unsupported_params` generalises the probe to any parameter, and the
orchestrator records the answer per run (``RunState.dropped_params``) so the
report can say so instead of the difference showing up as an unexplained
regression.

Asking LiteLLM rather than consulting a flag in the registry is deliberate. The
Gemini 3 preview ids that prompted this are *passthrough* ids — absent from
``_MODELS`` and resolved by prefix inference — so a static flag would miss the
very models it was added for, and would keep missing each new preview id until
an EvalShift release caught up.

Note this detects *withdrawal*, not value constraints. Reasoning-tier models
such as ``gpt-5.6-terra`` advertise ``temperature`` while rejecting every
value except their default; ``drop_params`` does not cover them (LiteLLM
special-cases only o-series names). That case is detected from the
provider's own 400 at dispatch time and adapted per model — see
``ModelClient._dispatch_with_retry`` in :mod:`evalshift_cli.models.client`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import litellm

from evalshift_cli.models.registry import resolve_model

log = logging.getLogger(__name__)


def unsupported_params(model_id: str, params: Iterable[str]) -> list[str]:
    """Return the subset of ``params`` LiteLLM says ``model_id`` does not accept.

    Args:
        model_id: Any user-supplied model id or alias. Resolved to its
            canonical LiteLLM form before the lookup, so bare aliases and
            unregistered passthrough ids both work.
        params: OpenAI-style parameter names to ask about — the vocabulary
            ``litellm.get_supported_openai_params`` answers in, so a
            provider-specific spelling (Gemini's ``response_mime_type``,
            ``tool_config``) must be mapped to its OpenAI name by the caller.
            Duplicates are collapsed.

    Returns:
        The requested names LiteLLM positively excludes, sorted. Empty on
        every uncertain outcome — a raised exception, ``None``, an empty list,
        or an empty ``params`` — and empty when everything is supported.

        The asymmetry mirrors :func:`honors_temperature` and is the point.
        A wrong non-empty answer puts a "constraints not honoured" banner on
        reports whose constraints were honoured fine, the moment LiteLLM
        changes a signature; a wrong empty answer costs one missed warning.
    """
    requested = sorted(set(params))
    if not requested:
        # No probe at all: nothing was asked for, so nothing can be dropped,
        # and a lookup here would only be a chance to raise.
        return []
    canonical = resolve_model(model_id).id
    try:
        supported = litellm.get_supported_openai_params(model=canonical)
    except Exception:
        log.debug("could not read supported params for %s; assuming all honoured", canonical)
        return []
    if not supported:
        return []
    known = set(supported)
    return [name for name in requested if name not in known]


def honors_temperature(model_id: str) -> bool:
    """Report whether ``model_id`` still accepts a ``temperature`` parameter.

    Args:
        model_id: Any user-supplied model id or alias. Resolved to its
            canonical LiteLLM form before the lookup, so bare aliases and
            unregistered passthrough ids both work.

    Returns:
        ``False`` only when LiteLLM positively reports a parameter set that
        excludes ``temperature``. Every uncertain outcome — a raised
        exception, ``None``, or an empty list — returns ``True``.

        The asymmetry is the point. A wrong ``False`` puts a
        "non-deterministic" banner on every report the moment LiteLLM changes
        a signature; a wrong ``True`` costs one missed warning. We take the
        second risk.

        Expressed in terms of :func:`unsupported_params` so the two probes
        cannot drift apart: both must treat an uncertain answer as "honoured".
    """
    return not unsupported_params(model_id, ["temperature"])


__all__ = ["honors_temperature", "unsupported_params"]
