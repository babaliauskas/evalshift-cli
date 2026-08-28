"""Runtime capability probes for models EvalShift is about to call.

Separate from :mod:`evalshift.models.registry` on purpose. The registry is
static data we ship; this module asks **LiteLLM** a question at call time.

The question that matters today is whether a model still honours
``temperature``. EvalShift sends ``temperature=0.0`` on every call and sets
``drop_params=True``, so when a provider withdraws the parameter LiteLLM drops
it silently rather than erroring: sampling reverts to the provider default and
the paired comparison quietly loses its control variable. Google has announced
exactly this for Gemini 3+.

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
``ModelClient._dispatch_with_retry`` in :mod:`evalshift.models.client`.
"""

from __future__ import annotations

import logging

import litellm

from evalshift.models.registry import resolve_model

log = logging.getLogger(__name__)


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
    """
    canonical = resolve_model(model_id).id
    try:
        supported = litellm.get_supported_openai_params(model=canonical)
    except Exception:
        log.debug("could not read supported params for %s; assuming temperature", canonical)
        return True
    if not supported:
        return True
    return "temperature" in supported


__all__ = ["honors_temperature"]
