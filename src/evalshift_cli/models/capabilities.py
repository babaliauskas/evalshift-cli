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

LiteLLM's answer is the authority, with one enumerated exception:
:func:`silently_unsent_params` lists the handful of parameters LiteLLM
*claims* to support and then discards inside a provider transformer, which the
probe cannot see by construction. See :data:`_KNOWN_LITELLM_GAPS`.

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
from typing import Final

import litellm

from evalshift_cli.evaluators.tool_parser import ToolParseError, detect_provider
from evalshift_cli.models.registry import resolve_model

log = logging.getLogger(__name__)

#: Pseudo-parameter name for a tool spec's ``strict`` flag.
#:
#: Not a generation parameter at all — it is a per-tool field on the ``tools``
#: array — so it has no entry in LiteLLM's OpenAI-param vocabulary and no probe
#: could ever answer for it. It is still a constraint the capture recorded and
#: the replay can lose, so it travels with the others under a dotted name that
#: says where it lives, keeping it distinguishable from a real parameter in
#: ``state.json``, the report banner, and a policy failure reason.
TOOL_STRICT_PARAM: Final = "tools.strict"

#: Provider → parameters LiteLLM accepts and reports as supported, but never
#: puts in the provider request body.
#:
#: Verified against **litellm 1.100.0** by reading its own source; both entries
#: are Gemini transformer behaviour:
#:
#: * ``parallel_tool_calls`` — ``litellm/llms/vertex_ai/gemini/transformation.py``
#:   filters the optional params against ``GenerationConfig.__annotations__``,
#:   which has no parallel-tool field, so the value is discarded before the
#:   body is built. ``get_supported_openai_params`` lists it regardless.
#: * :data:`TOOL_STRICT_PARAM` — a Gemini ``function_declaration`` has no
#:   ``strict`` member, and ``_map_function`` in the same package drops the key
#:   while building the declarations.
#:
#: Hard-coded, and deliberately tiny: this is the exception list to an
#: otherwise dynamic probe, so it has to be obvious what to delete. Remove an
#: entry the moment LiteLLM starts sending that parameter (or stops claiming
#: it, at which point :func:`unsupported_params` finds it unaided).
_KNOWN_LITELLM_GAPS: Final[dict[str, frozenset[str]]] = {
    "gemini": frozenset({"parallel_tool_calls", TOOL_STRICT_PARAM}),
}


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


def silently_unsent_params(model_id: str, params: Iterable[str]) -> list[str]:
    """Return the subset of ``params`` LiteLLM claims but never sends for ``model_id``.

    The companion to :func:`unsupported_params`, for the cases that probe is
    structurally blind to: LiteLLM answers "supported" and then drops the value
    while building the provider's request body. Deliberately a lookup in the
    hard-coded :data:`_KNOWN_LITELLM_GAPS` table rather than a probe — LiteLLM
    is not asked at all, because its answer here is the bug.

    Args:
        model_id: Any user-supplied model id or alias. Resolved to its
            canonical form, then to a provider, so aliases and unregistered
            passthrough preview ids both land on the right table entry.
        params: Names to check, in the same OpenAI-style vocabulary as
            :func:`unsupported_params` plus the :data:`TOOL_STRICT_PARAM`
            pseudo-parameter. Duplicates are collapsed.

    Returns:
        The requested names this model's provider is known to discard, sorted.
        Empty when the provider has no known gaps, when nothing was asked, and
        when the id cannot be placed with a provider at all — the same
        "uncertainty reads as honoured" asymmetry the probes use.
    """
    requested = sorted(set(params))
    if not requested:
        return []
    canonical = resolve_model(model_id).id
    try:
        provider = detect_provider(canonical)
    except ToolParseError:
        # Not one of the three providers we ship parsers for, so certainly not
        # one of the two we have verified a LiteLLM gap for.
        return []
    gaps = _KNOWN_LITELLM_GAPS.get(provider, frozenset())
    return [name for name in requested if name in gaps]


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


__all__ = [
    "TOOL_STRICT_PARAM",
    "honors_temperature",
    "silently_unsent_params",
    "unsupported_params",
]
