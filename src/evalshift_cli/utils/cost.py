"""Pre-flight cost estimation.

Before kicking off a paid run, the orchestrator asks: *roughly, how much
is this going to cost?* The estimate doesn't need to be perfect — it
just needs to keep users from accidentally spending $50 on a misconfigured
run when they meant to spend $0.50.

Approach: render the prompt against the first ``sample_size`` examples
(default 5), measure their token counts via :func:`litellm.token_counter`,
average the prompt size, assume a fixed completion cap, then multiply by
the run shape ``N prompts x M examples x {source, target}``.

The function deliberately accepts string prompts and a list of input
mappings rather than ``PromptTemplate`` / ``Suite`` objects so it stays
free of cyclic imports and is trivially testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import litellm

from evalshift_cli.models.registry import resolve_model

DEFAULT_SAMPLE_SIZE: int = 5


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Result of :func:`estimate_run_cost`.

    Attributes:
        total_calls: Number of LLM calls implied by the run shape.
        avg_prompt_tokens: Average token count across the sampled
            rendered prompts.
        assumed_completion_tokens: Completion length used in the math
            (typically the model's registered default).
        estimated_usd: Estimated total cost in USD. ``0.0`` if LiteLLM
            doesn't have pricing data for the model.
    """

    total_calls: int
    avg_prompt_tokens: int
    assumed_completion_tokens: int
    estimated_usd: float


def _render_safely(template: str, inputs: Mapping[str, Any]) -> str:
    """Best-effort render that falls back to the raw template on missing vars.

    The estimator runs *before* :func:`validate_suite_against_prompts`,
    so we have to tolerate examples that don't fully satisfy the
    template — return whatever we can rather than raising mid-estimate.
    """
    try:
        return template.format_map(_DefaultDict(inputs))
    except (KeyError, IndexError):
        return template


class _DefaultDict(dict[str, Any]):
    """Mapping that returns ``""`` for missing keys (template-friendly)."""

    def __init__(self, base: Mapping[str, Any]) -> None:
        super().__init__(base)

    def __missing__(self, _key: str) -> str:
        return ""


def _avg_prompt_tokens(
    template: str,
    examples: Sequence[Mapping[str, Any]],
    canonical_model: str,
    sample_size: int,
    per_example_extra_chars: Sequence[int] | None = None,
) -> int:
    """Average token count for the first ``sample_size`` rendered prompts.

    ``per_example_extra_chars``, when given, is a same-length,
    index-aligned sequence of extra character counts (e.g. a multi-turn
    example's recorded history prefix) added to each sampled example's
    rendered text before counting tokens — so a large history prefix
    inflates the estimate instead of being silently ignored.
    """
    sample = list(examples)[:sample_size]
    extras = list(per_example_extra_chars)[:sample_size] if per_example_extra_chars else []
    if not sample:
        return _safe_token_count(canonical_model, template)
    counts = []
    for i, ex in enumerate(sample):
        rendered = _render_safely(template, ex)
        extra_chars = extras[i] if i < len(extras) else 0
        if extra_chars:
            rendered += "x" * extra_chars
        counts.append(_safe_token_count(canonical_model, rendered))
    return max(1, sum(counts) // len(counts))


def _safe_token_count(model: str, text: str) -> int:
    """``litellm.token_counter`` with a defensive fallback.

    Some Gemini/OpenAI counters demand network access or vendor SDKs
    that may not be wired up; if anything goes wrong, fall back to the
    classic 4-chars-per-token heuristic so the estimator still returns
    a number.
    """
    try:
        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


def _safe_cost_per_call(
    canonical_model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """``litellm.cost_per_token`` with a defensive fallback to $0.

    Same philosophy as :func:`_safe_token_count`: never let a missing
    pricing entry block the estimator.
    """
    try:
        in_cost, out_cost = litellm.cost_per_token(
            model=canonical_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(in_cost) + float(out_cost)
    except Exception:
        return 0.0


def _price_table_key(model_id: str) -> str | None:
    """The key litellm's price table holds ``model_id`` under, or ``None``.

    Tries the id as recorded, the registry's canonical (provider-prefixed)
    form, and that form with the prefix stripped — litellm keys most
    first-party entries bare (``gpt-4o-mini``) and some prefixed
    (``gemini/gemini-2.5-flash``). A pure dict lookup: ``litellm.model_cost``
    is the bundled table, so a miss costs nothing and touches nothing.
    """
    canonical = resolve_model(model_id).id
    _, _, stripped = canonical.partition("/")
    table = litellm.model_cost
    for candidate in (model_id, canonical, stripped):
        if candidate and candidate in table:
            return candidate
    return None


def estimate_call_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Price one recorded model call from litellm's price table.

    Used at promotion for a capture whose ``model_call`` recorded tokens but
    no cost (the SDK never prices anything). Returns ``0.0`` — silently, at
    no log level — for a model the table does not price: a local or
    self-hosted model is the normal case, not a failure. The table lookup
    is done *before* calling litellm's pricer, deliberately: for an unknown
    id ``litellm.cost_per_token`` prints a provider banner, and for an
    ``ollama/`` id it opens a socket to the local daemon to ask for model
    info. Neither belongs in ``capture sync``.

    Args:
        model_id: The id the capture recorded — an alias, a bare vendor id,
            or a provider-prefixed one; resolved through the registry first.
        input_tokens: Prompt tokens the call recorded.
        output_tokens: Completion tokens the call recorded.

    Returns:
        Dollar cost, or ``0.0`` when no tokens were recorded, the model is
        unpriced, or the pricer fails.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return 0.0
    key = _price_table_key(model_id)
    if key is None:
        return 0.0
    return _safe_cost_per_call(
        canonical_model=key,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )


def estimate_run_cost(
    *,
    template: str,
    examples: Sequence[Mapping[str, Any]],
    n_prompts: int,
    models: Sequence[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    completion_tokens: int | None = None,
    per_example_extra_chars: Sequence[int] | None = None,
) -> CostEstimate:
    """Estimate the cost of running a (prompts x examples x models) sweep.

    Args:
        template: A single representative prompt template. Real runs
            often have multiple prompts; for a quick estimate we use one
            and scale by ``n_prompts``.
        examples: The suite examples used to render samples for token
            counting.
        n_prompts: How many distinct prompts the run will execute.
        models: List of canonical-or-alias model ids (typically two:
            source and target).
        sample_size: How many examples to render-and-count when
            estimating the average prompt size.
        completion_tokens: Override the assumed completion length. When
            ``None``, uses the *first* model's registered default.
        per_example_extra_chars: Optional, index-aligned with
            ``examples`` — extra character count added to each sampled
            example before token counting (e.g. a multi-turn example's
            recorded history prefix, sent on every call and otherwise
            invisible to the estimator). ``None`` or all-zeros reproduces
            the estimate exactly as if this parameter didn't exist.

    Returns:
        A :class:`CostEstimate` summarising the math.
    """
    if not models:
        return CostEstimate(
            total_calls=0,
            avg_prompt_tokens=0,
            assumed_completion_tokens=0,
            estimated_usd=0.0,
        )
    n_examples = len(examples)
    total_calls = n_prompts * n_examples * len(models)

    primary_meta = resolve_model(models[0])
    assumed_completion = (
        completion_tokens if completion_tokens is not None else primary_meta.default_max_tokens
    )

    avg_prompt = _avg_prompt_tokens(
        template=template,
        examples=examples,
        canonical_model=primary_meta.id,
        sample_size=sample_size,
        per_example_extra_chars=per_example_extra_chars,
    )

    per_call_cost_per_model = [
        _safe_cost_per_call(
            canonical_model=resolve_model(m).id,
            prompt_tokens=avg_prompt,
            completion_tokens=assumed_completion,
        )
        for m in models
    ]
    estimated = sum(c * n_prompts * n_examples for c in per_call_cost_per_model)

    return CostEstimate(
        total_calls=total_calls,
        avg_prompt_tokens=avg_prompt,
        assumed_completion_tokens=assumed_completion,
        estimated_usd=estimated,
    )


__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "CostEstimate",
    "estimate_call_cost",
    "estimate_run_cost",
]
