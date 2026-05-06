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

from aimigrate.models.registry import get_model

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
    except KeyError, IndexError:
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
) -> int:
    """Average token count for the first ``sample_size`` rendered prompts."""
    sample = list(examples)[:sample_size]
    if not sample:
        return _safe_token_count(canonical_model, template)
    counts = [_safe_token_count(canonical_model, _render_safely(template, ex)) for ex in sample]
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


def estimate_run_cost(
    *,
    template: str,
    examples: Sequence[Mapping[str, Any]],
    n_prompts: int,
    models: Sequence[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    completion_tokens: int | None = None,
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

    primary_meta = get_model(models[0])
    assumed_completion = (
        completion_tokens if completion_tokens is not None else primary_meta.default_max_tokens
    )

    avg_prompt = _avg_prompt_tokens(
        template=template,
        examples=examples,
        canonical_model=primary_meta.id,
        sample_size=sample_size,
    )

    per_call_cost_per_model = [
        _safe_cost_per_call(
            canonical_model=get_model(m).id,
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
    "estimate_run_cost",
]
