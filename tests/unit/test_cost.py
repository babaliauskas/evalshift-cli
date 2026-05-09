"""Tests for :mod:`evalshift.utils.cost`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from evalshift.utils import cost as cost_module
from evalshift.utils.cost import CostEstimate, estimate_run_cost


@pytest.fixture(autouse=True)
def _stub_litellm(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace LiteLLM's token + cost helpers with deterministic stubs."""
    # 1 token per 4 characters, like the real heuristic.
    monkeypatch.setattr(
        cost_module.litellm,
        "token_counter",
        lambda model, text: max(1, len(text) // 4),
    )
    # $0.001 per prompt token, $0.002 per completion token.
    monkeypatch.setattr(
        cost_module.litellm,
        "cost_per_token",
        lambda model, prompt_tokens, completion_tokens: (
            prompt_tokens * 0.001,
            completion_tokens * 0.002,
        ),
    )
    yield


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestEstimateHappy:
    def test_returns_cost_estimate(self) -> None:
        estimate = estimate_run_cost(
            template="Hello {name}",
            examples=[{"name": "Alex"}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert isinstance(estimate, CostEstimate)
        assert estimate.total_calls == 1

    def test_total_calls_math(self) -> None:
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}] * 50,
            n_prompts=2,
            models=["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
        )
        # 2 prompts x 50 examples x 2 models = 200
        assert estimate.total_calls == 200

    def test_zero_examples_yields_zero_calls(self) -> None:
        estimate = estimate_run_cost(
            template="Hi",
            examples=[],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert estimate.total_calls == 0
        # Prompt-token average should still be derivable from the bare template.
        assert estimate.avg_prompt_tokens >= 1

    def test_zero_models_short_circuits(self) -> None:
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}],
            n_prompts=1,
            models=[],
        )
        assert estimate.total_calls == 0
        assert estimate.estimated_usd == 0.0

    def test_estimated_usd_scales_with_run_shape(self) -> None:
        small = estimate_run_cost(
            template="Hi",
            examples=[{}] * 10,
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        big = estimate_run_cost(
            template="Hi",
            examples=[{}] * 100,
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        # 10x more examples → ~10x more cost.
        assert big.estimated_usd == pytest.approx(small.estimated_usd * 10)

    def test_assumed_completion_tokens_uses_primary_default(self) -> None:
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        # Registry default for every model is 1024.
        assert estimate.assumed_completion_tokens == 1024

    def test_explicit_completion_tokens_override(self) -> None:
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
            completion_tokens=64,
        )
        assert estimate.assumed_completion_tokens == 64

    def test_alias_resolved_to_canonical(self) -> None:
        # Should not raise UnknownModelError for an alias.
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}],
            n_prompts=1,
            models=["gemini-2.5-flash"],
        )
        assert estimate.total_calls == 1


# ---------------------------------------------------------------------------
# Render fallback
# ---------------------------------------------------------------------------


class TestRenderFallback:
    def test_missing_template_var_does_not_explode(self) -> None:
        # Estimator runs *before* the validator — it has to tolerate
        # incomplete examples.
        estimate = estimate_run_cost(
            template="Hello {name}",
            examples=[{}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert estimate.total_calls == 1

    def test_template_variables_get_substituted_in_token_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Make token_counter echo the *length* of the rendered text in chars
        # so we can assert substitution actually happened.
        seen: list[str] = []

        def spy_counter(model: str, text: str) -> int:
            seen.append(text)
            return max(1, len(text))

        monkeypatch.setattr(cost_module.litellm, "token_counter", spy_counter)
        estimate_run_cost(
            template="Hello {name}",
            examples=[{"name": "Alex"}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert any("Hello Alex" in s for s in seen)


# ---------------------------------------------------------------------------
# Defensive fallbacks
# ---------------------------------------------------------------------------


class TestDefensiveFallbacks:
    def test_token_counter_failure_uses_4_chars_per_token_heuristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(**_kwargs: Any) -> int:
            raise RuntimeError("network down")

        monkeypatch.setattr(cost_module.litellm, "token_counter", boom)
        estimate = estimate_run_cost(
            template="x" * 40,  # 40 chars → 10 tokens by heuristic
            examples=[{}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert estimate.avg_prompt_tokens == 10

    def test_cost_per_token_failure_treated_as_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(**_kwargs: Any) -> Any:
            raise RuntimeError("no pricing data")

        monkeypatch.setattr(cost_module.litellm, "cost_per_token", boom)
        estimate = estimate_run_cost(
            template="Hi",
            examples=[{}],
            n_prompts=1,
            models=["gemini/gemini-2.5-flash"],
        )
        assert estimate.estimated_usd == 0.0
