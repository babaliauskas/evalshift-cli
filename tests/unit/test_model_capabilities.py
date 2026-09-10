"""Tests for :mod:`evalshift_cli.models.capabilities`.

The registry is static data; this module asks LiteLLM at call time whether a
model still honours ``temperature``. LiteLLM is the authority because the ids
that matter here (fresh Gemini previews) are passthrough ids the registry has
never heard of.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from evalshift_cli.models.capabilities import (
    TOOL_STRICT_PARAM,
    honors_temperature,
    silently_unsent_params,
    unsupported_params,
)

# A realistic slice of what LiteLLM returns for a chat model.
_WITH_TEMPERATURE = ["max_tokens", "temperature", "top_p", "tools"]
_WITHOUT_TEMPERATURE = ["max_tokens", "top_p", "tools"]


def _stub_params(
    monkeypatch: pytest.MonkeyPatch,
    result: list[str] | Exception | None,
) -> list[str]:
    """Point LiteLLM's capability lookup at ``result``; record the ids asked for."""
    seen: list[str] = []

    def fake(*, model: str, **_: Any) -> list[str] | None:
        seen.append(model)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(litellm, "get_supported_openai_params", fake)
    return seen


class TestHonorsTemperature:
    def test_false_when_litellm_omits_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The day Google removes the parameter, LiteLLM drops it from the map."""
        _stub_params(monkeypatch, _WITHOUT_TEMPERATURE)
        assert honors_temperature("gemini/gemini-3.5-flash-lite") is False

    def test_true_when_litellm_lists_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert honors_temperature("gemini/gemini-2.5-flash") is True


class TestDefensiveFallback:
    """Unknown answers must read as "honoured".

    A false non-determinism banner on every run because LiteLLM changed a
    signature is worse than one missed warning, so every uncertain path
    returns ``True``.
    """

    def test_true_when_litellm_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, RuntimeError("litellm exploded"))
        assert honors_temperature("gemini/gemini-3.5-flash-lite") is True

    def test_true_when_litellm_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, None)
        assert honors_temperature("some-model-we-cannot-place") is True

    def test_true_when_litellm_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty list means LiteLLM knows nothing, not that nothing is supported."""
        _stub_params(monkeypatch, [])
        assert honors_temperature("some-model-we-cannot-place") is True


class TestIdResolution:
    def test_asks_litellm_about_the_canonical_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare alias must be resolved before LiteLLM sees it."""
        seen = _stub_params(monkeypatch, _WITH_TEMPERATURE)
        honors_temperature("gemini-2.5-flash")
        assert seen == ["gemini/gemini-2.5-flash"]

    def test_passthrough_ids_get_a_provider_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gemini 3 ids are absent from the registry; they still need prefixing."""
        seen = _stub_params(monkeypatch, _WITHOUT_TEMPERATURE)
        assert honors_temperature("gemini-3.5-flash-lite") is False
        assert seen == ["gemini/gemini-3.5-flash-lite"]


class TestUnsupportedParams:
    """The generalised probe behind ``dropped_params``.

    Same authority and the same asymmetry as :func:`honors_temperature`: only a
    positive answer from LiteLLM that excludes a parameter counts as "dropped".
    """

    def test_returns_the_params_litellm_omits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, ["temperature", "max_tokens"])
        assert unsupported_params(
            "gemini/gemini-3.5-flash-lite",
            ["tool_choice", "response_format", "max_tokens"],
        ) == ["response_format", "tool_choice"]

    def test_empty_when_every_param_is_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, ["temperature", "tool_choice", "response_format"])
        assert unsupported_params("openai/gpt-4o", ["tool_choice", "response_format"]) == []

    def test_result_is_sorted_and_deduplicated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two recorded keys can map to one litellm param; report it once."""
        _stub_params(monkeypatch, ["temperature"])
        assert unsupported_params(
            "openai/gpt-4o", ["tool_choice", "response_format", "tool_choice"]
        ) == ["response_format", "tool_choice"]

    def test_empty_when_nothing_was_asked_about(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No probe at all when the suite recorded no constraints."""
        seen = _stub_params(monkeypatch, _WITHOUT_TEMPERATURE)
        assert unsupported_params("openai/gpt-4o", []) == []
        assert seen == []

    def test_asks_litellm_about_the_canonical_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _stub_params(monkeypatch, _WITH_TEMPERATURE)
        unsupported_params("gemini-2.5-flash", ["tool_choice"])
        assert seen == ["gemini/gemini-2.5-flash"]


class TestUnsupportedParamsFallback:
    """Uncertainty reads as "supported" — a false banner is the worse error."""

    def test_empty_when_litellm_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, RuntimeError("litellm exploded"))
        assert unsupported_params("gemini/gemini-3.5-flash-lite", ["tool_choice"]) == []

    def test_empty_when_litellm_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, None)
        assert unsupported_params("some-model-we-cannot-place", ["tool_choice"]) == []

    def test_empty_when_litellm_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty list means LiteLLM knows nothing, not that nothing is supported."""
        _stub_params(monkeypatch, [])
        assert unsupported_params("some-model-we-cannot-place", ["tool_choice"]) == []


class TestSilentlyUnsentParams:
    """The hard-coded escape hatch for parameters LiteLLM claims but never sends.

    ``unsupported_params`` believes LiteLLM's own answer, which is right until
    LiteLLM lists a parameter its provider transformer then discards. Two such
    gaps exist on Gemini, so they are enumerated rather than probed — and the
    probe must not be consulted at all, since it would say "supported".
    """

    def test_gemini_never_sends_parallel_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, ["temperature", "tool_choice", "parallel_tool_calls"])
        assert silently_unsent_params(
            "gemini/gemini-2.5-flash", ["tool_choice", "parallel_tool_calls"]
        ) == ["parallel_tool_calls"]

    def test_gemini_never_sends_tool_strictness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert silently_unsent_params("gemini/gemini-2.5-pro", [TOOL_STRICT_PARAM]) == [
            TOOL_STRICT_PARAM
        ]

    def test_result_is_sorted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert silently_unsent_params(
            "gemini/gemini-2.5-pro", [TOOL_STRICT_PARAM, "parallel_tool_calls"]
        ) == ["parallel_tool_calls", TOOL_STRICT_PARAM]

    def test_other_providers_have_no_known_gaps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        asked = [TOOL_STRICT_PARAM, "parallel_tool_calls"]
        assert silently_unsent_params("openai/gpt-4o", asked) == []
        assert silently_unsent_params("anthropic/claude-4.5-sonnet", asked) == []

    def test_params_outside_the_table_are_never_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the enumerated gaps; everything else is ``unsupported_params``' job."""
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert (
            silently_unsent_params(
                "gemini/gemini-2.5-flash", ["tool_choice", "response_format", "max_tokens"]
            )
            == []
        )

    def test_does_not_consult_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: LiteLLM would answer "supported" for both of these."""
        seen = _stub_params(monkeypatch, RuntimeError("litellm must not be asked"))
        assert silently_unsent_params("gemini/gemini-2.5-flash", ["parallel_tool_calls"]) == [
            "parallel_tool_calls"
        ]
        assert seen == []

    def test_resolves_aliases_and_passthrough_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare alias and an unregistered preview id are both Gemini."""
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert silently_unsent_params("gemini-2.5-flash", ["parallel_tool_calls"]) == [
            "parallel_tool_calls"
        ]
        assert silently_unsent_params("gemini-3.5-flash-lite", ["parallel_tool_calls"]) == [
            "parallel_tool_calls"
        ]

    def test_empty_when_nothing_was_asked_about(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert silently_unsent_params("gemini/gemini-2.5-flash", []) == []

    def test_unplaceable_model_id_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No provider, no table entry — and never an exception out of a probe."""
        _stub_params(monkeypatch, _WITH_TEMPERATURE)
        assert silently_unsent_params("some-model-we-cannot-place", ["parallel_tool_calls"]) == []
