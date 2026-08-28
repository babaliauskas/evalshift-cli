"""Tests for :mod:`evalshift.models.capabilities`.

The registry is static data; this module asks LiteLLM at call time whether a
model still honours ``temperature``. LiteLLM is the authority because the ids
that matter here (fresh Gemini previews) are passthrough ids the registry has
never heard of.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from evalshift.models.capabilities import honors_temperature

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
