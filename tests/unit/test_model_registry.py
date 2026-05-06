"""Tests for :mod:`aimigrate.models.registry`."""

from __future__ import annotations

import pytest

from aimigrate.models.registry import (
    ModelMetadata,
    UnknownModelError,
    get_model,
    list_supported,
)


class TestGetModel:
    def test_lookup_by_canonical_id(self) -> None:
        meta = get_model("gemini/gemini-2.5-flash")
        assert meta.id == "gemini/gemini-2.5-flash"
        assert meta.provider == "google"

    def test_lookup_by_friendly_alias(self) -> None:
        # The PDF spec uses bare names; AIMigrate must accept them.
        meta = get_model("gemini-2.5-flash")
        assert meta.id == "gemini/gemini-2.5-flash"
        assert meta.display_name == "Gemini 2.5 Flash"

    def test_pdf_style_anthropic_alias(self) -> None:
        meta = get_model("claude-4.5-sonnet")
        assert meta.provider == "anthropic"
        assert meta.id.startswith("anthropic/")

    def test_pdf_style_openai_alias(self) -> None:
        meta = get_model("gpt-5-mini")
        assert meta.provider == "openai"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownModelError) as info:
            get_model("nonexistent/model-9000")
        assert info.value.requested == "nonexistent/model-9000"
        # Suggestions should be non-empty so users have a starting point.
        assert info.value.suggestions

    def test_unknown_model_error_message_includes_suggestions(self) -> None:
        with pytest.raises(UnknownModelError) as info:
            get_model("not-a-thing")
        assert "try one of" in str(info.value)


class TestListSupported:
    def test_returns_list_of_model_metadata(self) -> None:
        models = list_supported()
        assert all(isinstance(m, ModelMetadata) for m in models)
        assert len(models) >= 3  # at least the three providers covered

    def test_every_provider_represented(self) -> None:
        providers = {m.provider for m in list_supported()}
        assert providers == {"anthropic", "openai", "google"}

    def test_returns_fresh_list_each_call(self) -> None:
        # Caller mutation must not poison the registry.
        first = list_supported()
        first.clear()
        second = list_supported()
        assert second  # not empty


class TestRegistryIntegrity:
    def test_no_alias_collides_with_a_canonical_id(self) -> None:
        # If `_build_lookup` had silently overwritten a canonical id with
        # an alias, this test would fail because the alias's resolution
        # would point to the wrong model.
        for meta in list_supported():
            resolved = get_model(meta.id)
            assert resolved.id == meta.id

    def test_default_temperature_is_zero(self) -> None:
        # Determinism matters for evaluations; defaults must reflect that.
        for meta in list_supported():
            assert meta.default_temperature == 0.0
