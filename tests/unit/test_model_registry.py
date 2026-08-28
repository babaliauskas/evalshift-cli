"""Tests for :mod:`evalshift.models.registry`."""

from __future__ import annotations

import pytest

from evalshift.models.registry import (
    ModelMetadata,
    UnknownModelError,
    get_model,
    list_supported,
    resolve_model,
)


class TestGetModel:
    def test_lookup_by_canonical_id(self) -> None:
        meta = get_model("gemini/gemini-2.5-flash")
        assert meta.id == "gemini/gemini-2.5-flash"
        assert meta.provider == "google"

    def test_default_max_tokens_is_4096(self) -> None:
        # The fallback cap used when config supplies no max_tokens.
        assert get_model("gemini/gemini-2.5-flash").default_max_tokens == 4096

    def test_lookup_by_friendly_alias(self) -> None:
        # The PDF spec uses bare names; EvalShift must accept them.
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


class TestResolveModel:
    def test_known_alias_uses_registry(self) -> None:
        meta = resolve_model("gemini-2.5-flash")
        assert meta.id == "gemini/gemini-2.5-flash"
        assert meta.provider == "google"
        # Registered models don't carry the passthrough marker.
        assert "(passthrough)" not in meta.display_name

    def test_unknown_gemini_prefix_inferred(self) -> None:
        meta = resolve_model("gemini-3.1-flash-lite-preview")
        assert meta.id == "gemini/gemini-3.1-flash-lite-preview"
        assert meta.provider == "google"
        assert meta.display_name.endswith("(passthrough)")

    def test_unknown_claude_prefix_inferred(self) -> None:
        meta = resolve_model("claude-99-sonnet")
        assert meta.id == "anthropic/claude-99-sonnet"
        assert meta.provider == "anthropic"

    def test_unknown_gpt_prefix_inferred(self) -> None:
        meta = resolve_model("gpt-9-mini")
        assert meta.id == "openai/gpt-9-mini"
        assert meta.provider == "openai"

    def test_o1_prefix_inferred_as_openai(self) -> None:
        meta = resolve_model("o1-mini")
        assert meta.id == "openai/o1-mini"
        assert meta.provider == "openai"

    def test_already_prefixed_id_passes_through_unchanged(self) -> None:
        meta = resolve_model("gemini/gemini-3.1-flash-lite-preview")
        assert meta.id == "gemini/gemini-3.1-flash-lite-preview"
        assert meta.provider == "google"

    def test_truly_unknown_id_falls_back_to_other(self) -> None:
        meta = resolve_model("totally-novel-model")
        assert meta.id == "totally-novel-model"
        assert meta.provider == "other"

    def test_resolve_never_raises(self) -> None:
        # The whole point: no input value short of `None` should raise.
        for input_id in ("", "x", "/", "vendor/", "weird-vendor/model"):
            assert isinstance(resolve_model(input_id), ModelMetadata)


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
