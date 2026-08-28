"""Translating a recorded generation_config into dispatch kwargs."""

from __future__ import annotations

from evalshift.cache.store import cache_key
from evalshift.runner.generation import translate_generation_config


class TestTranslateGenerationConfig:
    def test_none_passthrough(self) -> None:
        assert translate_generation_config(None) == (None, None)

    def test_empty_dict_passthrough(self) -> None:
        assert translate_generation_config({}) == (None, None)

    def test_temperature_only(self) -> None:
        assert translate_generation_config({"temperature": 0.3}) == (0.3, None)

    def test_bool_temperature_is_not_a_temperature(self) -> None:
        assert translate_generation_config({"temperature": True}) == (None, None)

    def test_json_mime_without_schema_maps_to_json_object(self) -> None:
        _, extra = translate_generation_config({"response_mime_type": "application/json"})
        assert extra == {"response_format": {"type": "json_object"}}

    def test_json_mime_with_schema_maps_to_json_schema(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        _, extra = translate_generation_config(
            {"response_mime_type": "application/json", "response_schema": schema}
        )
        assert extra == {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "captured_schema", "schema": schema},
            }
        }

    def test_litellm_shaped_response_format_passes_through(self) -> None:
        rf = {"type": "json_object"}
        _, extra = translate_generation_config({"response_format": rf})
        assert extra == {"response_format": rf}

    def test_unknown_keys_ignored_and_garbage_is_safe(self) -> None:
        assert translate_generation_config({"beam_width": 4}) == (None, None)
        assert translate_generation_config({"temperature": "hot"}) == (None, None)


def _key_args() -> dict[str, object]:
    return {
        "model_id": "m",
        "prompt_text": "p",
        "inputs": {"a": 1},
        "temperature": 0.0,
        "max_tokens": 100,
    }


class TestCacheKeyGenerationConfig:
    def test_none_is_byte_stable_with_pre_change_keys(self) -> None:
        assert cache_key(**_key_args()) == cache_key(**_key_args(), generation_config=None)

    def test_config_changes_the_key(self) -> None:
        with_cfg = cache_key(
            **_key_args(), generation_config={"response_mime_type": "application/json"}
        )
        assert with_cfg != cache_key(**_key_args())

    def test_different_schemas_key_differently(self) -> None:
        a = cache_key(**_key_args(), generation_config={"response_schema": {"type": "object"}})
        b = cache_key(**_key_args(), generation_config={"response_schema": {"type": "array"}})
        assert a != b
