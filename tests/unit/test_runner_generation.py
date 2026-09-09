"""Translating a recorded generation_config into dispatch kwargs."""

from __future__ import annotations

import logging

import pytest

from evalshift_cli.cache.store import cache_key
from evalshift_cli.runner.generation import translate_generation_config


@pytest.fixture(autouse=True)
def _forget_ignored_key_warnings() -> None:
    """The ignored-keys warning is deduped process-wide; reset it per test."""
    from evalshift_cli.runner import generation

    generation._WARNED_IGNORED_KEYS.clear()


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

    def test_ignored_keys_are_warned_not_debugged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.generation"):
            translate_generation_config({"beam_width": 4})
        assert "beam_width" in caplog.text

    def test_ignored_keys_warn_once_per_key_set(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.generation"):
            translate_generation_config({"beam_width": 4})
            translate_generation_config({"beam_width": 9})
        assert caplog.text.count("beam_width") == 1


class TestToolChoice:
    """The three recorded spellings normalise to one OpenAI-style intent.

    LiteLLM translates OpenAI-style ``tool_choice`` / ``parallel_tool_calls``
    into each provider's own shape, so OpenAI-style is the single wire form
    the CLI emits (see :class:`TestLiteLLMTranslatesToolChoice` in
    ``tests/unit/test_model_client.py``).
    """

    @pytest.mark.parametrize("value", ["auto", "none", "required"])
    def test_openai_string_passes_through(self, value: str) -> None:
        _, extra = translate_generation_config({"tool_choice": value})
        assert extra == {"tool_choice": value}

    def test_openai_named_function_passes_through(self) -> None:
        choice = {"type": "function", "function": {"name": "search_db"}}
        _, extra = translate_generation_config({"tool_choice": choice})
        assert extra == {"tool_choice": choice}

    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [("auto", "auto"), ("any", "required"), ("none", "none")],
    )
    def test_anthropic_modes_normalise(self, recorded: str, expected: str) -> None:
        _, extra = translate_generation_config({"tool_choice": {"type": recorded}})
        assert extra == {"tool_choice": expected}

    def test_anthropic_named_tool_normalises(self) -> None:
        _, extra = translate_generation_config(
            {"tool_choice": {"type": "tool", "name": "search_db"}}
        )
        assert extra == {"tool_choice": {"type": "function", "function": {"name": "search_db"}}}

    def test_anthropic_disable_parallel_inverts(self) -> None:
        _, extra = translate_generation_config(
            {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
        )
        assert extra == {"tool_choice": "auto", "parallel_tool_calls": False}

    def test_anthropic_disable_parallel_false_means_parallel_allowed(self) -> None:
        _, extra = translate_generation_config(
            {"tool_choice": {"type": "any", "disable_parallel_tool_use": False}}
        )
        assert extra == {"tool_choice": "required", "parallel_tool_calls": True}

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("AUTO", "auto"), ("ANY", "required"), ("NONE", "none")],
    )
    def test_gemini_tool_config_modes_normalise(self, mode: str, expected: str) -> None:
        _, extra = translate_generation_config(
            {"tool_config": {"function_calling_config": {"mode": mode}}}
        )
        assert extra == {"tool_choice": expected}

    def test_gemini_single_allowed_function_becomes_a_named_choice(self) -> None:
        _, extra = translate_generation_config(
            {
                "tool_config": {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": ["search_db"],
                    }
                }
            }
        )
        assert extra == {"tool_choice": {"type": "function", "function": {"name": "search_db"}}}

    def test_gemini_multiple_allowed_functions_degrade_to_required(self) -> None:
        _, extra = translate_generation_config(
            {
                "tool_config": {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": ["a", "b"],
                    }
                }
            }
        )
        assert extra == {"tool_choice": "required"}

    def test_gemini_camel_case_rest_spelling_is_accepted(self) -> None:
        _, extra = translate_generation_config(
            {"tool_config": {"functionCallingConfig": {"mode": "ANY"}}}
        )
        assert extra == {"tool_choice": "required"}

    def test_top_level_parallel_tool_calls_passes_through(self) -> None:
        _, extra = translate_generation_config({"parallel_tool_calls": False})
        assert extra == {"parallel_tool_calls": False}

    def test_top_level_parallel_wins_over_anthropic_inversion(self) -> None:
        _, extra = translate_generation_config(
            {
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "parallel_tool_calls": True,
            }
        )
        assert extra == {"tool_choice": "auto", "parallel_tool_calls": True}

    def test_tool_choice_combines_with_response_format(self) -> None:
        _, extra = translate_generation_config(
            {"tool_choice": "auto", "response_format": {"type": "json_object"}}
        )
        assert extra == {
            "response_format": {"type": "json_object"},
            "tool_choice": "auto",
        }

    def test_recognised_keys_do_not_trip_the_ignored_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.generation"):
            translate_generation_config(
                {
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "tool_config": {"function_calling_config": {"mode": "AUTO"}},
                }
            )
        assert caplog.text == ""

    @pytest.mark.parametrize(
        "bad",
        [
            {"tool_choice": {"type": "wat"}},
            {"tool_choice": 7},
            {"tool_choice": {"nope": True}},
            {"tool_config": {"function_calling_config": {"mode": "SOMETIMES"}}},
            {"tool_config": "not a dict"},
            {"parallel_tool_calls": "yes"},
        ],
    )
    def test_unrecognised_shapes_are_warned_and_ignored(
        self, bad: dict[str, object], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.generation"):
            _, extra = translate_generation_config(bad)
        assert extra is None
        assert caplog.records


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
