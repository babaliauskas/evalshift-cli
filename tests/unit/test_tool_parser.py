"""Tests for :mod:`aimigrate.evaluators.tool_parser`.

Each provider has a small set of fixtures under
``tests/unit/fixtures/tool_responses/<provider>/`` covering the response
shapes the smoke-test script will populate from real API calls. We test
both the happy paths (one assertion per fixture) and an aggressive set
of malformed-input cases to lock in the parser's defensive behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aimigrate.evaluators.tool_parser import (
    ToolParseError,
    detect_provider,
    parse_response_to_trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tool_responses"


def _load(provider: str, name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / provider / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------


class TestDetectProvider:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("anthropic/claude-sonnet-4-5", "anthropic"),
            ("claude-4.5-sonnet", "anthropic"),
            ("claude-3-opus", "anthropic"),
            ("openai/gpt-4o", "openai"),
            ("gpt-4o", "openai"),
            ("gpt-5-mini", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("gemini/gemini-2.5-pro", "gemini"),
            ("gemini-2.5-flash", "gemini"),
        ],
    )
    def test_known_models(self, model_id: str, expected: str) -> None:
        assert detect_provider(model_id) == expected

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ToolParseError) as info:
            detect_provider("mystery/not-a-real-model")
        assert info.value.provider == "unknown"
        assert "mystery" in str(info.value)


# ---------------------------------------------------------------------------
# Anthropic parser
# ---------------------------------------------------------------------------


class TestParseAnthropic:
    def test_single_tool_call(self) -> None:
        raw = _load("anthropic", "single_tool_call")
        trace = parse_response_to_trace(
            raw, provider="anthropic", model_id="anthropic/claude-sonnet-4-5"
        )
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"
        assert trace.calls[0].arguments == {"query": "ACME Q3"}
        assert trace.calls[0].sequence_index == 0
        assert trace.calls[0].call_id == "toolu_01ABC"
        assert trace.final_text is None
        assert not trace.raised_refusal

    def test_parallel_tool_calls(self) -> None:
        raw = _load("anthropic", "parallel_tool_calls")
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-sonnet-4-5")
        assert trace.call_count == 2
        assert trace.has_parallel_calls()
        assert all(c.parent_call_id is None for c in trace.calls)
        # Sequence indices are dense after _renumber.
        assert [c.sequence_index for c in trace.calls] == [0, 1]

    def test_text_only(self) -> None:
        raw = _load("anthropic", "text_only")
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-sonnet-4-5")
        assert trace.call_count == 0
        assert trace.final_text == "Our standard refund policy is 30 days."

    def test_tool_call_with_text(self) -> None:
        raw = _load("anthropic", "tool_call_with_text")
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-sonnet-4-5")
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"
        # The text block before the tool_use is preserved as final_text.
        assert trace.final_text == "Looking that up now."

    def test_refusal(self) -> None:
        raw = _load("anthropic", "refusal")
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-sonnet-4-5")
        assert trace.call_count == 0
        assert trace.raised_refusal
        assert trace.refusal_text == "I cannot help with that request."

    def test_unexpected_shape_raises_clear_error(self) -> None:
        with pytest.raises(ToolParseError) as info:
            parse_response_to_trace(
                {"unexpected": "shape"},
                provider="anthropic",
                model_id="claude-sonnet-4-5",
            )
        assert info.value.provider == "anthropic"

    def test_string_content_treated_as_text(self) -> None:
        raw = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-3-haiku")
        assert trace.final_text == "hi"
        assert trace.call_count == 0

    def test_none_content(self) -> None:
        raw = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-3")
        assert trace.call_count == 0
        assert trace.final_text is None

    def test_non_dict_content_block_raises(self) -> None:
        raw = {"choices": [{"message": {"content": ["not-a-dict"]}}]}
        with pytest.raises(ToolParseError, match="not a dict"):
            parse_response_to_trace(raw, provider="anthropic", model_id="claude-3")

    def test_unknown_block_type_ignored(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "thinking", "text": "hmm"},
                            {"type": "text", "text": "hi"},
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-3")
        assert trace.call_count == 0
        assert trace.final_text == "hi"

    def test_anthropic_via_openai_normalisation(self) -> None:
        """When LiteLLM normalises an Anthropic response into OpenAI shape,
        we should still parse it correctly via the openai-shape branch."""
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_norm",
                                "type": "function",
                                "function": {
                                    "name": "search_db",
                                    "arguments": '{"query": "ACME"}',
                                },
                            },
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="anthropic", model_id="claude-sonnet-4-5")
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"
        assert trace.calls[0].arguments == {"query": "ACME"}


# ---------------------------------------------------------------------------
# OpenAI parser
# ---------------------------------------------------------------------------


class TestParseOpenAI:
    def test_single_tool_call(self) -> None:
        raw = _load("openai", "single_tool_call")
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"
        assert trace.calls[0].arguments == {"query": "ACME Q3"}
        assert trace.calls[0].call_id == "call_synthetic_1"

    def test_arguments_decoded_from_json_string(self) -> None:
        raw = _load("openai", "single_tool_call")
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        # Defining property: arguments is a dict, not a string.
        assert isinstance(trace.calls[0].arguments, dict)

    def test_parallel_tool_calls(self) -> None:
        raw = _load("openai", "parallel_tool_calls")
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.call_count == 2
        assert trace.has_parallel_calls()

    def test_text_only(self) -> None:
        raw = _load("openai", "text_only")
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.call_count == 0
        assert trace.final_text == "Our standard refund policy is 30 days."

    def test_refusal(self) -> None:
        raw = _load("openai", "refusal")
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.call_count == 0
        assert trace.raised_refusal
        assert trace.refusal_text == "I cannot help with that request."

    def test_malformed_arguments_marked(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": "{not valid json",
                                },
                            },
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.calls[0].arguments.get("_parse_error") is True
        assert "_raw" in trace.calls[0].arguments

    def test_arguments_already_dict_passes_through(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": {"already": "dict"},
                                },
                            },
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.calls[0].arguments == {"already": "dict"}

    def test_empty_arguments_string_yields_empty_dict(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc",
                                "type": "function",
                                "function": {"name": "ping", "arguments": ""},
                            },
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.calls[0].arguments == {}

    def test_missing_tool_name_raises(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "tc", "type": "function", "function": {"name": ""}},
                        ],
                    },
                },
            ],
        }
        with pytest.raises(ToolParseError, match=r"missing function\.name"):
            parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")

    def test_unexpected_shape_raises(self) -> None:
        with pytest.raises(ToolParseError, match="unexpected response shape"):
            parse_response_to_trace({"nope": True}, provider="openai", model_id="gpt-4o")

    def test_message_not_dict_raises(self) -> None:
        with pytest.raises(ToolParseError, match="expected message dict"):
            parse_response_to_trace(
                {"choices": [{"message": "string-not-dict"}]},
                provider="openai",
                model_id="gpt-4o",
            )

    def test_tool_calls_not_list_raises(self) -> None:
        with pytest.raises(ToolParseError, match="expected list for tool_calls"):
            parse_response_to_trace(
                {"choices": [{"message": {"tool_calls": "string"}}]},
                provider="openai",
                model_id="gpt-4o",
            )

    def test_tool_calls_missing_function_dict_raises(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "tc", "type": "function", "function": "x"}],
                    },
                },
            ],
        }
        with pytest.raises(ToolParseError, match="function is not a dict"):
            parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")

    def test_tool_call_entry_not_dict_raises(self) -> None:
        raw = {
            "choices": [
                {"message": {"content": None, "tool_calls": ["not-a-dict"]}},
            ],
        }
        with pytest.raises(ToolParseError, match="not a dict"):
            parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")

    def test_refusal_only_when_no_tool_calls(self) -> None:
        # If both refusal and tool_calls are present, prefer the calls
        # (the model executed something despite the refusal field).
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "refusal": "shouldn't happen",
                        "tool_calls": [
                            {
                                "id": "tc",
                                "type": "function",
                                "function": {"name": "x", "arguments": "{}"},
                            },
                        ],
                    },
                },
            ],
        }
        trace = parse_response_to_trace(raw, provider="openai", model_id="gpt-4o")
        assert trace.call_count == 1
        assert not trace.raised_refusal


# ---------------------------------------------------------------------------
# Gemini parser (delegates to OpenAI shape)
# ---------------------------------------------------------------------------


class TestParseGemini:
    def test_single_tool_call(self) -> None:
        raw = _load("gemini", "single_tool_call")
        trace = parse_response_to_trace(raw, provider="gemini", model_id="gemini/gemini-2.5-pro")
        assert trace.call_count == 1
        assert trace.calls[0].tool_name == "search_db"

    def test_text_only(self) -> None:
        raw = _load("gemini", "text_only")
        trace = parse_response_to_trace(raw, provider="gemini", model_id="gemini/gemini-2.5-pro")
        assert trace.call_count == 0
        assert trace.final_text is not None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ToolParseError, match="unsupported provider"):
            parse_response_to_trace({"choices": []}, provider="acme-llm", model_id="acme/x")
