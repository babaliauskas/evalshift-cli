"""Tests for :mod:`evalshift_cli.models.deepseek`.

LiteLLM's reasoning flag is stubbed in every test: the bundled table is
upstream data, and a LiteLLM upgrade must not flip these tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalshift_cli.models import deepseek as deepseek_module
from evalshift_cli.models.deepseek import (
    REASONING_PLACEHOLDER,
    backfill_reasoning_content,
    thinking_by_default,
)


def _stub_reasoning(monkeypatch: pytest.MonkeyPatch, result: bool | Exception) -> list[str]:
    seen: list[str] = []

    def fake(*, model: str, **_: Any) -> bool:
        seen.append(model)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(deepseek_module.litellm, "supports_reasoning", fake)
    return seen


class TestThinkingByDefault:
    def test_reasoning_deepseek_model_thinks_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _stub_reasoning(monkeypatch, True)
        assert thinking_by_default("deepseek-flash") is True
        # Asked about the canonical id, which is what LiteLLM keys on.
        assert seen == ["deepseek/deepseek-flash"]

    def test_non_reasoning_deepseek_model_does_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_reasoning(monkeypatch, False)
        assert thinking_by_default("deepseek/deepseek-chat") is False

    def test_uncertain_answer_reads_as_not_thinking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_reasoning(monkeypatch, RuntimeError("no model info"))
        assert thinking_by_default("deepseek-v5-preview") is False

    @pytest.mark.parametrize(
        "model_id", ["gemini/gemini-2.5-flash", "gpt-4o", "azure_ai/deepseek-v4-pro"]
    )
    def test_other_providers_never_ask(
        self, monkeypatch: pytest.MonkeyPatch, model_id: str
    ) -> None:
        seen = _stub_reasoning(monkeypatch, True)
        assert thinking_by_default(model_id) is False
        assert seen == []


class TestBackfillReasoningContent:
    def test_assistant_turn_without_reasoning_gets_the_placeholder(self) -> None:
        messages = [
            {"role": "user", "content": "find ACME"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_r0_0"}]},
            {"role": "tool", "tool_call_id": "call_r0_0", "content": "{}"},
        ]
        out = backfill_reasoning_content(messages)
        assert out[1]["reasoning_content"] == REASONING_PLACEHOLDER == " "
        assert "reasoning_content" not in out[0]
        assert "reasoning_content" not in out[2]

    def test_recorded_reasoning_is_kept(self) -> None:
        messages = [{"role": "assistant", "content": "x", "reasoning_content": "because"}]
        assert backfill_reasoning_content(messages)[0]["reasoning_content"] == "because"

    def test_input_is_not_mutated(self) -> None:
        turn = {"role": "assistant", "content": ""}
        backfill_reasoning_content([turn])
        assert "reasoning_content" not in turn
