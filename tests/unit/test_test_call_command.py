"""Tests for the ``aimigrate test-call`` command.

We monkeypatch the model client so no real LLM calls happen; the goal
is to verify CLI plumbing (arg parsing, alias resolution, exit codes,
error rendering), not LiteLLM behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from aimigrate.cli.commands import test_call as cmd
from aimigrate.cli.main import app
from aimigrate.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    RateLimitError,
)

runner = CliRunner()


def _good_result(model: str = "gemini/gemini-2.5-flash") -> CompletionResult:
    return CompletionResult(
        text="Hi there!",
        model_id=model,
        input_tokens=11,
        output_tokens=4,
        cost_usd=0.000123,
        latency_ms=312,
    )


def _patch_complete(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raises: Exception | None = None,
    result: CompletionResult | None = None,
) -> None:
    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        if raises is not None:
            raise raises
        return result or _good_result(kwargs["model"])

    monkeypatch.setattr(cmd.ModelClient, "complete", fake_complete)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTestCallHappy:
    def test_prints_response_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_complete(monkeypatch)
        result = runner.invoke(
            app,
            ["test-call", "--model", "gemini/gemini-2.5-flash", "--prompt", "hi"],
        )
        assert result.exit_code == 0, result.stdout
        assert "Hi there!" in result.stdout

    def test_shows_canonical_id_and_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_complete(monkeypatch, result=_good_result("gemini/gemini-2.5-flash"))
        result = runner.invoke(
            app,
            ["test-call", "--model", "gemini-2.5-flash"],
        )
        assert result.exit_code == 0, result.stdout
        # Canonical id is shown as the dispatch target.
        assert "gemini/gemini-2.5-flash" in result.stdout
        # And the alias appears in the resolution hint.
        assert "alias: gemini-2.5-flash" in result.stdout

    def test_panel_has_token_and_cost_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_complete(monkeypatch)
        result = runner.invoke(app, ["test-call", "--model", "gemini/gemini-2.5-flash"])
        assert "tokens:" in result.stdout
        assert "in=11" in result.stdout
        assert "out=4" in result.stdout
        assert "cost:" in result.stdout
        assert "ms" in result.stdout

    def test_default_prompt_used_when_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            captured.update(kwargs)
            return _good_result()

        monkeypatch.setattr(cmd.ModelClient, "complete", fake_complete)
        result = runner.invoke(app, ["test-call", "--model", "gemini-2.5-flash"])
        assert result.exit_code == 0, result.stdout
        # The default prompt should reach the model.
        assert isinstance(captured.get("prompt"), str)
        assert captured["prompt"].strip() != ""

    def test_temperature_and_max_tokens_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            captured.update(kwargs)
            return _good_result()

        monkeypatch.setattr(cmd.ModelClient, "complete", fake_complete)
        result = runner.invoke(
            app,
            [
                "test-call",
                "--model",
                "gemini-2.5-flash",
                "--prompt",
                "x",
                "--temperature",
                "0.7",
                "--max-tokens",
                "32",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert captured["temperature"] == pytest.approx(0.7)
        assert captured["max_tokens"] == 32


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestTestCallErrors:
    def test_unknown_model_passes_through_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Per the resolver contract: registry-unknown ids are passed
        # through to LiteLLM with a soft warning. The call still
        # happens; we just signal the user is outside the curated set.
        _patch_complete(monkeypatch)
        result = runner.invoke(
            app,
            ["test-call", "--model", "gemini-3.1-flash-lite-preview"],
        )
        assert result.exit_code == 0, result.stdout
        assert "not in AIMigrate registry" in result.stdout
        # The synthesised id should pick up the gemini/ prefix.
        assert "gemini/gemini-3.1-flash-lite-preview" in result.stdout

    def test_auth_error_renders_helpfully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_complete(monkeypatch, raises=AuthError("bad key"))
        result = runner.invoke(app, ["test-call", "--model", "gemini-2.5-flash"])
        assert result.exit_code == 1
        assert "Authentication failed" in result.stdout

    def test_rate_limit_error_renders_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_complete(monkeypatch, raises=RateLimitError("too fast"))
        result = runner.invoke(app, ["test-call", "--model", "gemini-2.5-flash"])
        assert result.exit_code == 1
        assert "rate-limited" in result.stdout


class TestHidden:
    def test_test_call_is_hidden_from_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "test-call" not in result.stdout


# ---------------------------------------------------------------------------
# --tools mode
# ---------------------------------------------------------------------------


from aimigrate.evaluators.tool_models import ToolCall, ToolTrace  # noqa: E402
from aimigrate.models.client import ToolCompletionResult  # noqa: E402


def _patch_tool_complete(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raises: Exception | None = None,
    result: ToolCompletionResult | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake(self: ModelClient, **kwargs: Any) -> ToolCompletionResult:
        captured.update(kwargs)
        if raises is not None:
            raise raises
        return result or ToolCompletionResult(
            trace=ToolTrace(
                calls=[
                    ToolCall(
                        tool_name="search_db",
                        arguments={"query": "ACME"},
                        sequence_index=0,
                    ),
                ],
                final_text=None,
            ),
            model_id=str(kwargs["model"]),
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.0001,
            latency_ms=42,
            raw_provider_response={},
        )

    monkeypatch.setattr(cmd.ModelClient, "complete_with_tools", fake)
    return captured


class TestTestCallTools:
    def test_tools_mode_prints_tool_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tools_path = tmp_path / "tools.yaml"
        tools_path.write_text(
            "- name: search_db\n"
            "  description: Search the customer DB.\n"
            "  input_schema: {type: object}\n",
            encoding="utf-8",
        )
        captured = _patch_tool_complete(monkeypatch)
        result = runner.invoke(
            app,
            [
                "test-call",
                "--model",
                "gemini-2.5-flash",
                "--tools",
                str(tools_path),
                "--prompt",
                "find ACME",
            ],
        )
        assert result.exit_code == 0, result.stdout
        # Tool name appears in the rendered panel.
        assert "search_db" in result.stdout
        # The wired call passed the tools list through.
        assert "tools" in captured
        assert captured["tools"][0].name == "search_db"

    def test_tools_mode_with_missing_file_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "test-call",
                "--model",
                "gemini-2.5-flash",
                "--tools",
                str(tmp_path / "nope.yaml"),
            ],
        )
        assert result.exit_code == 1
        assert "Invalid tools file" in result.stdout

    def test_tools_mode_renders_refusal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tools_path = tmp_path / "tools.yaml"
        tools_path.write_text(
            "- name: x\n  description: y\n  input_schema: {}\n",
            encoding="utf-8",
        )
        result_obj = ToolCompletionResult(
            trace=ToolTrace(
                calls=[],
                raised_refusal=True,
                refusal_text="Cannot help with that.",
            ),
            model_id="gemini/gemini-2.5-flash",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=10,
            raw_provider_response={},
        )
        _patch_tool_complete(monkeypatch, result=result_obj)
        result = runner.invoke(
            app,
            [
                "test-call",
                "--model",
                "gemini-2.5-flash",
                "--tools",
                str(tools_path),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "refusal" in result.stdout.lower()
