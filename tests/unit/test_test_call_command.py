"""Tests for the ``aimigrate test-call`` command.

We monkeypatch the model client so no real LLM calls happen; the goal
is to verify CLI plumbing (arg parsing, alias resolution, exit codes,
error rendering), not LiteLLM behaviour.
"""

from __future__ import annotations

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
    def test_unknown_model_exits_one(self) -> None:
        result = runner.invoke(app, ["test-call", "--model", "no-such-model-9000"])
        assert result.exit_code == 1
        assert "unknown model" in result.stdout.lower()

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
