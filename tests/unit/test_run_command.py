"""Tests for the ``evalshift run`` CLI command.

Higher-level than ``test_orchestrator.py``: we drive the actual Typer
command via :class:`CliRunner` and assert exit codes plus key bits of
the rendered output. The model client is monkeypatched so no real LLM
calls happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift.cli.main import app
from evalshift.models.client import CompletionResult, ModelClient

runner = CliRunner()


def _scaffold(tmp_path: Path, n_examples: int = 2) -> Path:
    """Lay out a minimal valid project under ``tmp_path``."""
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
          concurrency: 4
        """,
        encoding="utf-8",
    )
    rows = "\n".join(
        f'{{"id": "ex{i}", "inputs": {{"name": "User{i}"}}}}' for i in range(n_examples)
    )
    (tmp_path / "golden.jsonl").write_text(rows + "\n", encoding="utf-8")
    return tmp_path


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        return CompletionResult(
            text="ok",
            model_id=str(kwargs["model"]),
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.0,
            latency_ms=10,
        )

    monkeypatch.setattr(ModelClient, "complete", fake_complete)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunHappy:
    def test_run_completes_with_default_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 0, result.stdout
        assert "completed" in result.stdout.lower() or "calls:" in result.stdout
        # The run dir should exist under .evalshift/runs/.
        runs = list((tmp_path / ".evalshift" / "runs").iterdir())
        assert len(runs) == 1

    def test_from_to_flags_override_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "run",
                "--from",
                "gemini-2.5-pro",
                "--to",
                "gemini-2.5-flash",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.stdout

    def test_summary_includes_run_id(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert "r_" in result.stdout
        assert "evalshift run" in result.stdout


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRunErrors:
    def test_missing_config_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Invalid config" in result.stdout

    def test_missing_suite_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Valid config, but no golden.jsonl in the cwd.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n"
            "defaults:\n  source_model: gemini-2.5-flash\n"
            "  target_model: gemini-2.5-pro\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Invalid suite" in result.stdout

    def test_missing_models_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Valid config + suite, but no defaults.source_model and no --from.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        (tmp_path / "golden.jsonl").write_text('{"id": "ex1", "inputs": {}}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "missing model selection" in result.stdout

    def test_incompatible_suite_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Prompt requires `name`; suite provides only `tone`.
        (tmp_path / "evalshift.yaml").write_text(
            """
            version: 1
            prompts:
              - id: greet
                detection: manual
                content: "Hi {name}"
                variables: [name]
            defaults:
              source_model: gemini-2.5-flash
              target_model: gemini-2.5-pro
            """,
            encoding="utf-8",
        )
        (tmp_path / "golden.jsonl").write_text(
            '{"id": "ex1", "inputs": {"tone": "formal"}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Suite incompatible" in result.stdout
        assert "name" in result.stdout
