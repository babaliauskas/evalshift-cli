"""Smoke test for the checked-in simple example."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.main import app

runner = CliRunner()


def test_simple_example_validates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    example_root = repo_root / "examples" / "simple"
    for name in ("evalshift.yaml", "golden.jsonl", "prompts.py"):
        shutil.copy2(example_root / name, tmp_path / name)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0, result.stdout
    assert "Loaded 1 prompt" in result.stdout
