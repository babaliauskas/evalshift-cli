"""Tests for ``evalshift all --suite-name``.

``all`` gained a ``--suite-name`` option that resolves a named suite from
``evalshift.yaml`` (mirroring ``run``). The path-resolution logic itself is
covered by ``test_run_suite_name``; here we check the option is wired into
``all`` and that an unknown name fails cleanly before any run work happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.main import app

runner = CliRunner()


def test_all_help_lists_suite_name() -> None:
    result = runner.invoke(app, ["all", "--help"])
    assert result.exit_code == 0
    assert "--suite-name" in result.stdout


def test_all_unknown_suite_name_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    # init's scaffold sets defaults.source_model but leaves target_model
    # commented out; --to supplies it directly so model selection resolves
    # and suite-name resolution is reached (and fails) before any dispatch.
    assert runner.invoke(app, ["init"]).exit_code == 0
    result = runner.invoke(app, ["all", "--to", "gemini-3.1-pro-preview", "--suite-name", "nope"])
    assert result.exit_code == 1
    assert "unknown --suite-name" in result.stdout.lower()
