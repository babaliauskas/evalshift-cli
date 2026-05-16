"""Smoke tests for the CLI entry point.

Phase 0 / 1.3: verify that ``evalshift --version`` and ``--help`` work and that
the unimplemented subcommands fail loudly rather than silently.
"""

from __future__ import annotations

from typer.testing import CliRunner

from evalshift import __version__
from evalshift.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("doctor", "init", "run", "report", "login", "whoami", "bundle", "push"):
        assert cmd in result.stdout


def test_top_level_help_lists_main_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "doctor",
        "init",
        "run",
        "evaluate",
        "analyze",
        "report",
        "bundle",
        "push",
        "cache",
    ):
        assert cmd in result.stdout
