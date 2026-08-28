"""CLI tests for ``evalshift runs clean`` via CliRunner."""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from evalshift.cli.main import app

runner = CliRunner()


def _make_run(base: Path, name: str, *, order: int = 0) -> Path:
    """Create a bare run dir; ``order`` sets a stable mtime (higher = newer)."""
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.utime(run_dir, (1_000_000 + order, 1_000_000 + order))
    return run_dir


def test_help_lists_clean() -> None:
    result = runner.invoke(app, ["runs", "--help"])
    assert result.exit_code == 0
    assert "clean" in result.stdout


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    for i in range(4):
        _make_run(base, f"r_2026070{i}_chat_00000{i}", order=i)
    result = runner.invoke(
        app, ["runs", "clean", "--keep", "1", "--dry-run", "--runs-base", str(base)]
    )
    assert result.exit_code == 0, result.output
    assert "Would delete 3" in result.output
    assert len(list(base.iterdir())) == 4  # nothing removed


def test_keep_deletes_oldest(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    for i in range(4):
        _make_run(base, f"r_2026070{i}_chat_00000{i}", order=i)
    result = runner.invoke(app, ["runs", "clean", "--keep", "1", "--yes", "--runs-base", str(base)])
    assert result.exit_code == 0, result.output
    assert "removed 3" in result.output
    survivors = {p.name for p in base.iterdir()}
    assert survivors == {"r_20260703_chat_000003"}


def test_suite_filter(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    for i in range(3):
        _make_run(base, f"r_2026070{i}_alpha_a0000{i}", order=i)
        _make_run(base, f"r_2026070{i}_beta_b0000{i}", order=i)
    result = runner.invoke(
        app,
        ["runs", "clean", "--keep", "1", "--suite", "alpha", "--yes", "--runs-base", str(base)],
    )
    assert result.exit_code == 0, result.output
    remaining = {p.name for p in base.iterdir()}
    # All 3 beta survive; only newest alpha survives.
    assert sum(1 for n in remaining if "beta" in n) == 3
    assert sum(1 for n in remaining if "alpha" in n) == 1


def test_nothing_to_clean_when_within_limits(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    _make_run(base, "r_20260701_chat_000001", order=1)
    result = runner.invoke(app, ["runs", "clean", "--keep", "5", "--yes", "--runs-base", str(base)])
    assert result.exit_code == 0, result.output
    assert "already within limits" in result.output
    assert (base / "r_20260701_chat_000001").exists()


def test_keep_zero_without_ttl_is_noop(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    for i in range(3):
        _make_run(base, f"r_2026070{i}_chat_00000{i}", order=i)
    result = runner.invoke(app, ["runs", "clean", "--keep", "0", "--yes", "--runs-base", str(base)])
    assert result.exit_code == 0, result.output
    assert "nothing to do" in result.output
    assert len(list(base.iterdir())) == 3  # untouched


def test_confirmation_abort_keeps_runs(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    for i in range(3):
        _make_run(base, f"r_2026070{i}_chat_00000{i}", order=i)
    # Answer "n" to the confirmation prompt.
    result = runner.invoke(
        app, ["runs", "clean", "--keep", "1", "--runs-base", str(base)], input="n\n"
    )
    assert result.exit_code != 0  # aborted
    assert len(list(base.iterdir())) == 3  # nothing removed
