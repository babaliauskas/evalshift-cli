"""Tests for ``aimigrate init`` (:mod:`aimigrate.cli.commands.init`).

The high-level invariant we care about: after ``aimigrate init`` runs in a
directory, ``aimigrate.yaml`` parses cleanly via :func:`load_config` and the
referenced ``prompts.py`` is importable. If that breaks, every new user has
a broken first experience.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aimigrate.cli.commands.doctor import CONFIG_FILENAME
from aimigrate.cli.commands.init import PROMPTS_FILENAME, SUITE_FILENAME
from aimigrate.cli.main import app
from aimigrate.config.loader import load_config

runner = CliRunner()


@pytest.fixture
def in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run inside ``tmp_path`` for the duration of a test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestInitHappy:
    def test_writes_three_starter_files(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / CONFIG_FILENAME).is_file()
        assert (in_tmp / PROMPTS_FILENAME).is_file()
        assert (in_tmp / SUITE_FILENAME).is_file()

    def test_written_config_parses_via_load_config(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert len(cfg.prompts) >= 1

    def test_written_prompts_module_is_valid_python(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        # Must parse via ``ast`` so the python_string parser (Phase 2.3)
        # will be able to AST-walk it.
        ast.parse((in_tmp / PROMPTS_FILENAME).read_text(encoding="utf-8"))

    def test_written_prompts_module_defines_referenced_variable(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        # For every python_string prompt, the referenced module-level name
        # must exist in ``prompts.py``.
        module_src = (in_tmp / PROMPTS_FILENAME).read_text(encoding="utf-8")
        tree = ast.parse(module_src)
        names = {
            tgt.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for tgt in node.targets
            if isinstance(tgt, ast.Name)
        }
        for prompt in cfg.prompts:
            if prompt.detection == "python_string" and prompt.variable:
                assert prompt.variable in names, (
                    f"prompts.py is missing variable {prompt.variable!r} referenced "
                    f"by aimigrate.yaml prompt {prompt.id!r}"
                )

    def test_written_suite_is_valid_jsonl(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        text = (in_tmp / SUITE_FILENAME).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert len(rows) >= 1
        for row in rows:
            assert "id" in row
            assert "inputs" in row

    def test_prints_next_steps(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert "aimigrate doctor" in result.stdout
        assert "aimigrate run" in result.stdout


# ---------------------------------------------------------------------------
# Conflict / overwrite handling
# ---------------------------------------------------------------------------


class TestInitConflicts:
    def test_refuses_to_overwrite_by_default(self, in_tmp: Path) -> None:
        (in_tmp / CONFIG_FILENAME).write_text("# user's existing config\n", encoding="utf-8")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.stdout
        assert (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8") == (
            "# user's existing config\n"
        )

    def test_force_overwrites(self, in_tmp: Path) -> None:
        (in_tmp / CONFIG_FILENAME).write_text("# old\n", encoding="utf-8")
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0, result.stdout
        # The file should now be the template, not the old stub.
        assert "version: 1" in (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")

    def test_lists_all_conflicting_files(self, in_tmp: Path) -> None:
        (in_tmp / CONFIG_FILENAME).write_text("x", encoding="utf-8")
        (in_tmp / PROMPTS_FILENAME).write_text("y", encoding="utf-8")
        result = runner.invoke(app, ["init"])
        assert CONFIG_FILENAME in result.stdout
        assert PROMPTS_FILENAME in result.stdout


# ---------------------------------------------------------------------------
# --directory
# ---------------------------------------------------------------------------


class TestInitDirectoryFlag:
    def test_writes_into_specified_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "fresh"
        result = runner.invoke(app, ["init", "--directory", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / CONFIG_FILENAME).is_file()
        # The cwd itself should be untouched.
        assert not (tmp_path / CONFIG_FILENAME).exists()

    def test_creates_directory_if_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "nested" / "deep"
        assert not target.exists()
        result = runner.invoke(app, ["init", "-d", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / CONFIG_FILENAME).is_file()
