"""Tests for ``evalshift init`` (:mod:`evalshift.cli.commands.init`).

The high-level invariant we care about: after ``evalshift init`` runs in a
directory, ``evalshift.yaml`` parses cleanly via :func:`load_config` and the
referenced ``prompts.py`` is importable. If that breaks, every new user has
a broken first experience.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.cli.commands.init import (
    CI_WORKFLOW_PATH,
    PROMPTS_FILENAME,
    SUITE_FILENAME,
    TOOLS_FILENAME,
)
from evalshift.cli.main import app
from evalshift.config.loader import load_config

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
    def test_writes_four_starter_files(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / CONFIG_FILENAME).is_file()
        assert (in_tmp / PROMPTS_FILENAME).is_file()
        assert (in_tmp / TOOLS_FILENAME).is_file()
        assert (in_tmp / SUITE_FILENAME).is_file()

    def test_written_config_parses_via_load_config(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert len(cfg.prompts) >= 1
        # Single prompt should be wired up as an agent prompt.
        agent_prompts = [p for p in cfg.prompts if p.tools_path]
        assert agent_prompts, "expected at least one prompt with tools_path set"

    def test_config_does_not_enable_structural_length(self, in_tmp: Path) -> None:
        """structural.length scores 0/0 on agent runs (empty final_text).
        Make sure the scaffold doesn't include it as default noise."""
        runner.invoke(app, ["init"])
        cfg_text = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        # The block header itself shouldn't appear in a non-comment line.
        non_comment = "\n".join(
            line for line in cfg_text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "structural:" not in non_comment

    def test_profile_adds_migration_policy(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--profile", "cost-reduction"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# migration_profile: cost-reduction" in body
        assert "migration_policy:" in body
        assert "max_cost_increase: 0.05" in body

    def test_pack_is_recorded_in_config(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--pack", "tool-calling-agent"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# scenario_pack: tool-calling-agent" in body

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
                    f"by evalshift.yaml prompt {prompt.id!r}"
                )

    def test_tools_yaml_ships_six_tools(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        tools_text = (in_tmp / TOOLS_FILENAME).read_text(encoding="utf-8")
        # Each tool definition starts with `- name:` at column 0.
        tool_count = sum(1 for line in tools_text.splitlines() if line.startswith("- name:"))
        assert tool_count == 6, f"expected 6 tools, got {tool_count}"
        # And specifically these names should all appear.
        for name in (
            "search_orders",
            "lookup_customer",
            "issue_refund",
            "update_order_status",
            "send_email",
            "notify_security_team",
        ):
            assert f"- name: {name}" in tools_text

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
        assert "evalshift doctor" in result.stdout
        assert "evalshift run" in result.stdout


# ---------------------------------------------------------------------------
# Suite shape — locks in the post-Phase-6 fix
# ---------------------------------------------------------------------------


class TestInitSuiteShape:
    """The default suite must be big enough that the analysis layer doesn't
    skip with ``insufficient — small sample``.

    ``MIN_N_FOR_TEST = 5`` and ``MIN_N_RELIABLE = 20`` in
    ``src/evalshift/analysis/statistics.py``. The default scaffold should
    clear ``MIN_N_RELIABLE`` so brand-new users see real severity badges
    on their first analyze run, not a warning row.
    """

    def test_suite_has_at_least_40_rows(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        text = (in_tmp / SUITE_FILENAME).read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.strip()]
        assert len(rows) >= 40, (
            f"default suite shipped {len(rows)} rows; need >= 40 for each "
            f"slice to clear n>=5 and 'all' to clear n>=20"
        )

    def test_suite_covers_every_configured_slice(self, in_tmp: Path) -> None:
        """Every slice tag declared in evalshift.yaml should have rows."""
        runner.invoke(app, ["init"])
        text = (in_tmp / SUITE_FILENAME).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        tags = {tag for row in rows for tag in row.get("tags", [])}
        for slice_tag in ("security", "routine", "refund", "customer_lookup", "text_only"):
            assert slice_tag in tags, f"slice {slice_tag!r} has no rows"

    def test_suite_mixes_expected_tools_and_expected_no_tools(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        text = (in_tmp / SUITE_FILENAME).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert any("expected_tools" in r for r in rows), "no expected_tools rows"
        assert any(r.get("expected_no_tools") is True for r in rows), "no expected_no_tools rows"


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

    def test_refuses_to_overwrite_existing_tools_yaml(self, in_tmp: Path) -> None:
        (in_tmp / TOOLS_FILENAME).write_text("# user's tools\n", encoding="utf-8")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.stdout
        assert (in_tmp / TOOLS_FILENAME).read_text(encoding="utf-8") == "# user's tools\n"


# ---------------------------------------------------------------------------
# --directory
# ---------------------------------------------------------------------------


class TestInitCI:
    """``--ci`` drops a GitHub Actions workflow that wires the gate."""

    def test_no_ci_flag_skips_workflow(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        assert not (in_tmp / CI_WORKFLOW_PATH).exists()

    def test_ci_flag_writes_workflow(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--ci"])
        assert result.exit_code == 0, result.stdout
        wf = in_tmp / CI_WORKFLOW_PATH
        assert wf.is_file()
        body = wf.read_text(encoding="utf-8")
        # The workflow must use the reusable hosted action and token contract.
        assert "EVALSHIFT_NONINTERACTIVE" in body
        assert "babaliauskas/evalshift-action@v0" in body
        assert "EVALSHIFT_TOKEN" in body
        assert "fail-on: regression" in body
        assert "pull-requests: write" in body
        assert "statuses: write" in body

    def test_ci_flag_refuses_to_overwrite_existing_workflow(self, in_tmp: Path) -> None:
        wf = in_tmp / CI_WORKFLOW_PATH
        wf.parent.mkdir(parents=True)
        wf.write_text("# user's workflow\n", encoding="utf-8")
        result = runner.invoke(app, ["init", "--ci"])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.stdout
        assert wf.read_text(encoding="utf-8") == "# user's workflow\n"


class TestInitDirectoryFlag:
    def test_writes_into_specified_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "fresh"
        result = runner.invoke(app, ["init", "--directory", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / CONFIG_FILENAME).is_file()
        assert (target / TOOLS_FILENAME).is_file()
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
