"""Tests for ``evalshift init`` (:mod:`evalshift.cli.commands.init`).

``init`` writes a single, minimal, capture-ready ``evalshift.yaml`` — no demo
data. The invariant we care about: the file parses cleanly via
:func:`load_config`, is wired for the capture-first flow (a passthrough prompt
and output evaluators), and carries the marker block that ``capture sync``
rewrites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.commands._agents import (
    AGENT_INSTRUCTIONS_FILENAME,
    DEFAULT_AGENT_CONTEXT_FILE,
    POINTER_MARKER_BEGIN,
)
from evalshift.cli.commands._scaffold import CI_WORKFLOW_PATH
from evalshift.cli.commands._suites import SUITE_FILENAME, SUITES_MARKER_BEGIN, SUITES_MARKER_END
from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.cli.main import app
from evalshift.config.loader import load_config

runner = CliRunner()

# Demo-scaffold file names `init` must never write (see
# TestInitHappy.test_writes_only_the_config below). `_scaffold.py` used to
# define these and a deleted `demo` command used to write them; T7 removed
# both, so these are now purely local vocabulary for the negative
# assertion rather than production constants.
FIXTURES_FILENAME = "fixtures.jsonl"
PROMPTS_FILENAME = "prompts.py"
TOOLS_FILENAME = "tools.yaml"


@pytest.fixture
def in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run inside ``tmp_path`` for the duration of a test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestInitHappy:
    def test_writes_only_the_config(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / CONFIG_FILENAME).is_file()
        # init must NOT scaffold any demo data.
        for name in (PROMPTS_FILENAME, TOOLS_FILENAME, SUITE_FILENAME, FIXTURES_FILENAME):
            assert not (in_tmp / name).exists(), f"init unexpectedly wrote {name}"

    def test_written_config_parses_via_load_config(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert len(cfg.prompts) == 1
        prompt = cfg.prompts[0]
        assert prompt.id == "replay"
        assert prompt.detection == "manual"
        assert prompt.content == "{input}"
        assert prompt.variables == ["input"]
        # Empty suites block — capture sync fills it.
        assert cfg.suites == {}

    def test_config_wires_capture_first_evaluators(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "gemini/gemini-embedding-001"
        assert len(cfg.evaluators.llm_judge) == 1
        assert cfg.evaluators.llm_judge[0].judge_model == "gemini-3.1-pro-preview"
        # No tool evaluators in the minimal (non-agent) scaffold.
        assert not cfg.evaluators.tool_selection
        assert not cfg.evaluators.tool_arguments

    def test_config_defers_tool_evaluators_to_capture_sync(self, in_tmp: Path) -> None:
        """No commented-out tool block to uncomment: sync wires them per suite."""
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# tool_selection:" not in body
        assert "# tool_arguments:" not in body
        assert "capture sync" in body

    def test_config_documents_the_project_field(self, in_tmp: Path) -> None:
        """The hosted slug is opt-in, but the scaffold has to name it.

        Nothing in a local run needs ``project:``, so init must not invent one --
        but a user who never sees the key does not learn it exists until a push
        fails for the lack of it.
        """
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# project: your-org/your-project" in body
        assert "push" in body
        # Commented, not set: init must not guess a slug on the user's behalf.
        assert load_config(in_tmp / CONFIG_FILENAME).project is None

    def test_commented_project_slug_parses_once_uncommented(self, in_tmp: Path) -> None:
        """The placeholder must satisfy the ``org/project`` pattern.

        A scaffolded example that fails validation the moment it is uncommented
        teaches the wrong shape and fails at the user's first push.
        """
        runner.invoke(app, ["init"])
        path = in_tmp / CONFIG_FILENAME
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "# project: your-org/your-project",
                "project: your-org/your-project",
            ),
            encoding="utf-8",
        )
        assert load_config(path).project == "your-org/your-project"

    def test_config_carries_suites_markers(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert SUITES_MARKER_BEGIN in body
        assert SUITES_MARKER_END in body
        assert "suites: {}" in body

    def test_suites_region_is_last_in_the_file(self, in_tmp: Path) -> None:
        """The managed region goes at the tail, after ``migration_policy``.

        It is the only part of the file a command rewrites, and the only part
        that grows without bound -- one entry per suite, each carrying its own
        evaluator block. Last means a sync's diff stays confined to the tail and
        the hand-edited config above it keeps stable line numbers.
        """
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert body.index("migration_policy:") < body.index(SUITES_MARKER_BEGIN)
        assert body.rstrip().endswith(SUITES_MARKER_END)

    def test_profile_adds_migration_policy(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--profile", "cost-reduction"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# migration_profile: cost-reduction" in body
        assert "migration_policy:" in body
        assert "max_cost_increase: 0.05" in body

    def test_default_profile_budgets_leave_room_for_a_first_migration(self, in_tmp: Path) -> None:
        """A fresh suite should report its regressions, not fail on two of them."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# migration_profile: model-upgrade" in body
        for line in (
            "max_overall_regression_rate: 0.30",
            "max_critical_regressions: 1",
            "min_equivalence_rate: 0.75",
            "max_tool_argument_drift: 0.20",
            "max_tool_divergence: 0.20",
            "max_cost_increase: 0.30",
            "max_latency_increase: 0.30",
        ):
            assert f"  {line}" in body

    def test_prints_capture_first_next_steps(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert "evalshift capture sync" in result.stdout


class TestInitStrongDefaults:
    """The generated config must be honest-by-default (see 2026-07-21 spec)."""

    def test_judge_criterion_is_symmetric(self, in_tmp: Path) -> None:
        # The judge sees anonymized A/B outputs — a criterion phrased in
        # TARGET/SOURCE terms is unanswerable and degenerates to position bias.
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        criterion = cfg.evaluators.llm_judge[0].criterion_prompt
        assert "TARGET" not in criterion
        assert "SOURCE" not in criterion
        assert "tie" in criterion.lower()

    def test_semantic_and_judge_are_advisory(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.blocking is False
        assert cfg.evaluators.llm_judge[0].blocking is False


class TestInitProvider:
    def test_default_is_gemini(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "gemini/gemini-embedding-001"
        assert "gemini" in cfg.defaults.source_model

    def test_openai_provider_writes_openai_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "openai"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "gpt-5.4-mini"
        assert cfg.evaluators.llm_judge[0].judge_model == "gpt-5.6-luna"
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "openai/text-embedding-3-small"
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "gemini-3.1" not in body

    def test_anthropic_provider_comments_out_semantic(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "claude-sonnet-5"
        assert cfg.evaluators.llm_judge[0].judge_model == "claude-opus-4-8"
        # Anthropic has no embedding endpoint — semantic ships commented out.
        assert cfg.evaluators.semantic is None
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# semantic:" in body

    def test_unknown_provider_rejected(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "grok"])
        assert result.exit_code != 0

    def test_every_provider_config_round_trips(self, in_tmp: Path) -> None:
        for provider in ("gemini", "openai", "anthropic"):
            for f in in_tmp.iterdir():
                if f.is_file():
                    f.unlink()
            result = runner.invoke(app, ["init", "--provider", provider, "--force"])
            assert result.exit_code == 0, f"{provider}: {result.stdout}"
            cfg = load_config(in_tmp / CONFIG_FILENAME)
            assert cfg.prompts[0].id == "replay"


class TestInitAgentWiring:
    """``init`` wires coding-agent instructions by default."""

    def test_writes_guide_and_creates_agents_md(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / AGENT_INSTRUCTIONS_FILENAME).is_file()
        host = in_tmp / DEFAULT_AGENT_CONTEXT_FILE
        assert host.is_file()
        assert POINTER_MARKER_BEGIN in host.read_text(encoding="utf-8")

    def test_wires_existing_claude_md(self, in_tmp: Path) -> None:
        (in_tmp / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
        runner.invoke(app, ["init"])
        body = (in_tmp / "CLAUDE.md").read_text(encoding="utf-8")
        assert "# rules" in body
        assert f"@./{AGENT_INSTRUCTIONS_FILENAME}" in body
        # No fallback AGENTS.md when a context file already exists.
        assert not (in_tmp / "AGENTS.md").exists()

    def test_no_wire_agents_skips_everything(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--no-wire-agents"])
        assert result.exit_code == 0, result.stdout
        assert not (in_tmp / AGENT_INSTRUCTIONS_FILENAME).exists()
        assert not (in_tmp / DEFAULT_AGENT_CONTEXT_FILE).exists()


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
        assert "version: 1" in (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")


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
        assert "EVALSHIFT_NONINTERACTIVE" in body
        assert "babaliauskas/evalshift-action@v0" in body
        assert "EVALSHIFT_TOKEN" in body
        assert "fail-on: regression" in body


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
