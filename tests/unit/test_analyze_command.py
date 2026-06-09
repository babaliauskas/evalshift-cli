"""Tests for ``evalshift analyze <run-id>``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.cli.main import app
from evalshift.evaluators.base import EvalRecord
from evalshift.runner.checkpoint import write_state
from evalshift.runner.models import RunModels, RunState

runner = CliRunner()


def _scaffold(tmp_path: Path) -> tuple[Path, str]:
    """Create config + completed run + scores.jsonl. Returns (cwd, run_id)."""
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: hi
        evaluators:
          structural:
            - type: length
              max_chars: 100
        """,
        encoding="utf-8",
    )
    (tmp_path / "golden.jsonl").write_text(
        '{"id": "ex1", "inputs": {}, "tags": ["formal"]}\n'
        '{"id": "ex2", "inputs": {}, "tags": ["casual"]}\n',
        encoding="utf-8",
    )

    run_id = "r_20260601_aaaaaa"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id
    write_state(
        run_dir,
        RunState(
            run_id=run_id,
            status="completed",
            config_hash="x",
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
            models=RunModels(source="src", target="tgt"),
            prompt_ids=["greet"],
            suite_path=str(tmp_path / "golden.jsonl"),
            total_evaluations=4,
            completed_evaluations=4,
        ),
    )

    # Write 30 records — one per (example x evaluator x stuff). For a
    # detectable signal, we have to give enough examples per slice.
    rows = []
    for i in range(30):
        rows.append(
            EvalRecord(
                run_id=run_id,
                prompt_id="greet",
                example_id=f"ex{i}",
                evaluator_name="structural.length",
                source_score=1.0,
                target_score=0.5 - 0.01 * i,  # downward trend
                delta=(0.5 - 0.01 * i) - 1.0,
            ),
        )
    (run_dir / SCORES_FILENAME).write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path, run_id


class TestAnalyzeHappy:
    def test_writes_analysis_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        analysis_path = cwd / ".evalshift" / "runs" / run_id / ANALYSIS_FILENAME
        data = json.loads(analysis_path.read_text())
        assert data["run_id"] == run_id
        assert "comparisons" in data
        assert "aggregates" in data
        assert any(c["severity"] != "insufficient" for c in data["comparisons"])

    def test_writes_migration_decision_when_policy_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        config_path = cwd / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + "\n        migration_policy:\n"
            + "          max_overall_regression_rate: 0.01\n"
            + "          max_critical_regressions: 0\n"
            + "          min_equivalence_rate: 0.99\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["analyze", run_id])

        assert result.exit_code == 0, result.stdout
        decision_path = cwd / ".evalshift" / "runs" / run_id / "migration_decision.json"
        data = json.loads(decision_path.read_text(encoding="utf-8"))
        assert data["run_id"] == run_id
        assert data["verdict"] in {"fail", "conditional_pass", "pass", "inconclusive"}
        assert "budget_results" in data

    def test_summary_table_rendered(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        # Severity glyphs should reach stdout.
        assert "analysis" in result.stdout


class TestAnalyzeGate:
    """`--gate` is the CI hook: exit non-zero when a regression matches."""

    def test_gate_fails_when_severity_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # _scaffold builds a downward trend that produces high/critical
        # severity rows. critical+high should always trip the gate.
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["analyze", run_id, "--gate", "critical,high"])
        assert result.exit_code == 1, result.stdout
        assert "gate failed" in result.stdout.lower()
        # Artifact should still be written so CI can upload it.
        assert (cwd / ".evalshift" / "runs" / run_id / ANALYSIS_FILENAME).exists()

    def test_gate_passes_when_no_severity_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        # 'low' is a real severity but the synthetic trend is steep enough
        # that nothing should land at 'low' alone.
        result = runner.invoke(app, ["analyze", run_id, "--gate", "low"])
        # The trend produces high/critical comparisons, none at 'low' only,
        # so exit 0 is expected.
        analysis_path = cwd / ".evalshift" / "runs" / run_id / ANALYSIS_FILENAME
        data = json.loads(analysis_path.read_text())
        if any(c["severity"] == "low" for c in data["comparisons"]):
            assert result.exit_code == 1
        else:
            assert result.exit_code == 0, result.stdout

    def test_gate_rejects_unknown_severity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["analyze", run_id, "--gate", "fatal"])
        assert result.exit_code == 1
        assert "fatal" in result.stdout

    def test_no_gate_never_exits_one_on_severity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout

    def test_policy_gate_fails_on_policy_fail(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        config_path = cwd / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + "\n        migration_policy:\n"
            + "          max_overall_regression_rate: 0.0\n"
            + "          max_critical_regressions: 0\n"
            + "          min_equivalence_rate: 1.0\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["analyze", run_id, "--policy-gate"])

        assert result.exit_code == 1, result.stdout
        assert "policy gate failed" in result.stdout.lower()


class TestAnalyzeStepSummary:
    """Markdown summary is appended to $GITHUB_STEP_SUMMARY when set."""

    def test_writes_step_summary_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        summary_file = tmp_path / "step_summary.md"
        monkeypatch.chdir(cwd)
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        assert summary_file.exists()
        body = summary_file.read_text(encoding="utf-8")
        assert run_id in body
        # Markdown table header.
        assert "| severity |" in body
        assert "evaluator" in body

    def test_no_step_summary_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        monkeypatch.chdir(cwd)
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        # Should not crash and should not write any summary file.
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout


class TestAnalyzeErrors:
    def test_missing_run_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["analyze", "r_20260601_xxxxxx"])
        assert result.exit_code == 1

    def test_missing_scores_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        run_id = "r_20260601_yyyyyy"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["a"],
                suite_path="x.jsonl",
                total_evaluations=0,
                completed_evaluations=0,
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 1
        assert "scores.jsonl" in result.stdout
