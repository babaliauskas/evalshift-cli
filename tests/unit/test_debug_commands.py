"""Tests for inspect/diff/replay debug commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.cli.main import app
from evalshift.evaluators.base import EvalRecord
from evalshift.runner.checkpoint import append_call, write_state
from evalshift.runner.models import Call, RunModels, RunState
from evalshift.traces.loader import TRACES_FILENAME

runner = CliRunner()


def _scaffold_debug_run(tmp_path: Path) -> tuple[Path, str]:
    run_id = "r_20260601_debug1"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text(
        '{"id": "ex1", "inputs": {}, "tags": ["billing"]}\n',
        encoding="utf-8",
    )
    write_state(
        run_dir,
        RunState(
            run_id=run_id,
            status="completed",
            config_hash="x",
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
            models=RunModels(source="src", target="tgt"),
            prompt_ids=["p"],
            suite_path=str(suite_path),
            total_evaluations=2,
            completed_evaluations=2,
        ),
    )
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="p",
            example_id="ex1",
            model_id="src",
            role="source",
            text="Refund order T-1032.",
            cost_usd=0.01,
            latency_ms=100,
        ),
    )
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="p",
            example_id="ex1",
            model_id="tgt",
            role="target",
            text="Refund order T-1023.",
            cost_usd=0.02,
            latency_ms=150,
        ),
    )
    score = EvalRecord(
        run_id=run_id,
        prompt_id="p",
        example_id="ex1",
        evaluator_name="tool_arguments.refund",
        source_score=1.0,
        target_score=0.0,
        delta=-1.0,
        explanation="ticket id drifted",
        metadata={"failure_categories": ["ARGUMENT_VALUE_DRIFT"]},
    )
    (run_dir / SCORES_FILENAME).write_text(score.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "migration_decision.json").write_text(
        json.dumps({"verdict": "fail"}),
        encoding="utf-8",
    )
    return tmp_path, run_id


def _write_agent_traces(run_dir: Path, run_id: str) -> None:
    source = {
        "run_id": run_id,
        "prompt_id": "p",
        "example_id": "ex1",
        "role": "source",
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 0,
                "timestamp": "2026-06-09T12:00:00Z",
                "metadata": {},
                "name": "check_refund_policy",
                "arguments": {},
            },
            {
                "type": "tool_call",
                "sequence_index": 1,
                "timestamp": "2026-06-09T12:00:01Z",
                "metadata": {},
                "name": "issue_refund",
                "arguments": {"ticket_id": "T-1032"},
            },
        ],
    }
    target = {
        "run_id": run_id,
        "prompt_id": "p",
        "example_id": "ex1",
        "role": "target",
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 0,
                "timestamp": "2026-06-09T12:00:00Z",
                "metadata": {},
                "name": "issue_refund",
                "arguments": {"ticket_id": "T-1023"},
            },
        ],
    }
    (run_dir / TRACES_FILENAME).write_text(
        json.dumps(source) + "\n" + json.dumps(target) + "\n",
        encoding="utf-8",
    )


class TestInspectCommand:
    def test_inspect_failed_lists_regressions(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["inspect", run_id, "--failed"])

        assert result.exit_code == 0, result.stdout
        assert "ex1" in result.stdout
        assert "ARGUMENT_VALUE_DRIFT" in result.stdout

    def test_inspect_case_shows_outputs_and_scores(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["inspect", "case", run_id, "ex1"])

        assert result.exit_code == 0, result.stdout
        assert "Refund order T-1032" in result.stdout
        assert "ticket id drifted" in result.stdout

    def test_inspect_case_shows_agent_trace_summary(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_agent_traces(run_dir, run_id)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["inspect", "case", run_id, "ex1"])

        assert result.exit_code == 0, result.stdout
        assert "agent trace" in result.stdout.lower()
        assert "check_refund_policy" in result.stdout


class TestDiffAndReplayCommands:
    def test_diff_case_prints_source_and_target(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["diff", "case", run_id, "ex1"])

        assert result.exit_code == 0, result.stdout
        assert "source" in result.stdout.lower()
        assert "T-1023" in result.stdout

    def test_diff_case_prefers_agent_trace_timeline(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_agent_traces(run_dir, run_id)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["diff", "case", run_id, "ex1"])

        assert result.exit_code == 0, result.stdout
        assert "agent trace diff" in result.stdout.lower()
        assert "ARGUMENT_VALUE_DRIFT" in result.stdout
        assert "check_refund_policy" in result.stdout

    def test_replay_case_prints_recorded_side(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["replay", "case", run_id, "ex1", "--model", "target"])

        assert result.exit_code == 0, result.stdout
        assert "Refund order T-1023" in result.stdout

    def test_replay_case_trace_prints_normalized_json(self, monkeypatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_debug_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_agent_traces(run_dir, run_id)
        monkeypatch.chdir(cwd)

        result = runner.invoke(
            app, ["replay", "case", run_id, "ex1", "--model", "target", "--trace"]
        )

        assert result.exit_code == 0, result.stdout
        assert '"role": "target"' in result.stdout
        assert "issue_refund" in result.stdout
