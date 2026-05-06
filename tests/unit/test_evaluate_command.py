"""Tests for the ``aimigrate evaluate <run-id>`` command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aimigrate.cli.commands.evaluate import SCORES_FILENAME
from aimigrate.cli.main import app
from aimigrate.runner.checkpoint import append_call, write_state
from aimigrate.runner.models import Call, RunModels, RunState

runner = CliRunner()


def _write_config(tmp_path: Path, with_judge: bool = False) -> Path:
    judge_block = ""
    if with_judge:
        judge_block = """
              llm_judge:
                - criterion_name: brevity
                  criterion_prompt: which is more concise?
                  judge_model: gemini-2.5-flash"""
    cfg_yaml = f"""
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hi {{name}}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
        evaluators:
          structural:
            - type: length
              min_chars: 1
              max_chars: 100
            - type: regex
              pattern: ".+"
{judge_block}
    """
    path = tmp_path / "aimigrate.yaml"
    path.write_text(cfg_yaml, encoding="utf-8")
    return path


def _scaffold_run(tmp_path: Path, completed: bool = True) -> str:
    """Create a minimal completed run dir and return its run_id."""
    runs_base = tmp_path / ".aimigrate" / "runs"
    run_id = "r_20260601_aaaaaa"
    run_dir = runs_base / run_id
    state = RunState(
        run_id=run_id,
        status="completed" if completed else "in_progress",
        config_hash="x",
        started_at=datetime(2026, 6, 1, tzinfo=UTC),
        models=RunModels(
            source="gemini/gemini-2.5-flash",
            target="gemini/gemini-2.5-pro",
        ),
        prompt_ids=["greet"],
        suite_path="./golden.jsonl",
        total_evaluations=4,
        completed_evaluations=4,
    )
    write_state(run_dir, state)
    for ex_id in ("ex1", "ex2"):
        for role, model_id, text in (
            ("source", "gemini/gemini-2.5-flash", "Hi Alex (short)"),
            ("target", "gemini/gemini-2.5-pro", "Hi Alex! How are you?"),
        ):
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id=model_id,
                    role=role,  # type: ignore[arg-type]
                    text=text,
                ),
            )
    return run_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEvaluateHappy:
    def test_writes_scores_jsonl(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _write_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout

        scores_path = tmp_path / ".aimigrate" / "runs" / run_id / SCORES_FILENAME
        rows = [
            json.loads(line)
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # 2 examples x 2 evaluators (length + regex) = 4 rows.
        assert len(rows) == 4
        for row in rows:
            assert row["run_id"] == run_id
            assert row["evaluator_name"].startswith("structural.")
            assert "delta" in row

    def test_pairs_only_complete_pairs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        # Add an orphan source-only call — should be ignored.
        run_dir = tmp_path / ".aimigrate" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_orphan",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="orphan",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        scores_path = run_dir / SCORES_FILENAME
        rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
        # Still 4 (2 paired examples x 2 evaluators); the orphan's not paired.
        assert len(rows) == 4

    def test_upstream_failed_calls_recorded_with_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        run_id = "r_20260601_bbbbbb"
        run_dir = tmp_path / ".aimigrate" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(
                    source="gemini/gemini-2.5-flash",
                    target="gemini/gemini-2.5-pro",
                ),
                prompt_ids=["greet"],
                suite_path="./golden.jsonl",
                total_evaluations=2,
                completed_evaluations=2,
            ),
        )
        # Source failed; target succeeded.
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                error="auth failed",
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="Hi",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        rows = [
            json.loads(line)
            for line in (run_dir / SCORES_FILENAME).read_text().splitlines()
            if line.strip()
        ]
        assert all(r["error"] for r in rows)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestEvaluateErrors:
    def test_unknown_run_id_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", "r_20260601_nonexis"])
        assert result.exit_code == 1

    def test_no_evaluators_configured_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Config without any evaluator entries.
        (tmp_path / "aimigrate.yaml").write_text(
            "prompts:\n  - {id: greet, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 1
        assert "no evaluators" in result.stdout.lower()

    def test_no_pairs_in_raw_jsonl(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _write_config(tmp_path)
        # Create a run with state but no pairs (only source calls).
        run_id = "r_20260601_cccccc"
        run_dir = tmp_path / ".aimigrate" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["greet"],
                suite_path="x.jsonl",
                total_evaluations=2,
                completed_evaluations=2,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="a",
                role="source",
                text="hi",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 1
        assert "no (source, target) pairs" in result.stdout
