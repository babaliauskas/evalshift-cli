"""Tests for ``evalshift traces import``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from evalshift.cli.main import app
from evalshift.runner.checkpoint import append_call, write_state
from evalshift.runner.models import Call, RunModels, RunState
from evalshift.traces.loader import TRACES_FILENAME, load_traces_jsonl

runner = CliRunner()


def _event(name: str, index: int) -> dict[str, object]:
    return {
        "type": "tool_call",
        "sequence_index": index,
        "timestamp": "2026-06-09T12:00:00Z",
        "metadata": {},
        "name": name,
        "arguments": {"ticket_id": "T-1032"},
        "call_id": f"call_{name}",
    }


def _trace(*, run_id: str, role: str, example_id: str = "ex1") -> dict[str, object]:
    return {
        "run_id": run_id,
        "prompt_id": "p",
        "example_id": example_id,
        "role": role,
        "events": [_event("check_refund_policy", 0), _event("issue_refund", 1)],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scaffold_completed_run(tmp_path: Path) -> tuple[Path, str]:
    run_id = "r_20260609_trace1"
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
            started_at=datetime(2026, 6, 9, tzinfo=UTC),
            models=RunModels(source="src", target="tgt"),
            prompt_ids=["p"],
            suite_path=str(suite_path),
            total_evaluations=2,
            completed_evaluations=2,
        ),
    )
    append_call(
        run_dir,
        Call(run_id=run_id, prompt_id="p", example_id="ex1", model_id="src", role="source"),
    )
    append_call(
        run_dir,
        Call(run_id=run_id, prompt_id="p", example_id="ex1", model_id="tgt", role="target"),
    )
    return tmp_path, run_id


def test_traces_import_writes_normalized_run_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cwd, run_id = _scaffold_completed_run(tmp_path)
    source_path = tmp_path / "source-traces.jsonl"
    target_path = tmp_path / "target-traces.jsonl"
    _write_jsonl(source_path, [_trace(run_id=run_id, role="source")])
    _write_jsonl(target_path, [_trace(run_id=run_id, role="target")])
    monkeypatch.chdir(cwd)

    result = runner.invoke(
        app,
        [
            "traces",
            "import",
            run_id,
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "source: 1 traces" in result.stdout
    assert "target: 1 traces" in result.stdout
    output_path = cwd / ".evalshift" / "runs" / run_id / TRACES_FILENAME
    traces = load_traces_jsonl(output_path)
    assert [(t.role, t.example_id) for t in traces] == [("source", "ex1"), ("target", "ex1")]


def test_traces_import_rejects_unknown_example(monkeypatch, tmp_path: Path) -> None:
    cwd, run_id = _scaffold_completed_run(tmp_path)
    source_path = tmp_path / "source-traces.jsonl"
    target_path = tmp_path / "target-traces.jsonl"
    _write_jsonl(source_path, [_trace(run_id=run_id, role="source", example_id="missing")])
    _write_jsonl(target_path, [_trace(run_id=run_id, role="target")])
    monkeypatch.chdir(cwd)

    result = runner.invoke(
        app,
        [
            "traces",
            "import",
            run_id,
            "--source",
            str(source_path),
            "--target",
            str(target_path),
        ],
    )

    assert result.exit_code == 1
    assert "unknown prompt/example" in result.stdout


def test_traces_import_strict_rejects_missing_pair(monkeypatch, tmp_path: Path) -> None:
    cwd, run_id = _scaffold_completed_run(tmp_path)
    source_path = tmp_path / "source-traces.jsonl"
    target_path = tmp_path / "target-traces.jsonl"
    _write_jsonl(source_path, [_trace(run_id=run_id, role="source")])
    _write_jsonl(target_path, [])
    monkeypatch.chdir(cwd)

    result = runner.invoke(
        app,
        [
            "traces",
            "import",
            run_id,
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "missing trace pairs" in result.stdout
