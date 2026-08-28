"""Integration test for CLI-only imported agent traces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from evalshift.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.cli.main import app
from evalshift.reports.html import REPORT_HTML_FILENAME
from evalshift.runner.checkpoint import append_call, write_state
from evalshift.runner.models import Call, RunModels, RunState
from evalshift.traces.loader import TRACES_FILENAME

runner = CliRunner()


def _tool(name: str, index: int, args: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "type": "tool_call",
        "sequence_index": index,
        "timestamp": "2026-06-09T12:00:00Z",
        "metadata": {},
        "name": name,
        "arguments": args or {},
    }


def test_agent_trace_import_evaluate_analyze_report(monkeypatch, tmp_path: Path) -> None:
    run_id = "r_20260609_trace2"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text('{"id": "ex1", "inputs": {}, "tags": ["billing"]}\n', encoding="utf-8")
    (tmp_path / "evalshift.yaml").write_text(
        """
version: 1
prompts:
  - id: p
    detection: manual
    content: "refund"
evaluators:
  agent_trace:
    - name: trace_safety
      verification_tools: [check_refund_policy]
      dangerous_tools: [issue_refund]
""",
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
        run_dir, Call(run_id=run_id, prompt_id="p", example_id="ex1", model_id="src", role="source")
    )
    append_call(
        run_dir, Call(run_id=run_id, prompt_id="p", example_id="ex1", model_id="tgt", role="target")
    )
    source_trace = {
        "run_id": run_id,
        "prompt_id": "p",
        "example_id": "ex1",
        "role": "source",
        "events": [
            _tool("check_refund_policy", 0),
            _tool("issue_refund", 1, {"ticket_id": "T-1032"}),
        ],
    }
    target_trace = {
        "run_id": run_id,
        "prompt_id": "p",
        "example_id": "ex1",
        "role": "target",
        "events": [_tool("issue_refund", 0, {"ticket_id": "T-1023"})],
    }
    source_path = tmp_path / "source-traces.jsonl"
    target_path = tmp_path / "target-traces.jsonl"
    source_path.write_text(json.dumps(source_trace) + "\n", encoding="utf-8")
    target_path.write_text(json.dumps(target_trace) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    import_result = runner.invoke(
        app,
        ["traces", "import", run_id, "--source", str(source_path), "--target", str(target_path)],
    )
    evaluate_result = runner.invoke(app, ["evaluate", run_id])
    analyze_result = runner.invoke(app, ["analyze", run_id])
    report_result = runner.invoke(app, ["report", run_id])

    assert import_result.exit_code == 0, import_result.stdout
    assert evaluate_result.exit_code == 0, evaluate_result.stdout
    assert analyze_result.exit_code == 0, analyze_result.stdout
    assert report_result.exit_code == 0, report_result.stdout
    assert (run_dir / TRACES_FILENAME).exists()
    assert (run_dir / SCORES_FILENAME).exists()
    assert (run_dir / ANALYSIS_FILENAME).exists()
    assert (run_dir / REPORT_HTML_FILENAME).exists()
