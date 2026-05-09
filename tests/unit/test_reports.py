"""Tests for the reports package + the ``evalshift report`` command."""

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
from evalshift.reports.html import REPORT_HTML_FILENAME, render_html, write_html
from evalshift.reports.json import REPORT_JSON_FILENAME, build_report_payload
from evalshift.runner.checkpoint import append_call, write_state
from evalshift.runner.models import Call, RunModels, RunState

runner = CliRunner()


def _scaffold_full_run(tmp_path: Path) -> tuple[Path, str]:
    """Scaffold a run dir with raw.jsonl, scores.jsonl, and analysis.json."""
    run_id = "r_20260601_aaaaaa"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id

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
            total_evaluations=4,
            completed_evaluations=4,
        ),
    )

    # raw.jsonl with two pairs.
    for ex_id, src_text, tgt_text in (
        ("ex1", "Hello there!", "Hi"),
        ("ex2", "Greetings, friend.", "yo"),
    ):
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id=ex_id,
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text=src_text,
                cost_usd=0.0001,
                latency_ms=100,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id=ex_id,
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text=tgt_text,
                cost_usd=0.0002,
                latency_ms=120,
            ),
        )

    # scores.jsonl
    rows = [
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            evaluator_name="structural.length",
            source_score=1.0,
            target_score=0.5,
            delta=-0.5,
        ),
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex2",
            evaluator_name="structural.length",
            source_score=1.0,
            target_score=0.0,
            delta=-1.0,
        ),
    ]
    (run_dir / SCORES_FILENAME).write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n",
        encoding="utf-8",
    )

    # analysis.json
    analysis = {
        "run_id": run_id,
        "models": {
            "source": "gemini/gemini-2.5-flash",
            "target": "gemini/gemini-2.5-pro",
        },
        "n_examples": 2,
        "n_records": 2,
        "slices": ["all"],
        "aggregates": {},
        "comparisons": [
            {
                "prompt_id": "greet",
                "evaluator_name": "structural.length",
                "slice_name": "all",
                "n": 2,
                "test": "skipped",
                "statistic": 0.0,
                "p_value": 1.0,
                "p_value_corrected": 1.0,
                "effect_size": 0.0,
                "effect_size_ci_low": 0.0,
                "effect_size_ci_high": 0.0,
                "delta_mean": -0.75,
                "severity": "insufficient",
                "notes": ["n=2 < 5; no test run"],
            },
        ],
    }
    (run_dir / ANALYSIS_FILENAME).write_text(json.dumps(analysis), encoding="utf-8")
    return tmp_path, run_id


# ---------------------------------------------------------------------------
# build_report_payload
# ---------------------------------------------------------------------------


class TestReportPayload:
    def test_assembles_data(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        assert payload.run_id == run_id
        assert payload.source_model == "gemini/gemini-2.5-flash"
        assert payload.n_examples == 2
        assert payload.n_calls == 4
        assert len(payload.prompt_sections) == 1
        # Top regressions should have the negative-delta records.
        section = payload.prompt_sections[0]
        assert section.prompt_id == "greet"
        assert all(tr.delta < 0 for tr in section.top_regressions)

    def test_missing_analysis_raises(self, tmp_path: Path) -> None:
        run_id = "r_20260601_xxxxxx"
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
        with pytest.raises(FileNotFoundError, match="analysis"):
            build_report_payload(run_dir)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


class TestHtmlRender:
    def test_renders_with_inlined_css(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)
        assert html.startswith("<!DOCTYPE html>")
        # CSS must be inlined (we ship no external assets).
        assert "<style>" in html
        # Run id and key metadata reach the document.
        assert run_id in html
        assert "gemini/gemini-2.5-flash" in html
        # No external scripts.
        assert "<script" not in html.lower()

    def test_writes_file(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir)
        out = write_html(payload, run_dir)
        assert out.name == REPORT_HTML_FILENAME
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestReportCommand:
    def test_writes_html_and_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        # The command needs an evalshift.yaml in the cwd.
        (cwd / "evalshift.yaml").write_text(
            "prompts:\n  - {id: greet, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["report", run_id])
        assert result.exit_code == 0, result.stdout
        run_dir = cwd / ".evalshift" / "runs" / run_id
        assert (run_dir / REPORT_HTML_FILENAME).exists()
        assert (run_dir / REPORT_JSON_FILENAME).exists()

    def test_missing_analysis_exits_one_with_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
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
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["report", run_id])
        assert result.exit_code == 1
        assert "evaluate" in result.stdout
        assert "analyze" in result.stdout


# ---------------------------------------------------------------------------
# v0.2 — trace rendering
# ---------------------------------------------------------------------------


from evalshift.evaluators.tool_models import ToolCall, ToolTrace  # noqa: E402


class TestTraceRendering:
    def _scaffold_agent_run(self, tmp_path: Path) -> tuple[Path, str]:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Replace the calls in raw.jsonl with tool-bearing ones so the
        # trace plumbing shows up end-to-end.
        from evalshift.runner.checkpoint import append_call
        from evalshift.runner.models import Call

        (run_dir / "raw.jsonl").unlink()
        for ex_id in ("ex1", "ex2"):
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-flash",
                    role="source",
                    text="",
                    cost_usd=0.0,
                    latency_ms=1,
                    trace=ToolTrace(
                        calls=[
                            ToolCall(
                                tool_name="search_db",
                                arguments={"q": "ACME"},
                                sequence_index=0,
                            ),
                        ],
                    ),
                ),
            )
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-pro",
                    role="target",
                    text="",
                    cost_usd=0.0,
                    latency_ms=1,
                    trace=ToolTrace(calls=[]),  # target dropped the call
                ),
            )
        return cwd, run_id

    def test_payload_attaches_traces_to_top_regressions(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_agent_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        section = payload.prompt_sections[0]
        # The fixture has negative-delta scores → top regressions populated.
        if section.top_regressions:
            tr = section.top_regressions[0]
            assert tr.source_trace is not None
            assert tr.target_trace is not None
            assert tr.source_trace.tool_names == ["search_db"]
            assert tr.target_trace.tool_names == []
        # Section flag set.
        assert section.has_tool_traces is True

    def test_html_renders_trace_diff(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_agent_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)
        # Trace structure markers in the rendered HTML.
        if any(s.has_tool_traces for s in payload.prompt_sections) and any(
            tr.source_trace for s in payload.prompt_sections for tr in s.top_regressions
        ):
            assert "Source trace" in html
            assert "Target trace" in html
            assert "search_db" in html
        # No JS still — even with the new trace section.
        assert "<script" not in html.lower()
