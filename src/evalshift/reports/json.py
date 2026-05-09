"""Report-data assembly: stitches together everything Phase 7 needs.

Reads the artefacts produced by earlier phases:

* ``state.json`` — run metadata (Phase 4).
* ``raw.jsonl`` — per-call records (Phase 4).
* ``scores.jsonl`` — per (pair, evaluator) records (Phase 5).
* ``analysis.json`` — statistical comparisons (Phase 6).

…and produces a single :class:`ReportData` payload that the HTML
renderer consumes. Also writes a ``report.json`` artefact alongside it
so external tools can read the report data without parsing HTML.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalshift.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.tool_models import ToolTrace
from evalshift.runner.checkpoint import iter_calls, read_state
from evalshift.runner.models import Call, RunState

REPORT_JSON_FILENAME: str = "report.json"


@dataclass(frozen=True, slots=True)
class TopRegression:
    """One concrete (prompt, example, evaluator) regression worth showing."""

    prompt_id: str
    example_id: str
    evaluator_name: str
    delta: float
    source_text: str
    target_text: str
    # v0.2 — populated when the regression is on a tool evaluator. The
    # HTML report renders these as a side-by-side trace diff in place
    # of the source/target text panes.
    source_trace: ToolTrace | None = None
    target_trace: ToolTrace | None = None


@dataclass(frozen=True, slots=True)
class PromptSection:
    """Per-prompt slice of the report payload."""

    prompt_id: str
    aggregate_rows: list[dict[str, Any]]  # serialised SliceAggregate per evaluator
    slice_rows: list[dict[str, Any]]  # serialised ComparisonResult rows
    top_regressions: list[TopRegression]
    # v0.2 — when at least one Call has a populated trace, render a
    # "Tool Trace Comparison" section with example-level diffs.
    has_tool_traces: bool = False


@dataclass(frozen=True, slots=True)
class ReportData:
    """Everything the HTML template needs in one place."""

    run_id: str
    started_at: str
    source_model: str
    target_model: str
    suite_path: str
    n_examples: int
    n_calls: int
    cached_calls: int
    failed_calls: int
    total_cost_usd: float

    executive_summary: list[dict[str, Any]] = field(default_factory=list)
    prompt_sections: list[PromptSection] = field(default_factory=list)
    methodology_notes: list[str] = field(default_factory=list)


def build_report_payload(run_dir: Path) -> ReportData:
    """Assemble :class:`ReportData` from the artefacts in ``run_dir``.

    Raises:
        FileNotFoundError: If any required artefact is missing. The
            caller (``evalshift report``) is expected to surface a
            helpful error message.
    """
    state = read_state(run_dir)
    analysis = _read_analysis(run_dir)
    scores = _read_scores(run_dir)
    calls = list(iter_calls(run_dir))

    cached = sum(1 for c in calls if c.cached)
    failed = sum(1 for c in calls if c.error is not None)
    cost = sum(c.cost_usd for c in calls)

    summary = _build_executive_summary(analysis)
    sections = _build_prompt_sections(analysis, scores, calls)

    return ReportData(
        run_id=state.run_id,
        started_at=state.started_at.isoformat(),
        source_model=state.models.source,
        target_model=state.models.target,
        suite_path=state.suite_path,
        n_examples=_n_examples(calls),
        n_calls=len(calls),
        cached_calls=cached,
        failed_calls=failed,
        total_cost_usd=cost,
        executive_summary=summary,
        prompt_sections=sections,
        methodology_notes=_methodology_notes(state),
    )


def write_report_json(report: ReportData, run_dir: Path) -> Path:
    """Persist the report payload as ``report.json``."""
    out_path = run_dir / REPORT_JSON_FILENAME
    out_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_analysis(run_dir: Path) -> dict[str, Any]:
    path = run_dir / ANALYSIS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {ANALYSIS_FILENAME} in {run_dir}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{ANALYSIS_FILENAME} must contain a JSON object")
    return data


def _read_scores(run_dir: Path) -> list[EvalRecord]:
    path = run_dir / SCORES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {SCORES_FILENAME} in {run_dir}")
    out: list[EvalRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            out.append(EvalRecord.model_validate_json(text))
    return out


def _n_examples(calls: list[Call]) -> int:
    return len({c.example_id for c in calls})


def _build_executive_summary(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per prompt summarising the worst non-all-slice severity."""
    by_prompt: dict[str, dict[str, Any]] = {}
    severity_rank = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "improved": 4,
        "none": 5,
        "insufficient": 6,
    }
    for c in analysis["comparisons"]:
        prompt = c["prompt_id"]
        rank = severity_rank.get(c["severity"], 99)
        existing = by_prompt.get(prompt)
        if existing is None or rank < severity_rank.get(existing["severity"], 99):
            by_prompt[prompt] = c
    return [by_prompt[k] for k in sorted(by_prompt)]


def _build_prompt_sections(
    analysis: dict[str, Any],
    scores: list[EvalRecord],
    calls: list[Call],
) -> list[PromptSection]:
    by_prompt_calls: dict[tuple[str, str], dict[str, Call]] = {}
    for c in calls:
        by_prompt_calls.setdefault((c.prompt_id, c.example_id), {})[c.role] = c

    by_prompt_records: dict[str, list[EvalRecord]] = {}
    for rec in scores:
        by_prompt_records.setdefault(rec.prompt_id, []).append(rec)

    sections: list[PromptSection] = []
    for prompt_id in sorted(by_prompt_records):
        prompt_records = by_prompt_records[prompt_id]
        prompt_comparisons = [c for c in analysis["comparisons"] if c["prompt_id"] == prompt_id]
        # Aggregate rows: pick the implicit "all" slice rows for this prompt.
        aggregates_all = [c for c in prompt_comparisons if c["slice_name"] == "all"]
        # Slice rows: every non-all comparison for this prompt.
        slice_rows = [c for c in prompt_comparisons if c["slice_name"] != "all"]
        # Top regressions: the 5 records with the most-negative delta.
        ranked = sorted(prompt_records, key=lambda r: r.delta)[:5]
        top: list[TopRegression] = []
        for r in ranked:
            if r.delta >= 0 or r.error is not None:
                continue
            calls_for_pair = by_prompt_calls.get((r.prompt_id, r.example_id), {})
            src = calls_for_pair.get("source")
            tgt = calls_for_pair.get("target")
            top.append(
                TopRegression(
                    prompt_id=r.prompt_id,
                    example_id=r.example_id,
                    evaluator_name=r.evaluator_name,
                    delta=r.delta,
                    source_text=src.text if src else "",
                    target_text=tgt.text if tgt else "",
                    source_trace=src.trace if src else None,
                    target_trace=tgt.trace if tgt else None,
                ),
            )
        # v0.2 — does this prompt have any traces at all? Drives the
        # "Tool Trace Comparison" subsection in the HTML template.
        has_traces = any(
            (sides.get("source") and sides["source"].trace is not None)
            or (sides.get("target") and sides["target"].trace is not None)
            for (pid, _ex_id), sides in by_prompt_calls.items()
            if pid == prompt_id
        )
        sections.append(
            PromptSection(
                prompt_id=prompt_id,
                aggregate_rows=aggregates_all,
                slice_rows=slice_rows,
                top_regressions=top,
                has_tool_traces=has_traces,
            ),
        )
    return sections


def _methodology_notes(state: RunState) -> list[str]:
    return [
        f"Source model: {state.models.source}",
        f"Target model: {state.models.target}",
        "Statistical tests: paired t-test (Shapiro-Wilk passed) or Wilcoxon "
        "signed-rank (otherwise).",
        "Effect size: Cohen's d for paired samples, with 95% CI "
        "(analytical for t-tests; bootstrap for Wilcoxon, 2000 resamples).",
        "Multi-test correction: Benjamini-Hochberg across every "
        "(prompt x evaluator x slice) p-value at FDR=0.05.",
        "Severity classification uses the *corrected* p-value plus |Cohen's d|.",
    ]


def _to_jsonable(report: ReportData) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "started_at": report.started_at,
        "source_model": report.source_model,
        "target_model": report.target_model,
        "suite_path": report.suite_path,
        "n_examples": report.n_examples,
        "n_calls": report.n_calls,
        "cached_calls": report.cached_calls,
        "failed_calls": report.failed_calls,
        "total_cost_usd": report.total_cost_usd,
        "executive_summary": report.executive_summary,
        "prompt_sections": [
            {
                "prompt_id": ps.prompt_id,
                "aggregate_rows": ps.aggregate_rows,
                "slice_rows": ps.slice_rows,
                "has_tool_traces": ps.has_tool_traces,
                "top_regressions": [
                    {
                        "prompt_id": tr.prompt_id,
                        "example_id": tr.example_id,
                        "evaluator_name": tr.evaluator_name,
                        "delta": tr.delta,
                        "source_text": tr.source_text,
                        "target_text": tr.target_text,
                        "source_trace": (tr.source_trace.model_dump() if tr.source_trace else None),
                        "target_trace": (tr.target_trace.model_dump() if tr.target_trace else None),
                    }
                    for tr in ps.top_regressions
                ],
            }
            for ps in report.prompt_sections
        ],
        "methodology_notes": report.methodology_notes,
    }


__all__ = [
    "REPORT_JSON_FILENAME",
    "PromptSection",
    "ReportData",
    "TopRegression",
    "build_report_payload",
    "write_report_json",
]
