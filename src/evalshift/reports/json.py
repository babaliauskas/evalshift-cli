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

from evalshift.cli.commands.analyze import ANALYSIS_FILENAME, MIGRATION_DECISION_FILENAME
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.tool_models import ToolTrace
from evalshift.runner.checkpoint import iter_calls, read_state
from evalshift.runner.models import Call, RunState
from evalshift.suite.loader import SuiteError, load_jsonl
from evalshift.suite.models import Suite

REPORT_JSON_FILENAME: str = "report.json"


@dataclass(frozen=True, slots=True)
class ToolDiff:
    """One tool-call diff item for a top regression."""

    kind: str
    tool_name: str
    message: str
    source_arguments: dict[str, Any] | None = None
    target_arguments: dict[str, Any] | None = None


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
    tool_diffs: list[ToolDiff] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RoleEconomics:
    """Per-prompt × role rollup of operational stats from raw.jsonl.

    `latency_ms_avg` / `_p95` are computed only over live (non-cached,
    non-error) calls — cache hits replay text from disk so their
    `latency_ms` is 0 and would skew the averages downward.
    """

    calls: int
    live_calls: int
    cached_calls: int
    failed_calls: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    latency_ms_avg: float
    latency_ms_p95: float


@dataclass(frozen=True, slots=True)
class PromptEconomics:
    """Source vs target operational stats for a single prompt."""

    source: RoleEconomics
    target: RoleEconomics


@dataclass(frozen=True, slots=True)
class ExampleRow:
    """One row in the per-example breakdown table.

    Captures only deltas (target − source) — the report is a migration
    diff, so absolute values per role are surfaced upstream in
    ``PromptEconomics``.
    """

    example_id: str
    tags: list[str]
    delta_latency_ms: int
    delta_cost_usd: float
    worst_delta_score: float | None
    tool_match: bool | None  # None when no tool evaluator scored this example


@dataclass(frozen=True, slots=True)
class PromptSection:
    """Per-prompt slice of the report payload."""

    prompt_id: str
    aggregate_rows: list[dict[str, Any]]  # serialised SliceAggregate per evaluator
    slice_rows: list[dict[str, Any]]  # serialised ComparisonResult rows
    top_regressions: list[TopRegression]
    economics: PromptEconomics
    example_rows: list[ExampleRow]
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
    migration_decision: dict[str, Any] | None = None
    prompt_sections: list[PromptSection] = field(default_factory=list)
    methodology_notes: list[str] = field(default_factory=list)


def build_report_payload(
    run_dir: Path,
    *,
    tool_evaluator_names: frozenset[str] = frozenset(),
) -> ReportData:
    """Assemble :class:`ReportData` from the artefacts in ``run_dir``.

    Args:
        run_dir: The completed run directory.
        tool_evaluator_names: Names of evaluators that score tool-call
            correctness (configured under ``tool_selection`` /
            ``tool_arguments`` / ``tool_trace_structure`` in
            evalshift.yaml). Used to populate the per-example
            ``tool_match`` flag. Empty set is fine — every example's
            ``tool_match`` will then be ``None``.

    Raises:
        FileNotFoundError: If any required artefact is missing. The
            caller (``evalshift report``) is expected to surface a
            helpful error message.
    """
    state = read_state(run_dir)
    analysis = _read_analysis(run_dir)
    migration_decision = _read_migration_decision(run_dir)
    scores = _read_scores(run_dir)
    calls = list(iter_calls(run_dir))
    suite = _load_suite(state.suite_path)

    cached = sum(1 for c in calls if c.cached)
    failed = sum(1 for c in calls if c.error is not None)
    cost = sum(c.cost_usd for c in calls)

    summary = _build_executive_summary(analysis)
    sections = _build_prompt_sections(
        analysis,
        scores,
        calls,
        suite,
        tool_evaluator_names,
    )

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
        migration_decision=migration_decision,
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


def _read_migration_decision(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MIGRATION_DECISION_FILENAME
    if not path.exists():
        return None
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{MIGRATION_DECISION_FILENAME} must contain a JSON object")
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


def _load_suite(suite_path: str) -> Suite:
    """Load the suite referenced by RunState; tolerate failures.

    The report is read-only; if the suite has been moved/edited since the
    run, fall back to an empty Suite so we still render economics +
    statistics. Tag/expected-tools data simply won't be available.
    """
    try:
        return load_jsonl(Path(suite_path))
    except SuiteError, FileNotFoundError, OSError:
        return Suite()


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
    suite: Suite,
    tool_evaluator_names: frozenset[str],
) -> list[PromptSection]:
    by_prompt_calls: dict[tuple[str, str], dict[str, Call]] = {}
    for c in calls:
        by_prompt_calls.setdefault((c.prompt_id, c.example_id), {})[c.role] = c

    by_prompt_records: dict[str, list[EvalRecord]] = {}
    for rec in scores:
        by_prompt_records.setdefault(rec.prompt_id, []).append(rec)

    tags_by_example_id: dict[str, list[str]] = {ex.id: list(ex.tags) for ex in suite.examples}

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
                    tool_diffs=_build_tool_diffs(
                        src.trace if src else None, tgt.trace if tgt else None
                    ),
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
        prompt_calls = [c for c in calls if c.prompt_id == prompt_id]
        example_rows = _build_example_rows(
            prompt_id=prompt_id,
            calls=prompt_calls,
            records=prompt_records,
            tags_by_example_id=tags_by_example_id,
            tool_evaluator_names=tool_evaluator_names,
        )
        sections.append(
            PromptSection(
                prompt_id=prompt_id,
                aggregate_rows=aggregates_all,
                slice_rows=slice_rows,
                top_regressions=top,
                economics=_build_economics(prompt_calls),
                example_rows=example_rows,
                has_tool_traces=has_traces,
            ),
        )
    return sections


def _build_example_rows(
    *,
    prompt_id: str,
    calls: list[Call],
    records: list[EvalRecord],
    tags_by_example_id: dict[str, list[str]],
    tool_evaluator_names: frozenset[str],
) -> list[ExampleRow]:
    """Assemble one ExampleRow per example for a single prompt.

    Δ values are target − source. Latency is reported as 0 when either
    side cached (cached calls carry latency_ms = 0 by convention) so
    the figure stays meaningful only on live × live pairs.
    """
    sides_by_example: dict[str, dict[str, Call]] = {}
    for c in calls:
        sides_by_example.setdefault(c.example_id, {})[c.role] = c

    records_by_example: dict[str, list[EvalRecord]] = {}
    for r in records:
        records_by_example.setdefault(r.example_id, []).append(r)

    rows: list[ExampleRow] = []
    for example_id in sides_by_example:
        sides = sides_by_example[example_id]
        src = sides.get("source")
        tgt = sides.get("target")
        if src is None or tgt is None:
            continue  # incomplete pair; skip rather than misreport
        delta_lat = 0 if src.cached or tgt.cached else tgt.latency_ms - src.latency_ms
        delta_cost = tgt.cost_usd - src.cost_usd

        ex_records = records_by_example.get(example_id, [])
        scored = [r for r in ex_records if r.error is None]
        worst = min((r.delta for r in scored), default=None)

        tool_records = [r for r in scored if r.evaluator_name in tool_evaluator_names]
        if tool_records:
            tool_match: bool | None = all(r.target_score >= 1.0 for r in tool_records)
        else:
            tool_match = None

        rows.append(
            ExampleRow(
                example_id=example_id,
                tags=tags_by_example_id.get(example_id, []),
                delta_latency_ms=delta_lat,
                delta_cost_usd=delta_cost,
                worst_delta_score=worst,
                tool_match=tool_match,
            ),
        )

    # Worst regression first (most-negative delta score), tiebreak on
    # most-expensive (highest +delta cost), then example_id for stability.
    rows.sort(
        key=lambda r: (
            r.worst_delta_score if r.worst_delta_score is not None else 0.0,
            -r.delta_cost_usd,
            r.example_id,
        ),
    )
    return rows


def _build_tool_diffs(
    source_trace: ToolTrace | None,
    target_trace: ToolTrace | None,
) -> list[ToolDiff]:
    if source_trace is None and target_trace is None:
        return []
    source_calls = source_trace.calls if source_trace else []
    target_calls = target_trace.calls if target_trace else []
    diffs: list[ToolDiff] = []

    max_len = max(len(source_calls), len(target_calls))
    for index in range(max_len):
        src = source_calls[index] if index < len(source_calls) else None
        tgt = target_calls[index] if index < len(target_calls) else None
        if src is None and tgt is not None:
            diffs.append(
                ToolDiff(
                    kind="extra_tool",
                    tool_name=tgt.tool_name,
                    message=f"Target added {tgt.tool_name} at position {index + 1}.",
                    target_arguments=tgt.arguments,
                ),
            )
            continue
        if src is not None and tgt is None:
            diffs.append(
                ToolDiff(
                    kind="missing_tool",
                    tool_name=src.tool_name,
                    message=f"Target omitted {src.tool_name} at position {index + 1}.",
                    source_arguments=src.arguments,
                ),
            )
            continue
        if src is None or tgt is None:
            continue
        if src.tool_name != tgt.tool_name:
            diffs.append(
                ToolDiff(
                    kind="tool_order_or_selection",
                    tool_name=tgt.tool_name,
                    message=(
                        f"Position {index + 1}: source called {src.tool_name}, "
                        f"target called {tgt.tool_name}."
                    ),
                    source_arguments=src.arguments,
                    target_arguments=tgt.arguments,
                ),
            )
        elif src.arguments != tgt.arguments:
            diffs.append(
                ToolDiff(
                    kind="argument_drift",
                    tool_name=src.tool_name,
                    message=f"Arguments changed for {src.tool_name} at position {index + 1}.",
                    source_arguments=src.arguments,
                    target_arguments=tgt.arguments,
                ),
            )
    return diffs


def _build_economics(calls: list[Call]) -> PromptEconomics:
    return PromptEconomics(
        source=_role_economics([c for c in calls if c.role == "source"]),
        target=_role_economics([c for c in calls if c.role == "target"]),
    )


def _role_economics(calls: list[Call]) -> RoleEconomics:
    cached = sum(1 for c in calls if c.cached)
    failed = sum(1 for c in calls if c.error is not None)
    live_latencies = [c.latency_ms for c in calls if not c.cached and c.error is None]
    if live_latencies:
        sorted_l = sorted(live_latencies)
        avg_ms = sum(sorted_l) / len(sorted_l)
        # Nearest-rank p95.
        p95_idx = max(0, round(0.95 * len(sorted_l)) - 1)
        p95_ms = float(sorted_l[p95_idx])
    else:
        avg_ms = 0.0
        p95_ms = 0.0
    return RoleEconomics(
        calls=len(calls),
        live_calls=len(live_latencies),
        cached_calls=cached,
        failed_calls=failed,
        total_cost_usd=sum(c.cost_usd for c in calls),
        total_input_tokens=sum(c.input_tokens for c in calls),
        total_output_tokens=sum(c.output_tokens for c in calls),
        latency_ms_avg=float(avg_ms),
        latency_ms_p95=p95_ms,
    )


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
        "migration_decision": report.migration_decision,
        "prompt_sections": [
            {
                "prompt_id": ps.prompt_id,
                "aggregate_rows": ps.aggregate_rows,
                "slice_rows": ps.slice_rows,
                "has_tool_traces": ps.has_tool_traces,
                "economics": {
                    "source": _role_economics_to_dict(ps.economics.source),
                    "target": _role_economics_to_dict(ps.economics.target),
                },
                "example_rows": [
                    {
                        "example_id": er.example_id,
                        "tags": er.tags,
                        "delta_latency_ms": er.delta_latency_ms,
                        "delta_cost_usd": er.delta_cost_usd,
                        "worst_delta_score": er.worst_delta_score,
                        "tool_match": er.tool_match,
                    }
                    for er in ps.example_rows
                ],
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
                        "tool_diffs": [
                            {
                                "kind": d.kind,
                                "tool_name": d.tool_name,
                                "message": d.message,
                                "source_arguments": d.source_arguments,
                                "target_arguments": d.target_arguments,
                            }
                            for d in tr.tool_diffs
                        ],
                    }
                    for tr in ps.top_regressions
                ],
            }
            for ps in report.prompt_sections
        ],
        "methodology_notes": report.methodology_notes,
    }


def _role_economics_to_dict(r: RoleEconomics) -> dict[str, Any]:
    return {
        "calls": r.calls,
        "live_calls": r.live_calls,
        "cached_calls": r.cached_calls,
        "failed_calls": r.failed_calls,
        "total_cost_usd": r.total_cost_usd,
        "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens,
        "latency_ms_avg": r.latency_ms_avg,
        "latency_ms_p95": r.latency_ms_p95,
    }


__all__ = [
    "REPORT_JSON_FILENAME",
    "PromptSection",
    "ReportData",
    "ToolDiff",
    "TopRegression",
    "build_report_payload",
    "write_report_json",
]
