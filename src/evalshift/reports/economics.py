"""Per-role operational rollup, shared by the local report and the hosted bundle.

Lives outside ``reports/json.py`` so ``report.json`` and ``run_bundle.json.gz``
cannot drift: the bundle must never re-derive a figure the report already
computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalshift.runner.models import Call, RunState


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
    truncated_calls: int
    # Calls with empty text but output_tokens > 0, no error, a "stop" finish
    # reason, and no tool trace — usually a thinking-only response (see
    # `is_empty_output`). Not excluded from statistics (see
    # `TopRegression.target_empty_output`); this count just surfaces how
    # common the phenomenon is for this role.
    empty_output_calls: int
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


def build_economics(calls: list[Call]) -> PromptEconomics:
    """Split ``calls`` by role and roll each side up."""
    return PromptEconomics(
        source=role_economics([c for c in calls if c.role == "source"]),
        target=role_economics([c for c in calls if c.role == "target"]),
    )


def is_empty_output(call: Call) -> bool:
    """True when a call returned no visible text despite spending tokens.

    Usually a thinking-only response. Unlike `Call.truncated`, this is NOT
    excluded from the paired regression statistics — see `TopRegression`
    docstring — it's purely a report annotation.

    Tool-call responses (``call.trace is not None``) are excluded: the
    orchestrator's tool path sets ``text`` to the trace's final text (often
    empty for a pure tool-call turn), so treating that as "empty output"
    would flag nearly every agent-suite call. The ``finish_reason == "stop"``
    check mirrors the client-side warning gate in
    `evalshift.models.client._build_result` — a ``None`` or ``"length"``
    finish reason means we can't attribute the empty text to a genuine
    thinking-only stop, so it isn't counted either.
    """
    return (
        call.text == ""
        and call.output_tokens > 0
        and call.error is None
        and call.trace is None
        and call.finish_reason == "stop"
    )


def role_economics(calls: list[Call]) -> RoleEconomics:
    """Roll one role's calls up into cost, token and latency totals."""
    cached = sum(1 for c in calls if c.cached)
    failed = sum(1 for c in calls if c.error is not None)
    truncated = sum(1 for c in calls if c.truncated)
    empty_output = sum(1 for c in calls if is_empty_output(c))
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
        truncated_calls=truncated,
        empty_output_calls=empty_output,
        total_cost_usd=sum(c.cost_usd for c in calls),
        total_input_tokens=sum(c.input_tokens for c in calls),
        total_output_tokens=sum(c.output_tokens for c in calls),
        latency_ms_avg=float(avg_ms),
        latency_ms_p95=p95_ms,
    )


def methodology_notes(state: RunState) -> list[str]:
    """Return the statistical-contract notes shown in every report.

    When a model in the run no longer honours ``temperature``, one extra note
    per affected model is appended. Every paired test in the report assumes
    the only difference between the two arms is the model; a model that
    samples freely breaks that assumption, so it is stated alongside the
    contract it undermines rather than left for the reader to infer.
    """
    notes = [
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
    notes.extend(
        f"{model_id} does not honour temperature; its outputs are "
        "non-deterministic. Paired tests on this arm are under-powered — "
        "treat non-significant results with caution."
        for model_id in state.non_deterministic_models
    )
    return notes


def role_economics_to_dict(r: RoleEconomics) -> dict[str, Any]:
    """Serialise a :class:`RoleEconomics` to a JSON-ready dict."""
    return {
        "calls": r.calls,
        "live_calls": r.live_calls,
        "cached_calls": r.cached_calls,
        "failed_calls": r.failed_calls,
        "truncated_calls": r.truncated_calls,
        "empty_output_calls": r.empty_output_calls,
        "total_cost_usd": r.total_cost_usd,
        "total_input_tokens": r.total_input_tokens,
        "total_output_tokens": r.total_output_tokens,
        "latency_ms_avg": r.latency_ms_avg,
        "latency_ms_p95": r.latency_ms_p95,
    }


__all__ = [
    "PromptEconomics",
    "RoleEconomics",
    "build_economics",
    "is_empty_output",
    "methodology_notes",
    "role_economics",
    "role_economics_to_dict",
]
