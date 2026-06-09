"""Migration-policy evaluation.

This module turns statistical comparisons plus per-example records into the
product-level answer EvalShift exists to provide: whether the target model is
safe to migrate to under the configured regression budget.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from evalshift.analysis.statistics import ComparisonResult
from evalshift.config.models import MigrationPolicy, SliceMigrationPolicy
from evalshift.evaluators.base import EvalRecord
from evalshift.runner.models import Call

MigrationVerdict = Literal["pass", "conditional_pass", "fail", "inconclusive"]

_REGRESSION_SEVERITIES = frozenset({"critical", "high", "medium", "low"})
_BLOCKING_SEVERITIES = frozenset({"critical", "high"})
_TOOL_ARGUMENT_PREFIX = "tool_arguments."


@dataclass(frozen=True, slots=True)
class PolicyMetricSummary:
    """Aggregate migration outcome rates for a scope."""

    n_records: int
    improved_rate: float
    equivalent_rate: float
    regression_rate: float
    critical_regressions: int
    tool_argument_drift_rate: float
    cost_increase_rate: float
    latency_increase_rate: float


@dataclass(frozen=True, slots=True)
class BudgetResult:
    """One migration-policy budget check."""

    name: str
    observed: float
    allowed: float
    passed: bool
    scope: str = "overall"


@dataclass(frozen=True, slots=True)
class SliceDecision:
    """Verdict and metrics for one analysis slice."""

    name: str
    verdict: MigrationVerdict
    metrics: PolicyMetricSummary
    budget_results: list[BudgetResult]


@dataclass(frozen=True, slots=True)
class BlockingRegression:
    """A statistical regression that can block migration."""

    prompt_id: str
    evaluator_name: str
    slice_name: str
    severity: str
    delta_avg_score: float
    effect_size: float


@dataclass(frozen=True, slots=True)
class FailureCategoryCount:
    """Count of a deterministic failure category from evaluator metadata."""

    category: str
    count: int


@dataclass(frozen=True, slots=True)
class MigrationDecision:
    """Top-level migration decision persisted as JSON."""

    run_id: str
    source_model: str
    target_model: str
    verdict: MigrationVerdict
    overall: PolicyMetricSummary
    slices: dict[str, SliceDecision]
    budget_results: list[BudgetResult]
    blocking_regressions: list[BlockingRegression]
    failure_categories: list[FailureCategoryCount]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def evaluate_migration_policy(
    *,
    run_id: str,
    source_model: str,
    target_model: str,
    policy: MigrationPolicy,
    comparisons: list[ComparisonResult],
    records: list[EvalRecord],
    calls: list[Call],
) -> MigrationDecision:
    """Evaluate a run against the configured migration policy."""
    overall = _metrics(records=records, calls=calls)
    overall_budgets = _budget_results(policy=policy, metrics=overall, scope="overall")
    blocking = _blocking_regressions(comparisons)
    categories = _failure_categories(records)

    slice_names = sorted({c.slice_name for c in comparisons if c.slice_name != "all"})
    slice_decisions: dict[str, SliceDecision] = {}
    for name in slice_names:
        scoped_records = _records_for_slice(records, comparisons, name)
        scoped_metrics = _metrics(records=scoped_records, calls=calls)
        slice_policy = _slice_policy(policy, policy.slices.get(name))
        budgets = _budget_results(policy=slice_policy, metrics=scoped_metrics, scope=name)
        slice_decisions[name] = SliceDecision(
            name=name,
            verdict=_verdict_for(
                comparisons=[c for c in comparisons if c.slice_name == name],
                budgets=budgets,
            ),
            metrics=scoped_metrics,
            budget_results=budgets,
        )

    overall_comparisons = [c for c in comparisons if c.slice_name == "all"]
    verdict = _verdict_for(comparisons=overall_comparisons, budgets=overall_budgets)
    if verdict == "pass":
        failed_slices = [s for s in slice_decisions.values() if s.verdict == "fail"]
        if failed_slices:
            verdict = "conditional_pass"
    elif not overall_comparisons and any(
        s.verdict in {"fail", "conditional_pass"} for s in slice_decisions.values()
    ):
        verdict = "conditional_pass"

    return MigrationDecision(
        run_id=run_id,
        source_model=source_model,
        target_model=target_model,
        verdict=verdict,
        overall=overall,
        slices=slice_decisions,
        budget_results=overall_budgets,
        blocking_regressions=blocking,
        failure_categories=categories,
        recommendations=_recommendations(verdict=verdict, slices=slice_decisions),
    )


def _metrics(*, records: list[EvalRecord], calls: list[Call]) -> PolicyMetricSummary:
    scored = [r for r in records if r.error is None]
    n = len(scored)
    regressions = [r for r in scored if r.delta < 0]
    improvements = [r for r in scored if r.delta > 0]
    equivalents = [r for r in scored if r.delta == 0]
    critical = [r for r in scored if str(r.metadata.get("severity", "")) == "critical"]
    tool_argument_records = [
        r for r in scored if r.evaluator_name.startswith(_TOOL_ARGUMENT_PREFIX)
    ]
    tool_argument_drift = [r for r in tool_argument_records if r.delta < 0]

    return PolicyMetricSummary(
        n_records=n,
        improved_rate=_rate(len(improvements), n),
        equivalent_rate=_rate(len(equivalents), n),
        regression_rate=_rate(len(regressions), n),
        critical_regressions=len(critical),
        tool_argument_drift_rate=_rate(len(tool_argument_drift), len(tool_argument_records)),
        cost_increase_rate=_relative_increase(calls, field="cost_usd"),
        latency_increase_rate=_relative_increase(calls, field="latency_ms"),
    )


def _budget_results(
    *,
    policy: MigrationPolicy,
    metrics: PolicyMetricSummary,
    scope: str,
) -> list[BudgetResult]:
    return [
        BudgetResult(
            name="max_overall_regression_rate",
            observed=metrics.regression_rate,
            allowed=policy.max_overall_regression_rate,
            passed=metrics.regression_rate <= policy.max_overall_regression_rate,
            scope=scope,
        ),
        BudgetResult(
            name="max_critical_regressions",
            observed=float(metrics.critical_regressions),
            allowed=float(policy.max_critical_regressions),
            passed=metrics.critical_regressions <= policy.max_critical_regressions,
            scope=scope,
        ),
        BudgetResult(
            name="min_equivalence_rate",
            observed=metrics.equivalent_rate,
            allowed=policy.min_equivalence_rate,
            passed=metrics.equivalent_rate >= policy.min_equivalence_rate,
            scope=scope,
        ),
        BudgetResult(
            name="max_tool_argument_drift",
            observed=metrics.tool_argument_drift_rate,
            allowed=policy.max_tool_argument_drift,
            passed=metrics.tool_argument_drift_rate <= policy.max_tool_argument_drift,
            scope=scope,
        ),
        BudgetResult(
            name="max_cost_increase",
            observed=metrics.cost_increase_rate,
            allowed=policy.max_cost_increase,
            passed=metrics.cost_increase_rate <= policy.max_cost_increase,
            scope=scope,
        ),
        BudgetResult(
            name="max_latency_increase",
            observed=metrics.latency_increase_rate,
            allowed=policy.max_latency_increase,
            passed=metrics.latency_increase_rate <= policy.max_latency_increase,
            scope=scope,
        ),
    ]


def _verdict_for(
    *,
    comparisons: list[ComparisonResult],
    budgets: list[BudgetResult],
) -> MigrationVerdict:
    if comparisons and all(c.severity == "insufficient" for c in comparisons):
        return "inconclusive"
    if any(not b.passed for b in budgets):
        return "fail"
    if any(c.severity in _BLOCKING_SEVERITIES for c in comparisons):
        return "fail"
    if any(c.severity in _REGRESSION_SEVERITIES for c in comparisons):
        return "conditional_pass"
    return "pass"


def _blocking_regressions(comparisons: list[ComparisonResult]) -> list[BlockingRegression]:
    return [
        BlockingRegression(
            prompt_id=c.prompt_id,
            evaluator_name=c.evaluator_name,
            slice_name=c.slice_name,
            severity=c.severity,
            delta_avg_score=c.delta_avg_score,
            effect_size=c.effect_size,
        )
        for c in comparisons
        if c.severity in _BLOCKING_SEVERITIES
    ]


def _failure_categories(records: list[EvalRecord]) -> list[FailureCategoryCount]:
    counter: Counter[str] = Counter()
    for r in records:
        raw = r.metadata.get("failure_categories", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item:
                    counter[item] += 1
    return [
        FailureCategoryCount(category=category, count=count)
        for category, count in counter.most_common()
    ]


def _records_for_slice(
    records: list[EvalRecord],
    comparisons: list[ComparisonResult],
    slice_name: str,
) -> list[EvalRecord]:
    evaluator_names = {c.evaluator_name for c in comparisons if c.slice_name == slice_name}
    if not evaluator_names:
        return []
    return [r for r in records if r.evaluator_name in evaluator_names]


def _slice_policy(
    base: MigrationPolicy,
    override: SliceMigrationPolicy | None,
) -> MigrationPolicy:
    if override is None:
        return base
    return MigrationPolicy(
        max_overall_regression_rate=(
            base.max_overall_regression_rate
            if override.max_overall_regression_rate is None
            else override.max_overall_regression_rate
        ),
        max_critical_regressions=(
            base.max_critical_regressions
            if override.max_critical_regressions is None
            else override.max_critical_regressions
        ),
        min_equivalence_rate=(
            base.min_equivalence_rate
            if override.min_equivalence_rate is None
            else override.min_equivalence_rate
        ),
        max_tool_argument_drift=(
            base.max_tool_argument_drift
            if override.max_tool_argument_drift is None
            else override.max_tool_argument_drift
        ),
        max_cost_increase=(
            base.max_cost_increase
            if override.max_cost_increase is None
            else override.max_cost_increase
        ),
        max_latency_increase=(
            base.max_latency_increase
            if override.max_latency_increase is None
            else override.max_latency_increase
        ),
    )


def _recommendations(
    *,
    verdict: MigrationVerdict,
    slices: dict[str, SliceDecision],
) -> list[str]:
    safe = sorted(name for name, decision in slices.items() if decision.verdict == "pass")
    unsafe = sorted(name for name, decision in slices.items() if decision.verdict == "fail")
    if verdict == "pass":
        return ["Safe to migrate under the configured policy."]
    if verdict == "inconclusive":
        return ["Collect more examples before making a migration decision."]
    if safe and unsafe:
        return [
            "Safe slices: "
            + ", ".join(safe)
            + ". Keep "
            + ", ".join(unsafe)
            + " on the source model.",
        ]
    return ["Do not migrate globally under the configured policy."]


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _relative_increase(calls: list[Call], *, field: str) -> float:
    source = [float(getattr(c, field)) for c in calls if c.role == "source" and c.error is None]
    target = [float(getattr(c, field)) for c in calls if c.role == "target" and c.error is None]
    if not source or not target:
        return 0.0
    source_avg = sum(source) / len(source)
    target_avg = sum(target) / len(target)
    if source_avg <= 0:
        return 0.0 if target_avg <= 0 else 1.0
    return max(0.0, (target_avg - source_avg) / source_avg)


__all__ = [
    "BudgetResult",
    "FailureCategoryCount",
    "MigrationDecision",
    "MigrationVerdict",
    "PolicyMetricSummary",
    "SliceDecision",
    "evaluate_migration_policy",
]
