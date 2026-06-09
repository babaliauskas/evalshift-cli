"""Tests for migration-policy verdict generation."""

from __future__ import annotations

import pytest

from evalshift.analysis.policy import (
    evaluate_migration_policy,
)
from evalshift.analysis.statistics import ComparisonResult
from evalshift.config.models import MigrationPolicy
from evalshift.evaluators.base import EvalRecord
from evalshift.runner.models import Call


def _comparison(
    *,
    severity: str,
    slice_name: str = "all",
    evaluator_name: str = "structural.length",
    prompt_id: str = "p",
    delta_avg_score: float = -0.1,
) -> ComparisonResult:
    return ComparisonResult(
        prompt_id=prompt_id,
        evaluator_name=evaluator_name,
        slice_name=slice_name,
        n=30,
        test="paired_t",
        statistic=1.0,
        p_value=0.01,
        p_value_corrected=0.01,
        effect_size=-0.8 if severity not in {"improved", "none"} else 0.2,
        effect_size_ci_low=-1.0,
        effect_size_ci_high=-0.2,
        delta_avg_score=delta_avg_score,
        severity=severity,  # type: ignore[arg-type]
        notes=[],
    )


def _record(
    *,
    example_id: str,
    delta: float,
    evaluator_name: str = "structural.length",
    category: str | None = None,
) -> EvalRecord:
    metadata = {"failure_categories": [category]} if category else {}
    return EvalRecord(
        run_id="r1",
        prompt_id="p",
        example_id=example_id,
        evaluator_name=evaluator_name,
        source_score=1.0,
        target_score=max(0.0, 1.0 + delta),
        delta=delta,
        metadata=metadata,
    )


def _call(
    *,
    example_id: str,
    role: str,
    cost_usd: float,
    latency_ms: int,
) -> Call:
    return Call(
        run_id="r1",
        prompt_id="p",
        example_id=example_id,
        model_id=role,
        role=role,  # type: ignore[arg-type]
        text="ok",
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


class TestEvaluateMigrationPolicy:
    def test_fail_when_blocking_regression_exceeds_budget(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_critical_regressions=0),
            comparisons=[_comparison(severity="critical", evaluator_name="tool_selection.routing")],
            records=[
                _record(
                    example_id="ex1",
                    delta=-1.0,
                    evaluator_name="tool_selection.routing",
                    category="TOOL_SELECTION_DRIFT",
                ),
            ],
            calls=[],
        )

        assert decision.verdict == "fail"
        assert decision.blocking_regressions
        assert decision.failure_categories[0].category == "TOOL_SELECTION_DRIFT"

    def test_conditional_pass_when_only_specific_slice_regresses(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                max_critical_regressions=1,
                min_equivalence_rate=0.0,
            ),
            comparisons=[
                _comparison(severity="high", slice_name="billing"),
                _comparison(severity="none", slice_name="support", delta_avg_score=0.0),
            ],
            records=[
                _record(example_id="ex1", delta=-0.8, category="SEMANTIC_REGRESSION"),
                _record(example_id="ex2", delta=0.0),
            ],
            calls=[],
        )

        assert decision.verdict == "conditional_pass"
        assert decision.slices["billing"].verdict == "fail"
        assert decision.slices["support"].verdict == "pass"
        assert "support" in decision.recommendations[0]

    def test_pass_when_all_budgets_are_met(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"ex{i}", delta=0.0) for i in range(20)],
            calls=[
                _call(example_id="ex1", role="source", cost_usd=1.0, latency_ms=100),
                _call(example_id="ex1", role="target", cost_usd=1.1, latency_ms=110),
            ],
        )

        assert decision.verdict == "pass"
        assert decision.overall.regression_rate == pytest.approx(0.0)
        assert all(result.passed for result in decision.budget_results)

    def test_inconclusive_when_all_comparisons_are_insufficient(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="insufficient", delta_avg_score=0.0)],
            records=[],
            calls=[],
        )

        assert decision.verdict == "inconclusive"
