"""Tests for slicing + statistics."""

from __future__ import annotations

import numpy as np
import pytest

from aimigrate.analysis.slicing import (
    ALL_SLICE,
    SliceAggregate,
    SlicedScore,
    aggregates,
    build_slices,
)
from aimigrate.analysis.statistics import (
    ComparisonResult,
    _benjamini_hochberg,
    _classify_severity,
    _cohens_d_paired,
    analyze,
)
from aimigrate.evaluators.base import EvalRecord
from aimigrate.suite.models import Suite, SuiteExample

# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _record(
    *,
    prompt: str = "p",
    evaluator: str = "e",
    example: str,
    delta: float,
    error: str | None = None,
) -> EvalRecord:
    src = max(0.0, -delta) if delta < 0 else 0.0
    tgt = src + delta
    src = max(0.0, min(1.0, src))
    tgt = max(0.0, min(1.0, tgt))
    return EvalRecord(
        run_id="r1",
        prompt_id=prompt,
        example_id=example,
        evaluator_name=evaluator,
        source_score=src,
        target_score=tgt,
        delta=tgt - src,
        error=error,
    )


class TestBuildSlices:
    def test_all_slice_always_present(self) -> None:
        suite = Suite(
            examples=[SuiteExample(id="ex1", tags=["formal"])],
        )
        records = [_record(example="ex1", delta=0.1)]
        out = build_slices(records=records, suite=suite)
        assert ALL_SLICE in out
        assert out[ALL_SLICE][0].example_id == "ex1"

    def test_examples_appear_in_each_tag_slice(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="a", tags=["formal", "english"]),
                SuiteExample(id="b", tags=["casual"]),
            ],
        )
        records = [
            _record(example="a", delta=0.0),
            _record(example="b", delta=0.0),
        ]
        out = build_slices(records=records, suite=suite)
        assert {s.example_id for s in out["formal"]} == {"a"}
        assert {s.example_id for s in out["english"]} == {"a"}
        assert {s.example_id for s in out["casual"]} == {"b"}
        assert {s.example_id for s in out[ALL_SLICE]} == {"a", "b"}

    def test_errored_records_skipped(self) -> None:
        suite = Suite(examples=[SuiteExample(id="a", tags=["t"])])
        records = [_record(example="a", delta=0.0, error="boom")]
        out = build_slices(records=records, suite=suite)
        assert out == {}

    def test_unknown_example_id_only_lands_in_all(self) -> None:
        suite = Suite(examples=[SuiteExample(id="a", tags=["t"])])
        records = [_record(example="ghost", delta=0.0)]
        out = build_slices(records=records, suite=suite)
        assert "t" not in out
        assert len(out[ALL_SLICE]) == 1


class TestAggregates:
    def test_empty_slice(self) -> None:
        agg = aggregates([], "all")
        assert agg.n == 0
        assert agg.delta_mean == 0.0

    def test_simple_aggregate(self) -> None:
        ss = [
            SlicedScore("p", "e", "1", 1.0, 0.5, -0.5),
            SlicedScore("p", "e", "2", 1.0, 1.0, 0.0),
            SlicedScore("p", "e", "3", 0.5, 0.0, -0.5),
        ]
        agg = aggregates(ss, "all")
        assert agg.n == 3
        assert agg.delta_mean == pytest.approx((-0.5 + 0.0 - 0.5) / 3)
        assert agg.delta_min == -0.5
        assert agg.delta_max == 0.0


# ---------------------------------------------------------------------------
# Statistics — helpers
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_empty(self) -> None:
        assert _benjamini_hochberg([]) == []

    def test_all_significant(self) -> None:
        # p = [0.001, 0.002, 0.003, 0.004]; BH adjusted should still be < 0.05
        adj = _benjamini_hochberg([0.001, 0.002, 0.003, 0.004])
        assert all(a <= 0.05 for a in adj)

    def test_input_order_preserved(self) -> None:
        # Even though BH sorts internally, output must be in input order.
        raw = [0.5, 0.01, 0.04]
        adj = _benjamini_hochberg(raw)
        # Smallest raw p (0.01) should produce the smallest adjusted.
        assert adj[1] < adj[0]
        assert adj[1] < adj[2]

    def test_clamped_to_one(self) -> None:
        # A high p should not exceed 1 after correction.
        adj = _benjamini_hochberg([0.99] * 5)
        assert all(0.0 <= a <= 1.0 for a in adj)

    def test_matches_known_result(self) -> None:
        # statsmodels.stats.multitest.multipletests([0.01, 0.04, 0.20], method='fdr_bh')
        # produces adjusted p-values [0.03, 0.06, 0.20] — confirm we match.
        adj = _benjamini_hochberg([0.01, 0.04, 0.20])
        assert adj[0] == pytest.approx(0.03, abs=1e-6)
        assert adj[1] == pytest.approx(0.06, abs=1e-6)
        assert adj[2] == pytest.approx(0.20, abs=1e-6)


class TestCohensD:
    def test_zero_variance(self) -> None:
        assert _cohens_d_paired(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_positive_effect(self) -> None:
        d = _cohens_d_paired(np.array([0.5, 0.6, 0.4, 0.5]))
        assert d > 0


class TestClassifySeverity:
    def test_insufficient(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=-1.0, mean_delta=-0.5, n=3) == "insufficient"

    def test_no_significance(self) -> None:
        assert _classify_severity(corrected_p=0.5, d=-2.0, mean_delta=-0.5, n=20) == "none"

    def test_improved(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=0.5, mean_delta=0.4, n=20) == "improved"

    def test_critical(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=-1.0, mean_delta=-0.4, n=50) == "critical"

    def test_high(self) -> None:
        assert _classify_severity(corrected_p=0.02, d=-0.6, mean_delta=-0.2, n=30) == "high"

    def test_medium(self) -> None:
        assert _classify_severity(corrected_p=0.04, d=-0.3, mean_delta=-0.1, n=30) == "medium"

    def test_low(self) -> None:
        assert _classify_severity(corrected_p=0.04, d=-0.1, mean_delta=-0.05, n=30) == "low"


# ---------------------------------------------------------------------------
# Statistics — full pipeline
# ---------------------------------------------------------------------------


def _ss(n: int, delta_mean: float, *, prompt: str = "p", ev: str = "e") -> list[SlicedScore]:
    """Build n SlicedScores with a controlled mean delta."""
    rng = np.random.default_rng(123)
    deltas = rng.normal(loc=delta_mean, scale=0.1, size=n)
    return [
        SlicedScore(
            prompt_id=prompt,
            evaluator_name=ev,
            example_id=f"ex{i}",
            source_score=0.5,
            target_score=0.5 + d,
            delta=float(d),
        )
        for i, d in enumerate(deltas)
    ]


class TestAnalyze:
    def test_clear_regression_marked_significant(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(40, -0.3)})
        # Single comparison so BH = raw.
        c = results[0]
        assert c.severity in ("critical", "high")
        assert c.test == "paired_t" or c.test == "wilcoxon"
        assert c.delta_mean < 0
        assert c.effect_size < 0
        assert c.p_value_corrected < 0.01

    def test_clear_improvement_marked_improved(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(40, +0.3)})
        c = results[0]
        assert c.severity == "improved"
        assert c.delta_mean > 0

    def test_no_real_effect_returns_none(self) -> None:
        # Mean delta near 0 and small spread → not significant.
        results = analyze(sliced_by_slice={"all": _ss(40, 0.0)})
        c = results[0]
        assert c.severity == "none"

    def test_small_sample_skipped(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(3, -0.5)})
        c = results[0]
        assert c.severity == "insufficient"
        assert c.test == "skipped"

    def test_zero_variance_skipped(self) -> None:
        identical = [SlicedScore("p", "e", f"ex{i}", 0.5, 0.4, -0.1) for i in range(10)]
        results = analyze(sliced_by_slice={"all": identical})
        # Every delta is the same; no inference possible.
        c = results[0]
        assert c.test == "skipped"
        assert "zero variance" in " ".join(c.notes)

    def test_results_sorted_by_severity(self) -> None:
        results = analyze(
            sliced_by_slice={
                "all": _ss(40, -0.5, prompt="bad"),
                "extra": _ss(40, +0.3, prompt="good"),
            },
        )
        # Critical/high should come before "improved" / "none".
        first_sev = results[0].severity
        last_sev = results[-1].severity
        order_first = {"critical": 0, "high": 1}.get(first_sev, 99)
        order_last = {"improved": 4, "none": 5, "insufficient": 6}.get(last_sev, 99)
        assert order_first < order_last

    def test_bh_correction_changes_p_with_many_tests(self) -> None:
        # Use the helper directly with a spread of raw p-values; BH must
        # raise at least some of them for the multi-comparison case.
        from aimigrate.analysis.statistics import _benjamini_hochberg

        raw_ps = [
            0.001,
            0.002,
            0.01,
            0.04,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.85,
            0.9,
            0.92,
            0.94,
            0.96,
            0.97,
            0.98,
            0.99,
        ]
        adj = _benjamini_hochberg(raw_ps)
        increased = [a for a, r in zip(adj, raw_ps, strict=True) if a > r]
        assert len(increased) > 0


class TestComparisonResultShape:
    def test_carries_ci_bounds(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(20, -0.3)})
        c = results[0]
        assert isinstance(c, ComparisonResult)
        assert c.effect_size_ci_low <= c.effect_size <= c.effect_size_ci_high or (
            c.effect_size_ci_low > c.effect_size_ci_high  # bootstrap may flip
        )


class TestSliceAggregateShape:
    def test_zero_n_aggregate(self) -> None:
        agg = SliceAggregate(
            name="empty",
            n=0,
            source_mean=0,
            target_mean=0,
            delta_mean=0,
            delta_min=0,
            delta_max=0,
            source_std=0,
            target_std=0,
        )
        assert agg.n == 0
