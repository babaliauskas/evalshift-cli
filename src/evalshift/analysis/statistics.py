"""Statistical analysis of paired source/target scores.

Implements the contract from PDF §5.5 step-by-step:

1. Pair scores per slice; require ``n>=5`` to test, flag ``n<20``.
2. Test normality via Shapiro-Wilk on the deltas; route to a paired
   t-test (normal) or Wilcoxon signed-rank (non-normal).
3. Compute Cohen's d for paired samples (``mean(deltas)/std(deltas)``).
4. Compute a 95% CI on the effect size — analytical for the t-test,
   bootstrap for Wilcoxon.
5. After every comparison in the run is collected, apply the
   Benjamini-Hochberg correction (FDR=0.05) across all p-values to
   control false-discovery in the multi-test setting.
6. Classify severity from the *corrected* p-value + effect size.

Why these choices:

* **Paired tests** because each example is run on both models — the
  measurements are inherently paired and ignoring that throws away
  power.
* **Shapiro-Wilk gate** because the t-test is sensitive to non-normal
  deltas at small N; falling back to Wilcoxon (which only assumes
  symmetric distribution) is the standard MVP-grade safety net.
* **Cohen's d** as the effect size because it's interpretable in units
  of standard deviation and the field has an intuition for it
  (``|d|>0.8`` = "large").
* **Benjamini-Hochberg** instead of Bonferroni because BH controls FDR
  at the same level with substantially more power, especially when
  some comparisons are real (which is exactly the case in a real
  migration eval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from evalshift.analysis.slicing import SlicedScore

Severity = Literal[
    "critical",
    "high",
    "medium",
    "low",
    "improved",
    "none",
    "insufficient",
]

TestKind = Literal["paired_t", "wilcoxon", "skipped"]

MIN_N_FOR_TEST: int = 5
MIN_N_RELIABLE: int = 20
NORMALITY_ALPHA: float = 0.05
FDR_ALPHA: float = 0.05
BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_RNG_SEED: int = 0


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One (prompt_id, evaluator_name, slice_name) statistical verdict."""

    prompt_id: str
    evaluator_name: str
    slice_name: str
    n: int
    test: TestKind
    statistic: float
    p_value: float
    p_value_corrected: float
    effect_size: float
    effect_size_ci_low: float
    effect_size_ci_high: float
    delta_mean: float
    severity: Severity
    notes: list[str]


def analyze(
    *,
    sliced_by_slice: dict[str, list[SlicedScore]],
    rng: np.random.Generator | None = None,
) -> list[ComparisonResult]:
    """Run every comparison and apply BH correction across the whole set.

    Returns a list of :class:`ComparisonResult` ordered by descending
    severity then absolute effect size — i.e. the worst regressions
    bubble to the top.
    """
    rng_inst = rng or np.random.default_rng(DEFAULT_RNG_SEED)

    # Group within each slice by (prompt_id, evaluator_name) so we run
    # one test per (prompt, evaluator, slice).
    raw: list[ComparisonResult] = []
    for slice_name, sliced in sliced_by_slice.items():
        by_key: dict[tuple[str, str], list[SlicedScore]] = {}
        for s in sliced:
            by_key.setdefault((s.prompt_id, s.evaluator_name), []).append(s)
        for (prompt_id, evaluator_name), pairs in by_key.items():
            raw.append(
                _one_comparison(
                    prompt_id=prompt_id,
                    evaluator_name=evaluator_name,
                    slice_name=slice_name,
                    pairs=pairs,
                    rng=rng_inst,
                ),
            )

    # Apply Benjamini-Hochberg across every comparison that actually
    # ran a test (skipped comparisons keep p_value = 1.0 and don't
    # participate in the correction).
    testable_idx = [i for i, c in enumerate(raw) if c.test != "skipped"]
    p_raw = [raw[i].p_value for i in testable_idx]
    p_corrected_seq = _benjamini_hochberg(p_raw)
    corrected_lookup: dict[int, float] = dict(
        zip(testable_idx, p_corrected_seq, strict=True),
    )

    finalised: list[ComparisonResult] = []
    for i, c in enumerate(raw):
        if c.test == "skipped":
            finalised.append(c)
            continue
        cp = corrected_lookup[i]
        sev = _classify_severity(corrected_p=cp, d=c.effect_size, mean_delta=c.delta_mean, n=c.n)
        finalised.append(
            ComparisonResult(
                prompt_id=c.prompt_id,
                evaluator_name=c.evaluator_name,
                slice_name=c.slice_name,
                n=c.n,
                test=c.test,
                statistic=c.statistic,
                p_value=c.p_value,
                p_value_corrected=cp,
                effect_size=c.effect_size,
                effect_size_ci_low=c.effect_size_ci_low,
                effect_size_ci_high=c.effect_size_ci_high,
                delta_mean=c.delta_mean,
                severity=sev,
                notes=c.notes,
            ),
        )

    finalised.sort(key=_severity_sort_key)
    return finalised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_comparison(
    *,
    prompt_id: str,
    evaluator_name: str,
    slice_name: str,
    pairs: list[SlicedScore],
    rng: np.random.Generator,
) -> ComparisonResult:
    """Compute one comparison, deferring severity until BH is done."""
    n = len(pairs)
    deltas = np.array([p.delta for p in pairs], dtype=float)
    notes: list[str] = []

    if n < MIN_N_FOR_TEST:
        return ComparisonResult(
            prompt_id=prompt_id,
            evaluator_name=evaluator_name,
            slice_name=slice_name,
            n=n,
            test="skipped",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            effect_size=0.0,
            effect_size_ci_low=0.0,
            effect_size_ci_high=0.0,
            delta_mean=float(np.mean(deltas)) if n else 0.0,
            severity="insufficient",
            notes=[f"n={n} < {MIN_N_FOR_TEST}; no test run"],
        )
    if n < MIN_N_RELIABLE:
        notes.append(f"n={n} < {MIN_N_RELIABLE}; results uncertain")

    # Variance ≈ 0 (every delta identical, modulo float noise) → no
    # inference possible. Use a small tolerance to dodge catastrophic
    # cancellation that scipy already warns about.
    if float(np.std(deltas, ddof=1)) < 1e-9:
        return ComparisonResult(
            prompt_id=prompt_id,
            evaluator_name=evaluator_name,
            slice_name=slice_name,
            n=n,
            test="skipped",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            effect_size=0.0,
            effect_size_ci_low=0.0,
            effect_size_ci_high=0.0,
            delta_mean=float(np.mean(deltas)),
            severity="none",
            notes=[*notes, "zero variance — no test"],
        )

    # Pick the test based on a normality screen.
    use_t = _is_normal(deltas)

    if use_t:
        result = stats.ttest_rel(
            [p.target_score for p in pairs],
            [p.source_score for p in pairs],
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        d = _cohens_d_paired(deltas)
        ci_low, ci_high = _t_test_d_ci(d=d, n=n)
        test_kind: TestKind = "paired_t"
    else:
        result = stats.wilcoxon(deltas, zero_method="wilcox", correction=False)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        d = _cohens_d_paired(deltas)
        ci_low, ci_high = _bootstrap_d_ci(deltas, rng=rng)
        test_kind = "wilcoxon"
        notes.append("non-normal deltas; used Wilcoxon")

    return ComparisonResult(
        prompt_id=prompt_id,
        evaluator_name=evaluator_name,
        slice_name=slice_name,
        n=n,
        test=test_kind,
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value,  # placeholder; overwritten by BH
        effect_size=d,
        effect_size_ci_low=ci_low,
        effect_size_ci_high=ci_high,
        delta_mean=float(np.mean(deltas)),
        severity="none",  # placeholder; classified after BH
        notes=notes,
    )


def _is_normal(deltas: np.ndarray) -> bool:
    """Shapiro-Wilk normality screen.

    Skips for n>5000 where Shapiro is unreliable / overly powerful;
    in that regime a t-test is robust enough by central-limit-theorem
    grounds.
    """
    if deltas.size > 5000:
        return True
    try:
        result = stats.shapiro(deltas)
    except Exception:
        return False
    return float(result.pvalue) > NORMALITY_ALPHA


def _cohens_d_paired(deltas: np.ndarray) -> float:
    sd = float(np.std(deltas, ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(np.mean(deltas) / sd)


def _t_test_d_ci(*, d: float, n: int) -> tuple[float, float]:
    """Analytical 95% CI on Cohen's d (paired) using the SE approximation.

    Standard error: sqrt(1/n + d^2/(2n)). Times 1.96 for 95%.
    """
    se = float(np.sqrt(1.0 / n + (d * d) / (2.0 * n)))
    return d - 1.96 * se, d + 1.96 * se


def _bootstrap_d_ci(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI on Cohen's d (paired)."""
    n = deltas.size
    sample_ds = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        d = deltas[idx]
        sd = float(np.std(d, ddof=1))
        sample_ds[i] = float(np.mean(d)) / sd if sd != 0 else 0.0
    return float(np.percentile(sample_ds, 2.5)), float(np.percentile(sample_ds, 97.5))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Apply the BH FDR correction; returns adjusted p-values in input order.

    Mirrors ``statsmodels.stats.multitest.multipletests(..., method='fdr_bh')``
    semantics: each adjusted p-value is the smallest q such that the
    BH cutoff would still call the corresponding raw p significant.
    """
    n = len(p_values)
    if n == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted_ranked = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity (right-to-left running min) and clamp to 1.
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted_ranked
    return [float(x) for x in out]


def _classify_severity(
    *,
    corrected_p: float,
    d: float,
    mean_delta: float,
    n: int,
) -> Severity:
    """Map (corrected p, effect size, direction, n) to severity per PDF §5.5."""
    if n < MIN_N_FOR_TEST:
        return "insufficient"
    if corrected_p >= FDR_ALPHA:
        return "none"
    abs_d = abs(d)
    if mean_delta > 0:
        return "improved"
    # mean_delta < 0 → regression
    if corrected_p < 0.01 and abs_d > 0.8:
        return "critical"
    if abs_d > 0.5:
        return "high"
    if abs_d > 0.2:
        return "medium"
    return "low"


_SEVERITY_RANK: dict[Severity, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "improved": 4,
    "none": 5,
    "insufficient": 6,
}


def _severity_sort_key(c: ComparisonResult) -> tuple[int, float, str, str, str]:
    return (
        _SEVERITY_RANK[c.severity],
        -abs(c.effect_size),
        c.prompt_id,
        c.evaluator_name,
        c.slice_name,
    )


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "FDR_ALPHA",
    "MIN_N_FOR_TEST",
    "MIN_N_RELIABLE",
    "NORMALITY_ALPHA",
    "ComparisonResult",
    "Severity",
    "TestKind",
    "analyze",
]
