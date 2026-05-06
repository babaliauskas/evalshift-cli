# Methodology

How AIMigrate decides whether a model swap caused a regression.

## The shape of the problem

For each (prompt × example) pair, AIMigrate produces two outputs:
`source_output` (from the old model) and `target_output` (from the
new one). Each evaluator scores both halves on a 0–1 scale, and the
quantity we care about is the per-example **delta**:

```
delta_i = target_score_i - source_score_i
```

Negative deltas mean the target lost ground; positive deltas mean it
gained. A "regression" is a slice where the deltas are systematically
negative across many examples.

## Per-comparison testing

For each `(prompt, evaluator, slice)` triple we run one paired test
on the deltas:

1. **Sample-size guards.**
   * `n < 5`: no test runs; severity is `insufficient`.
   * `5 ≤ n < 20`: test runs but the result carries an "uncertain"
     note in the report.

2. **Normality screen.** Shapiro-Wilk on the deltas. If the screen
   passes (`p > 0.05`), we run a paired t-test. Otherwise we fall
   back to a Wilcoxon signed-rank test, which only assumes
   symmetric distributions. (For `n > 5000`, Shapiro-Wilk is
   over-powered, so we skip the screen and use the t-test by CLT
   grounds.)

3. **Effect size.** Cohen's d for paired samples:
   ```
   d = mean(deltas) / std(deltas, ddof=1)
   ```
   Sign of `d` tracks the sign of `mean(deltas)` (negative = regression).

4. **95% CI on the effect size.** Analytical (`d ± 1.96 × SE`)
   when the t-test ran; percentile bootstrap (2000 resamples) when
   Wilcoxon ran.

## Multi-test correction

The whole point of slicing is to find regressions in *some* subset
of the data, but every additional comparison inflates the family-wise
false-discovery rate. We control it with **Benjamini–Hochberg** at
FDR α = 0.05 across every `(prompt × evaluator × slice)` p-value
in the run.

We pick BH over Bonferroni because BH controls the same error rate
with substantially more power, especially when several comparisons
are real (the typical case in a real eval suite).

The implementation matches `statsmodels.stats.multitest.multipletests(method='fdr_bh')`
exactly — verified against a known-result test in the suite.

## Severity classification

After BH correction, each comparison gets a severity tag based on the
*adjusted* p-value and the absolute effect size:

| Severity        | Rule                                                  |
| --------------- | ----------------------------------------------------- |
| `critical`      | adjusted p < 0.01 AND \|d\| > 0.8 AND mean(delta) < 0 |
| `high`          | adjusted p < 0.05 AND \|d\| > 0.5 AND mean(delta) < 0 |
| `medium`        | adjusted p < 0.05 AND \|d\| > 0.2 AND mean(delta) < 0 |
| `low`           | adjusted p < 0.05 AND mean(delta) < 0                 |
| `improved`      | adjusted p < 0.05 AND mean(delta) > 0                 |
| `none`          | adjusted p ≥ 0.05                                     |
| `insufficient`  | n < 5                                                 |

## What we always report

Every comparison row in `analysis.json` and `report.html` carries:

* raw p-value
* BH-adjusted p-value
* Cohen's d and its 95% CI
* sample size
* test used (`paired_t`, `wilcoxon`, or `skipped`)
* severity classification

You will never see a naked "regression detected" without the numbers
backing it up.

## Limitations to be aware of

* **Score scales must be comparable across (source, target).** Most
  evaluators are inherently per-output (structural, judge), so
  this is fine. The semantic evaluator deliberately frames source
  as 1.0 to make deltas interpretable, but that means the absolute
  similarity is the signal — interpret with care.

* **Wilcoxon's assumption is symmetry.** If your delta distribution
  is asymmetric (e.g. small wins, occasional huge losses), Wilcoxon's
  null can be off. The test still flags real shifts but the p-values
  may be approximate.

* **BH controls expected FDR, not per-test type-I error.** A few
  false positives are expected when many tests run. Use the effect
  sizes and CIs to triage.

* **Per-call sampling variance.** The MVP doesn't run repeated trials
  per (prompt, example) pair. If your models are stochastic,
  consider setting `temperature=0` (the registry default) or running
  multiple seeds and averaging the scores upstream.
