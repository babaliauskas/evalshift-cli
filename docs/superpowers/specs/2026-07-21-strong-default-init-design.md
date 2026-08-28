# Strong-default `evalshift init` — design

**Date:** 2026-07-21
**Status:** approved (approach + sections 1–2 approved explicitly; remainder
approved by delegation — "use your own judgement")

## Problem

`evalshift init` scaffolds a config that fails real migrations for reasons
unrelated to model quality. Observed on a live personalButler migration
(gemini-3.1-flash-lite-preview → gpt-5.4-mini, runs `r_20260720_*_3a56ff`
and `r_20260720_*_8bd2ba`):

1. **Directional judge criterion vs anonymized A/B.** The scaffold's
   `criterion_prompt` asks about "TARGET vs SOURCE", but
   `PairwiseJudgeEvaluator` shuffles outputs into anonymous slots A/B.
   The judge cannot orient the question; in run `3a56ff` it answered "A"
   12/12 times and the win/loss split was 100% explained by the slot
   shuffle, 0% by content.
2. **Judge failures score as ties.** Judge/semantic API failures return
   neutral 0.5/0.5 (or 0/0) scores with `error=None`; a run where 100%
   of judge calls failed reads as a clean pass.
3. **Reasoning-model judges are unusable.** `temperature=0.0` is
   hardcoded; gpt-5.6-class models reject any temperature ≠ 1 →
   every call fails (then silently ties, per #2).
4. **`semantic.cosine` always ends "critical".** Source is pinned at 1.0
   (compared to itself), so the paired Wilcoxon tests "is target
   byte-identical to source", which is always significant. The evaluator
   can only ever produce a blocking regression.
5. **Policy budgets assume large n.** Profile defaults
   (`max_overall_regression_rate: 0.03`, `min_equivalence_rate: 0.95`)
   are unreachable at capture-scale n≈8–20 where one flipped example is
   a 5–12% swing. Users loosen budgets to meaningless values to cope.
6. **`capture sync` promotes duplicates.** Re-exercising an agent on the
   same data produces captures with identical `input_hash`; sync
   promotes both. Observed twice: 12 rows/6 unique, 16 rows/8 unique —
   every n, p-value, and effect size doubled/inflated.

## Goals

Config written by `init` should give honest verdicts with zero edits for
the common case. Deterministic signals gate; noisy signals inform. Small
n yields `inconclusive`, not false `fail`. Broken evaluator calls are
excluded, never neutral-scored.

## Design

### 1. `blocking` flag per evaluator (approved)

* Every evaluator config model in `config/models.py` gains
  `blocking: bool = True` (`SemanticEvaluatorConfig`, `LLMJudgeConfig`,
  `StructuralEvaluatorConfig`, `ToolSelectionEvaluatorConfig`,
  `ToolArgumentsEvaluatorConfig`, `ToolTraceStructureEvaluatorConfig`,
  `AgentTraceEvaluatorConfig`).
* `EvalRecord` gains `blocking: bool = True`; stamped at evaluate time
  (`_build_evaluators` attaches the config value to each evaluator
  instance; `_score_one`/tool/trace paths copy it onto the record).
  Old `scores.jsonl` files load as blocking (back-compat default).
* Scaffold ships `semantic` and `llm_judge` with `blocking: false` +
  a comment explaining advisory semantics and when to flip.

### 2. Decision engine: advisory-aware + CI-aware budgets (approved)

In `analysis/policy.py`:

* Records partition into blocking/advisory via `EvalRecord.blocking`.
  Rate metrics (`regression_rate`, `equivalent_rate`, `improved_rate`,
  `critical_regressions`, `tool_argument_drift_rate`) computed from
  blocking records only. A parallel `advisory: PolicyMetricSummary | None`
  field on `MigrationDecision` carries the same shape for advisory
  records (None when none exist).
* Comparisons whose `evaluator_name` belongs to an advisory evaluator are
  excluded from `_verdict_for` and `_blocking_regressions`; they are
  collected into `advisory_regressions` (same `BlockingRegression` shape)
  for the report.
* **Wilson CI on rate budgets.** `BudgetResult` gains
  `ci_low: float | None`, `ci_high: float | None`,
  `conclusive: bool = True`. For `max_overall_regression_rate` and
  `min_equivalence_rate`, compute a 95% Wilson interval at blocking-n:
  * observed within budget → **pass** (wide CI does not block a clean run)
  * observed breaches and CI excludes the budget → **fail**
  * observed breaches but CI straddles the budget → **inconclusive**,
    with `reason` explaining n and the interval
  Count budgets (`max_critical_regressions`) and cost/latency ratios
  stay exact.
* Verdict precedence: conclusive budget fail → `fail`; blocking-severity
  comparison → `fail`; non-conclusive breach → `inconclusive`;
  regression-severity comparison → `conditional_pass`; else `pass`.
* Profile policy numbers stay strict (0.03/0.95 etc.) — small-n runs now
  report `inconclusive` instead of false `fail`, and the numbers become
  binding as suites grow.

### 3. Judge fixes

* **Symmetric scaffold criterion** (no TARGET/SOURCE, explicit tie):

  > Which output is more complete and correct? Prefer valid JSON over
  > fenced or malformed JSON, no dropped or invented fields or entity
  > ids, and conclusions grounded in the input. Answer "tie" when both
  > are equivalent in substance and differ only in wording.

* **`drop_params`.** `ModelClient` passes `drop_params=True` to
  `litellm.acompletion` so unsupported params (temperature on
  reasoning models) are dropped instead of erroring. Judge keeps
  `temperature=0.0` for determinism where supported.
* **Failures raise.** `PairwiseJudgeEvaluator.score` re-raises judge
  call/parse failures as `EvaluatorError`; `_score_one` already converts
  any raise into an `error=`-stamped record, which `_metrics` and
  slicing exclude. Same change in `CosineSimilarityEvaluator` for
  embedding failures. The `Evaluator` protocol docstring drops
  "should never raise".

### 4. Semantic stays paired but advisory

Scoring model unchanged (source ≡ 1.0 is by design a target-preservation
metric), but the scaffold marks it `blocking: false`, so its
guaranteed-significant Wilcoxon can no longer gate a verdict alone. Its
regressions surface in `advisory_regressions` and the report. The
`min_similarity` flag still drives `SEMANTIC_REGRESSION` failure
categories.

### 5. Interactive provider selection in `init` (approved)

* New `--provider [gemini|openai|anthropic]` option. When omitted and
  stdin is a TTY, `init` prompts (numbered choice, default gemini).
  Non-TTY without the flag → gemini + a hint that `--provider` exists.
* Provider fills the scaffold's model ids:

  | provider  | source (edit-me baseline)          | judge                  | embedding                      |
  |-----------|------------------------------------|------------------------|--------------------------------|
  | gemini    | gemini-3.1-flash-lite-preview      | gemini-3.1-pro-preview | gemini/gemini-embedding-001    |
  | openai    | gpt-5.4-mini                       | gpt-5.6-luna           | openai/text-embedding-3-small  |
  | anthropic | claude-sonnet-5                    | claude-opus-4-8        | *(semantic commented out — no Anthropic embedding endpoint; comment explains)* |

* Scaffold comment on `judge_model`: prefer a judge from a different
  family than source *and* target when the target is known.

### 6. `capture sync` dedup

* Before grouping, drop captures whose `(suite, input_hash)` was already
  seen (first occurrence wins, iteration order = existing
  `iter_captures` order). Summary line reports
  `skipped N duplicate capture(s) (same input)`.
* `--keep-duplicates` opts out (variance measurement use case).

## Out of scope

* Entity-id set evaluator (deterministic JSON extraction scoring) —
  separate feature, tracked for a follow-up.
* Report HTML redesign; only additive advisory labelling data in JSON
  (template picks it up opportunistically).
* Registry refresh of stale model rows.

## Testing

* config: `blocking` parses on every evaluator model; default true;
  `extra="forbid"` still rejects typos.
* policy: advisory records excluded from gating metrics; `advisory`
  block populated; Wilson verdicts (small-n breach → inconclusive,
  large-n breach → fail, clean small-n → pass); advisory comparisons
  out of `blocking_regressions`, into `advisory_regressions`.
* judge/semantic: API failure raises → `_score_one` writes
  `error=`-record; scores excluded from `_metrics`.
* client: `drop_params=True` present in completion kwargs.
* capture sync: duplicate `input_hash` skipped + reported;
  `--keep-duplicates` keeps both; distinct hashes unaffected.
* init: `--provider openai` writes OpenAI ids; interactive prompt path;
  non-TTY default; existing init tests updated for new template.

## Compatibility

* Old configs (no `blocking`) behave exactly as today (all blocking).
* Old `scores.jsonl` analyze fine (`blocking` defaults true).
* `migration_decision.json` gains additive fields
  (`advisory`, `advisory_regressions`, per-budget `ci_low`/`ci_high`/
  `conclusive`); existing consumers unaffected.
* Judge/semantic failures that previously produced fake ties now produce
  excluded error records — metrics change only for runs that were
  already broken.
