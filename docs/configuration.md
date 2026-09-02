# Configuration reference

Every EvalShift run is driven by a single `evalshift.yaml` file. This
page documents every field — types, defaults, and what they do.

`evalshift init` writes a minimal, capture-first `evalshift.yaml` —
a passthrough `replay` prompt, default models, evaluators, and an empty
managed `suites:` block you fill in with `evalshift capture sync`. Below is
the canonical reference.

## Top-level shape

```yaml
version: 1                # required, must be 1
project: org/project      # optional, required for hosted push unless passed by flag
thresholds: {...}         # optional hosted project thresholds
migration_policy: {...}   # optional local migration verdict policy
prompts: [...]            # required, at least one
defaults: {...}           # optional
evaluators: {...}         # optional (but at least one is needed for `evaluate`)
slices: [...]             # optional
suites: {...}             # optional named suites for `run --suite-name`, each with
                          # its own optional `evaluators:` block
retention: {...}          # optional run-history pruning policy
```

Unknown keys are rejected (`extra: forbid` everywhere) so typos fail
fast instead of silently dropping.

## Config version policy

`version: 1` changes **only for breaking changes** — a field renamed, removed,
or given a different meaning. Additive fields (a new evaluator option, a new
top-level block) do *not* bump it; they ride on the CLI version instead.

Because unknown keys are rejected, that puts one rule on you: the CLI that
*reads* a config must be at least as new as the CLI that *wrote* it. In
practice that means the `evalshift-version` your CI workflow installs must be
at least the version you run `capture sync` and `init` with locally. The CI
pin check is the mechanism that enforces it: `capture sync`, `init`, `doctor`,
and `validate` warn when a workflow under `.github/workflows/` pins an older
CLI, or none at all, and print the exact line to set. See
[Pin drift](github-action.md#pin-drift).

## `project` and `thresholds`

These fields are used only by hosted commands. Local `run`, `evaluate`,
`analyze`, and `report` do not require them.

| Field        | Type   | Required | Description |
| ------------ | ------ | -------- | ----------- |
| `project`    | string | no       | Hosted project slug in `org-slug/project-slug` form. `evalshift push` also accepts `--project`, which overrides the config value. |
| `thresholds` | object | no       | Free-form hosted project thresholds. When provided during `push`, the backend syncs them for owners and returns canonical thresholds. |

Example:

```yaml
project: acme/model-migration
thresholds:
  pass_rate_min: 0.95
  regression_max: 0
```

`evalshift init` scaffolds `project:` **commented out**, with the slug shape and
a placeholder — uncomment it once you have a hosted project. It is left unset
because a local run never needs it and nothing uploads without an explicit
`push` / `all --push`, so a guessed slug would be wrong in the one place it
matters.

## `migration_policy`

Optional local regression budget used by `evalshift analyze`,
`evalshift all`, and the HTML report to produce a migration verdict.
Ratio fields use decimal fractions: `0.03` means 3%.

```yaml
migration_policy:
  max_overall_regression_rate: 0.30
  max_critical_regressions: 1
  min_equivalence_rate: 0.75
  max_tool_argument_drift: 0.20
  max_tool_divergence: 0.20  # share of pairs where the target routed elsewhere
  tool_argument_drift_floor: 0.9   # below this a call counts as drifted
  max_cost_increase: 0.50    # target may cost up to 50% more
  max_latency_increase: 2.0  # …and be up to 200% slower (small→large migration)
```

The rate values above are the defaults `evalshift init` writes (and the
`MigrationPolicy` field defaults a config that omits the block inherits): a
first-migration starting point, deliberately loose enough that a fresh suite
reports its regressions instead of failing on a couple of reworded tool
arguments. Tighten them as the suite grows and the migration nears merge; the
other `--profile` presets (`cost-reduction`, `quantization`, `provider-switch`,
`local-model`) scaffold tighter numbers already.

`max_overall_regression_rate`, `min_equivalence_rate`,
`max_tool_argument_drift` and `max_tool_divergence` are true rates bounded
`0.0`–`1.0`.
`min_equivalence_rate` is a floor on the **non-regression** rate: a record that
is equivalent *or improved* counts as passing, so a target that beats the source
still meets the `0.75` default (the downside is bounded separately by
`max_overall_regression_rate`). The two *increase*
budgets — `max_cost_increase` and `max_latency_increase` — may exceed `1.0`
(e.g. `2.0` = the target may be 200% more expensive / slower before the policy
fails), since migrating a small model to a much larger one is legitimately far
costlier and slower. They are capped at `10.0` (1000%) only to reject obvious
typos (e.g. writing `200` instead of `2.0`).

Rates are counted over whole rows, so pick values your suite can actually
express. With 10 tool-argument rows the achievable drift rates are
`{0.0, 0.1, 0.2, …}` and `max_tool_argument_drift: 0.01` means "any drift at
all fails" — the budget is below one row's worth. `analyze` says so rather than
enforcing it silently:

```
The tool-argument drift budget of 1% (max_tool_argument_drift in
evalshift.yaml) is below the 10% granularity of 10 tool-argument
comparisons — effective tolerance is zero at this sample size.
```

The same check covers `max_overall_regression_rate` and
`max_tool_divergence`, and each slice is judged on its own row count. Capture more examples, or write a budget the suite can
represent; `0.0` is silent, since an explicit zero-tolerance budget is a choice.

The two *increase* budgets have no row denominator, but they can go unmeasured
in their own way: if neither model is priced (LiteLLM has no cost table entry
for the id), every `cost_usd` is `0.0` and the ratio has nothing to divide.
`analyze` reports that budget as `conclusive: false` rather than as a clean
pass, and says why:

```
The cost increase budget could not be measured: all 4 error-free calls
across both models recorded a cost of 0, so its observed 0.00 is a default,
not a measurement.
```

`max_latency_increase` behaves the same when every call reports `latency_ms: 0`
— typically an unpriced or unmetered model. Price the models, or re-run
live, if you need those budgets to bind.

`max_tool_divergence` is the divergence axis's own budget: the share of
`tool_selection.divergence` rows on which the target called different tools
than the source, counted over *those* rows only. It has deliberately no
materiality floor of its own — an argument score slides continuously (a
reworded query is not a wrong call), whereas a divergence score below `1.0`
means the target called a tool the source did not, or skipped one it did.

A `tool_selection.conformance` row where **both** models missed the recorded
ground truth by the same margin is excluded from every policy rate: its zero
delta is a shared failure, not evidence the migration is safe. Such rows are
still reported, as a `TOOL_GROUND_TRUTH_MISS` count and as a recommendations
line naming how many were excluded — see [methodology](methodology.md). And
when the **source** model missed conformance on half or more of at least four
rows, `evalshift evaluate` reports a broken eval harness in red before any of
these budgets are read.

`tool_argument_drift_floor` is not a budget — it is the materiality threshold
`max_tool_argument_drift` is counted at. Argument scoring is continuous, so a
reworded search query or an omitted optional filter lands below `1.0` without
being wrong; counting every non-identical call would weigh a `0.98` the same as
a `0.0` and burn the drift budget on calls that were never wrong. A call counts as
drifted only when its target argument score falls **below** the floor. The `0.9`
default mirrors [`evaluators.semantic.min_similarity`](#evaluatorssemantic) —
the same "close enough" line on the same kind of score — and deliberately sits
above the `0.5` an omitted optional argument earns, because omitting a filter
changes which rows the tool returns. Lower it to `0.4` if you want presence
differences ignored.

Semantic-evaluator drift counts toward `max_overall_regression_rate` and
`min_equivalence_rate` only when it breaches
[`evaluators.semantic.min_similarity`](#evaluatorssemantic). A near-identical
output (cosine ~0.98) that stays within `min_similarity` is treated as
*equivalent*, not a regression — so the policy gate and the report agree.

Per-slice overrides live under `migration_policy.slices` — a map of
slice name to a partial policy block; unset fields inherit the top
level. A slice budget gates the run exactly like a top-level one: a
conclusively breached slice budget **fails** the run, an unconfirmed
breach makes it `inconclusive`, and `recommendations` names which slice
budget blocked. A slice that fails on comparison *severity* rather than
a budget still only downgrades an overall `pass` to `conditional_pass`.

```yaml
migration_policy:
  max_overall_regression_rate: 0.03
  slices:
    security:
      max_overall_regression_rate: 0.0   # zero tolerance on this slice
```

Two behaviours to know:

* **Only blocking evaluators gate quality.** Advisory
  (`blocking: false`) records are summarised separately and never flip
  the verdict. If *every* configured evaluator is advisory (the
  fresh-`init` state), the verdict is `inconclusive` — unless
  `max_cost_increase` or `max_latency_increase` is breached, which
  still `fail`s: those are computed from the run's calls, not from
  evaluator records.
* **The four rate budgets are Wilson-CI-aware.**
  `max_overall_regression_rate`, `min_equivalence_rate`,
  `max_tool_argument_drift` and `max_tool_divergence` are each a
  proportion of records, so each
  carries a 95% Wilson interval. A breach only *fails* when the interval
  confirms it; if the suite is too small to be sure, the verdict is
  `inconclusive`, not `fail`. A budget the observation *held* stays
  conclusive however wide its interval.
  Cost and latency budgets are exact, but report
  `conclusive: false` when neither model priced its calls (both averages
  zero is a default, not a measurement). Record-derived budgets report
  `conclusive: false` on a scope that scored zero records — their
  `0/0` default looks clean but measures nothing.

Verdicts are `pass`, `conditional_pass`, `fail`, or `inconclusive`.
When configured, `analyze` writes `migration_decision.json` next to
`analysis.json`; `report` renders it as the top-level migration verdict.
Use `--policy-gate` on `analyze` or `all` to fail CI for `fail` and
`conditional_pass`.

## `prompts`

A list of prompt definitions. Each entry has:

| Field         | Type    | Required                          | Description |
| ------------- | ------- | --------------------------------- | ----------- |
| `id`          | string  | yes                               | Stable identifier surfaced in reports. Must be unique within the file. |
| `detection`   | enum    | yes                               | `manual` or `python_string`. |
| `content`     | string  | when `detection: manual`          | Inline prompt body. Forbidden when `detection: python_string`. |
| `path`        | string  | when `detection: python_string`   | Relative or absolute path to a `.py` file. Resolved against the directory containing `evalshift.yaml`. |
| `variable`    | string  | when `detection: python_string`   | Module-level variable name holding the prompt string. |
| `variables`   | list    | optional                          | Names of `{template}` placeholders the prompt expects. Used by the pre-flight compatibility check. |
| `max_tokens`  | int     | optional (`> 0`)                  | Per-prompt override of `defaults.max_tokens`. Raise it for prompts whose models emit long JSON / tool arguments that would otherwise be truncated. |

### Two prompt-detection modes

* `manual` — write the prompt body inline:
  ```yaml
  - id: greet
    detection: manual
    content: "Hello {name}"
    variables: [name]
  ```
* `python_string` — point at an existing module-level string in your
  codebase:
  ```yaml
  - id: greet
    detection: python_string
    path: src/prompts/greet.py
    variable: GREET_PROMPT
    variables: [name]
  ```
  EvalShift AST-walks the file and extracts the string literal. **It
  does not run user code.** F-strings, concatenations, `.format()`
  calls, and other dynamic forms are explicitly rejected.

## `defaults`

| Field           | Type   | Default                       | Description |
| --------------- | ------ | ----------------------------- | ----------- |
| `source_model`  | string | (none)                        | Default `--from` model id (or alias). |
| `target_model`  | string | (none)                        | Default `--to` model id (or alias). |
| `judge_model`   | string | `gemini-3.1-flash-lite-preview` | Default LLM-as-judge model. |
| `insights_model`| string | (none)                        | Model that writes the run-insights narrative rendered in `report.html` and uploaded with the bundle. Falls back to `judge_model` when unset — writing analytical prose is a harder task than a pairwise A/B verdict, so it is worth tuning separately. See [Run insights](#run-insights). |
| `concurrency`   | int    | 10 (1 ≤ x ≤ 64)               | Max in-flight LLM calls during `evalshift run` **and** `evalshift evaluate` (the embedding and judge calls made while scoring). |
| `cache`         | bool   | `true`                        | Read/write the local SQLite cache at `~/.evalshift/cache.db`. Covers run-stage completions plus `semantic` embeddings and `llm_judge` verdicts. |
| `max_cost_usd`  | float  | 50.0                          | Soft ceiling reserved for future enforcement. The pre-flight cost prompt currently triggers above $10 (skip with `--yes`). |
| `max_tokens`    | int    | 4096 (`> 0`)                  | Completion length cap sent to every model call. Raise it if outputs are being truncated (the provider returns `finish_reason == "length"`); a `prompts[].max_tokens` entry overrides it per prompt. Truncated calls are detected, surfaced in the report, and **excluded from the regression statistics** so a cut-off output can't manufacture a false regression. |

### Run insights

`evalshift report` (and therefore `all`) writes a plain-language
explanation of the run — a summary each for the verdict, the advisory
signal and the economics, plus behavioural findings and a
recommendation. It is rendered at the top of `report.html` and uploaded
with the bundle.

```yaml
defaults:
  insights_model: gemini-3.1-flash-lite-preview   # optional
```

The prose is machine-written, but the figures in it are not: every
number is computed first and handed to the model pre-rendered as a
display string to copy verbatim, and the output is rejected and
regenerated if it contains a numeric token that was not supplied. Two
bad generations fall back to deterministic templated prose.

- One model call per run (a second only on a rejected generation),
  cached in `insights.json` and keyed on the run's `config_hash` plus
  the model id — re-running `report` or `push` costs nothing.
- Skip it with `--insights/--no-insights` on `report` and `all`. It is
  also skipped when no API key is configured for the chosen model, and
  when the run has no usable `evalshift.yaml`.
- A generation failure never fails the run.
- The worst 8 regressions' inputs and both models' outputs are sent to
  `insights_model` (truncated to 2000 characters each) — the same
  exposure an `llm_judge` criterion already has. Use `--no-insights`
  if that is not acceptable for your suite.

## `evaluators`

Three sub-keys, all optional. **At least one evaluator must be
configured for `evalshift evaluate` to do anything.**

### `blocking` (every evaluator)

Every evaluator entry accepts `blocking: bool` (default `true`).

| Value   | Behaviour |
| ------- | --------- |
| `true`  | Regressions from this evaluator count toward `migration_policy` budgets and can fail the migration verdict. |
| `false` | *Advisory*: the evaluator still scores every pair and its results appear in the report (under advisory metrics/regressions), but it never gates the verdict. |

The `init` scaffold marks `semantic` and `llm_judge` advisory: at the
small suite sizes fresh captures start with, embedding drift and judge
noise would otherwise dominate the verdict. Flip them to `blocking: true`
once your suite is large enough that you trust their calls. Deterministic
evaluators (structural, tool-call) default to blocking.

### `evaluators.structural`

A list. Each entry has a `type` and the fields that type needs.

| `type`        | Required fields                          | Behaviour |
| ------------- | ---------------------------------------- | --------- |
| `json_schema` | `schema_path` (string)                   | Each output is parsed as JSON; score 1.0 if it validates against the schema, 0.0 otherwise. |
| `regex`       | `pattern` (string)                       | Score 1.0 if the regex matches anywhere in the output, 0.0 otherwise. |
| `length`      | `min_chars` and/or `max_chars` (int)     | Score 1.0 inside the bounds, distance-decayed outside. |

Optional `applies_to: ["prompt-id-glob", ...]` (default `["*"]`) for
future per-prompt scoping.

### `evaluators.semantic`

A single object (not a list).

| Field             | Type   | Default                  | Description |
| ----------------- | ------ | ------------------------ | ----------- |
| `embedding_model` | string | `text-embedding-3-small` | LiteLLM-compatible embedding model id. Use a Gemini one (e.g. `gemini/gemini-embedding-001`) if you don't have an OpenAI key. |
| `min_similarity`  | float  | `0.9`                    | Cosine similarity (0–1) below which the target is flagged as a semantic regression. Minor rewording/formatting typically scores ~0.98, so the default 0.9 avoids false flags; set to `1.0` to flag any deviation from byte-identical. Also governs whether semantic drift counts toward the [`migration_policy`](#migration_policy) regression/equivalence gates. |

The semantic evaluator scores the **target's similarity to the source**:
target_score = cosine(source, target), source_score = 1.0. A
negative `delta` means the target drifted from the source's meaning.

### `evaluators.tool_selection`

A list. Each entry has:

| Field             | Type   | Default       | Description |
| ----------------- | ------ | ------------- | ----------- |
| `name`            | string | (required)    | Identifier surfaced in reports.  |
| `conformance`     | enum   | `expected`    | Ground-truth axis: `expected` / `expected_set` / `off`. `expected_set` is `expected` made order-insensitive: multiset recall of `example.expected_tools` names. Use it when the expected calls are a parallel fan-out whose order carries no meaning. |
| `divergence`      | enum   | `set`         | Target-vs-source axis: `set` (Jaccard on tool names) / `exact` (sequence equality) / `first` (first call only) / `off`. |
| `applies_to`      | list   | `["*"]`       | Glob list of prompt ids. |
| `severity_floor`  | enum   | `null`        | If set, surfaces in metadata so the analysis layer can floor severity. |

The two axes are independent and each writes **its own record**, under
`kind: tool_selection.conformance` and `kind: tool_selection.divergence`.
They answer different questions:

* **conformance** grades *each side* against the example's ground truth, so
  both can fail at once and the delta stays 0 — the migration did not cause a
  failure both models share. An example carrying `expected_no_tools` is graded
  against *that*, under either strategy; an example with no ground truth at all
  is not measured and writes no row. When **both** sides miss, the record is
  tagged `TOOL_GROUND_TRUTH_MISS` — ground truth captured from the source model
  that the source model then fails means the harness is misconfigured (wrong
  toolset attached, wrong prompt, suite promoted from a different agent), not
  that the migration regressed.
* **divergence** grades the target against the source, which is its own
  baseline at 1.0, so behaving differently is a negative delta — a regression.
  It needs no ground truth, which is the point: it is what catches two models
  failing the same expectation in two different ways.

`divergence` defaults to `set` rather than `exact` so that reordered identical
calls do not read as drift. Setting both axes to `off` is a config error — the
evaluator would measure nothing. There is no `mode` field: it was removed, not
deprecated, and a config still carrying one fails validation.

### `evaluators.tool_arguments`

| Field                   | Type   | Default | Description |
| ----------------------- | ------ | ------- | ----------- |
| `name`                  | string | (required) | Identifier. |
| `applies_to`            | list   | `["*"]` | Glob list. |
| `against`               | enum   | `source` | What arguments are compared to: `source` (drift from the source model) or `expected` (correctness against `expected_tools[].arguments`, scored on both sides). |
| `strategies`            | dict   | `{}`    | Per-field strategy overrides (`exact`/`subset`/`numeric`/`semantic`/`auto`). |
| `default_strategy`      | enum   | `auto`  | Strategy for fields `strategies` does not name. `auto` is the ladder below; `exact` restores byte-equality scoring. |
| `numeric_tolerance`     | float  | `0.05`  | Relative-error tolerance for `numeric`. |
| `optional_fields_scored`| string | `lenient` | How a field present on one side only is scored: `lenient` = 0.5, `strict` = 0.0. |
| `use_llm_judge_fallback`| bool   | `false` | Reserved; not yet implemented. |

`strategies` keys are **field names matched across every tool**, so pick names
that mean the same thing throughout your toolset. The `semantic` strategy needs
a configured [`evaluators.semantic`](#evaluatorssemantic) to borrow an embedding
model (and its cache) from; without one it degrades to `exact`.

#### The `auto` strategy ladder

Every field `strategies` does not name is scored by `default_strategy`, which
defaults to `auto`. `auto` is a ladder, cheapest rung first:

1. **Normalized exact.** Two strings that compare equal after normalization —
   case, surrounding whitespace, repeated internal whitespace — score `1.0`.
   No schema lookup, no API call. `"Find people"` vs `"find  people"` is a
   capitalization difference, not a wrong value.
2. **Schema dispatch.** The field is looked up in the toolset the example
   carries (`toolset_ref` sidecar or inline `tools`) and its declared type
   picks the strategy: identifiers (`*_id`, `*_ids`), `enum` values, booleans
   and `date` / `date-time` / `uuid` / `email` formats are scored `exact` —
   a reworded timestamp is wrong, not "similar"; `number` / `integer` go to
   `numeric`; `object` / `array` to `subset`.
3. **Graded similarity.** Whatever the schema did not decide — free text, and
   anything the example has no schema for — is graded rather than failed:
   `semantic` when an `evaluators.semantic` block lent an embedding model,
   `difflib` sequence ratio when it did not, so partial credit survives with
   no embedding model configured. Numbers still go through `numeric`, dicts
   and lists through `subset`, everything else through `exact`.

Set `default_strategy: exact` for byte-equality scoring, where a
capitalization difference is a wrong value. A per-field entry in `strategies`
always wins over `default_strategy`.

`against: source` (the default) answers *"did the arguments change?"* — the
source's own score is 1.0 by construction, so a source model that passed a value
that does not exist still scores perfectly. `against: expected` scores **both**
models against `expected_tools[].arguments`, which is what a capture-first suite
wants. Each expectation's `match_strategy` decides which keys are compared:
`exact` takes the union of expected and actual keys, `subset` and
`contains_per_field` (what `capture promote` writes) compare the recorded keys
only. An example with no expected arguments is skipped at a neutral 1.0/1.0.

`optional_fields_scored` governs *presence*, never values. A tool parameter that
is `"required": []` in your schema can legitimately be passed by one model and
omitted by the other — `lenient` scores that 0.5 rather than treating it as a
total failure, which is what turned every tool with optional parameters into a
false regression at `blocking: true`. A field both sides passed with different
values still scores by its strategy, so a wrong value is still 0.0.

Under `against: expected`, a ground-truth field that **neither** model produced
is dropped from that call's denominator on both sides and disclosed as
`unmeasured_fields` in the record's per-call metadata (`scores.json`). It is a
stale expectation, not a model defect: scored, it would cap the call below 1.0
for good, since no model change could ever lift it. A field only one side
omitted is unaffected — that is `optional_fields_scored`' business. A call whose
expectation consists entirely of such fields scores 1.0/1.0, consistent with the
neutral score for an expectation with nothing comparable in it.

`ARGUMENT_VALUE_DRIFT` is stamped on a record only when the target scored
**below the source** — a regression. Both sides failing the same expectation by
the same margin is a fact about your ground truth, and stamping it counted one
migration defect twice. The `migration_policy.max_tool_argument_drift` budget is
unaffected: it counts calls whose *target* score fell below
`tool_argument_drift_floor`, not failure-category labels.

### `evaluators.tool_trace_structure`

| Field                 | Type   | Default | Description |
| --------------------- | ------ | ------- | ----------- |
| `name`                | string | (required) | Identifier. |
| `applies_to`          | list   | `["*"]` | Glob list. |
| `check_call_count`    | bool   | `true`  | Score the number of tool calls. |
| `check_parallelism`   | bool   | `true`  | Score parallel-vs-sequential alignment. |
| `check_refusals`      | bool   | `true`  | Score refusal alignment; mismatches force `severity_floor: high`. |
| `call_count_tolerance`| int    | `1`     | `+/- N` calls considered equivalent. |

### `evaluators.agent_trace`

A list. These evaluators consume traces imported with
`evalshift traces import`; they do not run your agent and do not replace
the normal `raw.jsonl` model-call artifact.

| Field                        | Type   | Default | Description |
| ---------------------------- | ------ | ------- | ----------- |
| `name`                       | string | required | Identifier surfaced in scores and reports. |
| `applies_to`                 | list   | `["*"]` | Glob list of prompt ids. |
| `check_tool_order`           | bool   | `true`  | Compare source and target tool-call order. |
| `check_arguments`            | bool   | `true`  | Compare arguments for same-name matched tool calls. |
| `check_missing_verification` | bool   | `true`  | Check dangerous tools have an earlier verification tool. |
| `verification_tools`         | list   | `[]`    | Tool names treated as verification steps. |
| `dangerous_tools`            | list   | `[]`    | Tool names that should not appear without verification or as extras. |

Example:

```yaml
evaluators:
  agent_trace:
    - name: trace_safety
      verification_tools: ["check_refund_policy"]
      dangerous_tools: ["issue_refund"]
```

### `evaluators.llm_judge`

A list of pairwise judges. Each entry has:

| Field              | Type   | Required | Description |
| ------------------ | ------ | -------- | ----------- |
| `criterion_name`   | string | yes      | Short id surfaced in reports. |
| `criterion_prompt` | string | yes      | Free-form criterion the judge applies (e.g. "which output preserves more factual detail?"). |
| `judge_model`      | string | optional | Model used as the judge (built-in default `gemini-3.1-flash-lite-preview`). Prefer a judge from a third model family so it isn't grading its own relatives. |

The judge sees both outputs (with random A/B order to defang positional
bias) and produces strict-JSON `{"winner": "A"|"B"|"tie", "reason":
"..."}`. Target wins → `(0.0, 1.0)`; tie → `(0.5, 0.5)`; source wins →
`(1.0, 0.0)`. Malformed responses degrade to `(0.5, 0.5)` with the
error preserved.

## `slices`

A list of named subsets used for slice-level statistical analysis.
The implicit `"all"` slice always exists.

`overall` is reserved and cannot be used as a slice `name`, as an example tag,
or as a `migration_policy.slices` key. It names the run-level scope in the run
bundle, so a slice by that name would shadow the whole-run numbers wherever the
two are rendered together. All three spellings are rejected when the config or
suite loads.

| Field         | Type   | Required | Description |
| ------------- | ------ | -------- | ----------- |
| `name`        | string | yes      | Slice name surfaced in reports. |
| `filter`      | string | yes      | A tag string. Currently the filter is a literal tag — examples whose `tags` list contains the value land in this slice. |
| `applies_to`  | list   | optional | Glob list of prompt ids this slice applies to (default `["*"]`). |

Slices with identical membership are collapsed to one before analysis, so
duplicate tags cannot inflate the Benjamini–Hochberg correction. `all` and any
slice named under `migration_policy.slices` always survive; see
[methodology.md](methodology.md#slice-deduplication) for the full rule.

## Suite (`golden.jsonl`) shape

The suite is JSON Lines — one example per non-blank line. Each row:

| Field      | Type    | Required | Description |
| ---------- | ------- | -------- | ----------- |
| `id`       | string  | yes      | Unique within the suite. |
| `inputs`   | object  | yes      | Mapping of template-variable name to value. |
| `tags`     | list    | optional | Slice tags. |
| `expected` | object  | optional | Reference output (unused by most evaluators). |

Also present (v0.2, tool-call ground truth — see `docs/agents.md`):
`expected_tools`, `expected_tool_count`, `expected_no_tools`,
`expected_parallel`.

Each `expected_tools` entry carries `provenance`: `captured` (the default, and
what `capture promote` / `capture sync` write) means its arguments were
transcribed verbatim from the source model's own recorded call — nobody has
checked that they are *right*. Set it to `reviewed` once a human has confirmed
the row. Scoring is identical either way; the flag only controls whether the run
discloses that its `against: expected` ground truth is source-derived (see
[Agent migrations](agents.md#scoring-arguments-against-ground-truth)).

**`toolset_ref` (string) or `tools` (list) — required, exactly one of the
two.** Every model call records the toolset it was offered, so every suite
example must carry it too: `toolset_ref` points at a `<base>/toolsets/<hex>.json`
sidecar (what `capture promote`/`sync` write); `tools` inlines the toolset
directly (what a hand-authored suite uses — `[]` is a valid "no tools offered"
value). Neither present, or both, fails to load. See `docs/agents.md`.

Multi-turn conversation fields (see `docs/conversations.md` for the full
walkthrough — all optional, additive, single-turn suites parse unchanged):

| Field             | Type                                 | Required | Description |
| ----------------- | ------------------------------------- | -------- | ----------- |
| `history`          | list of `{role, content}` or `null`  | optional | Conversation prefix replayed verbatim before the current turn (teacher-forced). `role` is `system`, `user`, or `assistant`; at most one `system` message, and it must come first if present. `null` (the default) means single-turn — no message-mode dispatch. |
| `conversation_id`  | string or `null`                     | optional | Id of the recorded conversation this turn came from. Provenance only. |
| `turn_index`       | integer (`>= 0`) or `null`           | optional | Zero-based position of this turn within its conversation. Shown as a `turn N` badge in the HTML report. |
| `generation_config` | object or `null`                    | optional | Generation settings recorded by the SDK on the capture's first model call (`temperature`, `response_mime_type`, `response_schema`, ...). Written by `capture promote`/`sync`; the runner translates it at dispatch — `temperature` overrides the model default, and `response_mime_type: application/json` (plus an optional `response_schema`) becomes a LiteLLM `response_format` on both the source and target calls. Delete the field to disable the override; unknown keys inside it are ignored. |

Unknown keys are rejected (typos fail fast).

## `suites`

A map of named suites so `evalshift run --suite-name <name>` can resolve a
suite path without retyping it. Optional — omit it and `run` uses `--suite`
(or the default `golden.jsonl`).

```yaml
suites:
  support_agent:
    source: captured
    path: .evalshift/suites/support_agent/golden.jsonl
```

| Field        | Type   | Required | Description |
| ------------ | ------ | -------- | ----------- |
| `source`     | string | optional | `captured` (built by `capture promote`) or `jsonl` (hand-authored). Advisory provenance; both resolve to `path`. Default `captured`. |
| `path`       | string | yes      | Path to the suite JSONL, **relative to the config file's directory**. |
| `evaluators` | block  | optional | Evaluators this suite is scored with, replacing the top-level `evaluators:` family by family. Omitted (the default) scores the suite with the top-level block unchanged. |
| `managed`    | bool   | optional | Whether `capture sync` owns this entry. Default `true`. |

Resolution precedence for `run`: an explicit `--suite <path>` wins, then
`--suite-name <name>` (looked up here), then the default `golden.jsonl`.

### Per-suite `evaluators`

A project's suites are rarely homogeneous: one calls tools, six answer in prose.
One top-level `evaluators:` block either leaves the tool-calling suite unmeasured
or hands the tool-free ones a tool evaluator with an empty denominator, which
reads as an *inconclusive* gate rather than as "not applicable here". So a suite
can carry its own block:

```yaml
suites:
  main_chat:
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected
          divergence: set
      tool_arguments:
        - name: routing_args
          against: expected
  briefing:
    source: captured
    path: .evalshift/suites/briefing/golden.jsonl   # inherits the top level as-is
```

The block takes the same families as the top-level `evaluators:` — `structural`,
`semantic`, `llm_judge`, `tool_selection`, `tool_arguments`,
`tool_trace_structure`, `agent_trace` — and resolution is **family-level
replacement**:

- A family the suite does **not** mention is inherited from the top level.
- A family it **does** mention replaces the top-level one wholesale. There is no
  deep merge and no per-evaluator-name merge: the suite's list is the suite's list.
- Writing the family as `[]` or `null` is how a suite **removes** a family it
  would otherwise inherit. (Absent and `null` are different instructions here.)

So in the example above `main_chat` is scored with its own two tool evaluators
plus whatever `semantic` / `llm_judge` / `structural` the top level declares,
and `briefing` is scored with the top level alone. The same resolution feeds
`evaluate`, the HTML report and the hosted bundle, so what was scored is what is
reported. A run launched with a raw `--suite <path>` (no `--suite-name`), or with
a name that has no `suites:` entry, resolves to the top-level block.

### `managed`

`capture sync` regenerates a managed suite's entire entry — `path` and
`evaluators` — from what that suite's captures contain, so hand edits inside the
marker-delimited region are overwritten. Set `managed: false` to freeze an entry:

```yaml
suites:
  main_chat:
    managed: false
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_arguments:
        - name: routing_args
          against: expected
          strategies:
            amount_usd: numeric
```

Sync then leaves the entry alone and prints the block it *would* have written,
so you can diff your edits against the current derivation. Syncing one suite
never touches another's entry either way.

## `retention`

Bounds how much run history accumulates under `.evalshift/runs/`. Every `run` / `all` invocation
writes a fresh `r_<date>_<suite>_<hex>/` directory, so without a cap they pile up indefinitely.
After each **completed** run the orchestrator prunes old directories automatically; an in-progress
run and the run that just finished are never touched.

```yaml
retention:
  max_runs_per_suite: 20   # keep the 20 newest runs per suite; 0 disables count pruning
  run_ttl_days: 30         # (optional) also delete runs older than 30 days
```

| Field                | Type      | Default | Description |
| -------------------- | --------- | ------- | ----------- |
| `max_runs_per_suite` | int (≥ 0) | 20      | Keep at most this many run directories **per suite** (grouped by the suite slug in the run id), evicting the oldest by mtime. `0` disables count-based pruning. |
| `run_ttl_days`       | int (≥ 1) | (none)  | Also evict run directories older than this many days. Omit to disable age-based pruning. |

Pruning is grouped per suite, so a rarely-run suite isn't evicted just because another suite is
busy. The two rules combine (a run is deleted if **either** applies). `EVALSHIFT_MAX_RUNS` overrides
`max_runs_per_suite` from the environment (`0` / `none` / `unlimited` disables count pruning), which
is handy in CI where you don't want to keep any history.

Clean up on demand with `evalshift runs clean` — it applies the same rules with explicit overrides:

```shell
evalshift runs clean --dry-run            # preview what would be deleted
evalshift runs clean --keep 5             # keep the 5 newest per suite
evalshift runs clean --older-than 14      # delete runs older than 14 days
evalshift runs clean --suite main_chat    # restrict to one suite
```

`--keep` beats `EVALSHIFT_MAX_RUNS`, which beats the config value. `runs clean` works even without a
valid `evalshift.yaml` (it falls back to the defaults), so it's always available for disk cleanup.

## Capture lifecycle

The companion `evalshift-sdk` package records real agent runs to
`.evalshift/captures/<suite>/<capture_id>.json` (set `EVALSHIFT_DIR` to relocate
the base). The CLI consumes them via the `capture` commands:

```shell
evalshift capture list                          # see what the SDK recorded
evalshift capture sync --input-var query        # promote every capture + wire suites:
evalshift run --suite-name support_agent --yes  # score a candidate model against it
evalshift capture clean                         # prune already-promoted captures
```

`evalshift capture sync` is the one-shot path: it promotes **every** capture
under `.evalshift/captures/` into golden suites at
`.evalshift/suites/<suite>/golden.jsonl` **and** injects the resulting
`suites:` block into `evalshift.yaml` between the managed marker comments.

Each suite's entry is generated from what that suite's own rows contain, tool
evaluators included, so nothing has to be wired by hand:

- No row was offered a toolset → no `evaluators:` block at all; the suite
  inherits the top level, and no tool evaluator scores an empty denominator.
- Any row was offered a toolset → `tool_selection: [{name: routing, conformance:
  expected, divergence: set}]`.
- Any row recorded tool-call arguments → `tool_arguments: [{name: routing_args,
  against: expected}]`. No `strategies:` block: the default `auto` strategy
  already grades free text by meaning rather than by bytes.

The generated names (`routing`, `routing_args`) are stable by contract — reports
key on evaluator names across runs, so regenerating a suite must not rename what
it already wired. `structural` is deliberately not derived: nothing in a capture
says what shape an answer must have. Sync regenerates only the suites it just
promoted and carries every other entry in the region forward verbatim, and
`managed: false` freezes an entry entirely (see [`suites`](#suites)).
Captures with no recorded events are skipped, and captures whose replayed
content duplicates an already-promoted case (or an earlier capture in the
same run) are skipped too — duplicate examples inflate *n* and corrupt the
paired statistics (`--keep-duplicates` opts out). The dedup set is seeded
from the cases already in the suite dir, so re-syncing after recording more
captures can't slip a duplicate past it. Useful flags: `--input-var`
(default `input`), `--suite <name>` to filter to one suite, `--tag`,
`--names-only`, `--tool-count`, `--strict-args`, `--force`/`-f` to overwrite
existing suite files, and `--print` to preview the wiring without writing
(`--write` is the default). After syncing, run the whole pipeline against a
named suite with `evalshift all --suite-name <suite>` (it mirrors
`evalshift run --suite-name`).

To promote a single capture instead of all of them, use
`evalshift capture promote`:

```shell
evalshift capture promote cap_abc --as case1 \
    --input-var query                           # → .evalshift/suites/<suite>/case1.json (+ golden.jsonl)
```

`promote` derives a golden case from the recorded run: the captured tool calls
become `expected_tools`, the final output becomes `expected`, and (best-effort)
the first model input becomes `inputs`. Because a capture stores only a one-way
`input_hash`, structured/opaque inputs can't always be recovered — use
`--input-var` for single-string prompts, or edit the generated case file.
