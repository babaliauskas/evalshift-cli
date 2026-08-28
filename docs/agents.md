# Agent migrations

EvalShift compares **agent behaviour** — which tools the model called,
what arguments it passed, and how it sequenced them — across two model
versions.

The killer scenario it catches:

> A team migrates a customer-support agent from Gemini 2.5 Flash to
> 3.1 Flash-Lite. The new model silently stops calling
> `notify_security_team` on sensitive requests. Text-only eval reports
> green; EvalShift marks it CRITICAL and blocks the migration.

## How it works

* **`ToolTrace` data model**: provider-agnostic, populated from
  Anthropic / OpenAI / Gemini responses.
* **Three new evaluators**: `tool_selection`, `tool_arguments`,
  `tool_trace_structure`.
* **Suite extension**: optional `expected_tools`, `expected_tool_count`,
  `expected_no_tools`, `expected_parallel` per example, plus a required
  `toolset_ref` or inline `tools` — see [Suite ground truth](#suite-ground-truth).
* **HTML report**: side-by-side trace diffs in place of text panes
  for tool-evaluator regressions.
* **Hosted run-detail**: the same trace — tool calls, arguments, final text,
  round markers — is visible on the hosted run-detail page after `evalshift
  push`, not just in the local HTML report. See [Hosted alpha](hosted.md).

## Walkthrough

`examples/agent/` in this repo is a complete, checked-in agent project —
six customer-support tools (`search_orders`, `lookup_customer`,
`issue_refund`, `update_order_status`, `send_email`,
`notify_security_team`) and a golden suite across five slices
(`security`, `routine`, `refund`, `customer_lookup`, `text_only`), each
row carrying its own `toolset_ref`:

```bash
cd examples/agent
export GOOGLE_API_KEY=<google-api-key>
evalshift run --yes --from gemini-2.5-flash --to gemini-3.1-flash-lite-preview
RUN_ID=$(ls .evalshift/runs/ | head -1)
evalshift evaluate "$RUN_ID"
evalshift analyze "$RUN_ID"
evalshift report "$RUN_ID" --open
```

`evalshift run` resolves each golden-suite example's own toolset
(`toolset_ref` or inline `tools`, see [Suite ground truth](#suite-ground-truth))
and dispatches through the live agent path automatically for any example
whose toolset is non-empty — every such `Call` row in `raw.jsonl` carries a
parsed `ToolTrace`, and the configured `tool_*` evaluators score against the
per-example `expected_tools` ground truth.

For your own agent, the recommended path is capture-first: instrument it
with the [evalshift-sdk](https://github.com/babaliauskas/evalshift-sdk),
exercise it to record captures, then `evalshift capture sync` promotes them
into a golden suite carrying the toolset your production agent actually
offered — see [Getting started](getting-started.md).

## Configuration

A minimal agent config:

```yaml
version: 1
prompts:
  - id: routing
    detection: python_string
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-3.1-flash-lite-preview
  judge_model: gemini-3.1-pro-preview
  cache: false

evaluators:
  # Top-level: what every suite is scored with. Tool evaluators do not belong
  # here — `capture sync` writes them per suite (see below).
  semantic:
    embedding_model: gemini/gemini-embedding-001
    blocking: false

# >>> evalshift suites (managed by `evalshift capture sync`) >>>
suites:
  main_chat:
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected     # grade each side vs example.expected_tools
          divergence: set           # ...and the target vs what the source did
      tool_arguments:
        - name: routing_args
          against: expected         # score both sides vs the recorded arguments
      # Optional — not derived, add by hand under `managed: false`:
      # tool_trace_structure:
      #   - name: routing_structure
# <<< evalshift suites <<<
```

The `suites:` region is generated: `evalshift capture sync` reads what each
suite's rows actually contain and writes that suite's own evaluator block —
`tool_selection` when any row was offered a toolset, `tool_arguments` when any
row recorded call arguments, and nothing at all for a suite whose captures never
called a tool. That last case is the point: a tool evaluator pointed at a
tool-free suite scores an empty denominator, which the policy reads as an
*inconclusive* gate rather than as "not applicable here". A family a suite
declares replaces the top-level one wholesale; families it does not mention are
inherited. See
[Configuration → per-suite evaluators](configuration.md#per-suite-evaluators).

Hand edits inside the markers are regenerated away on the next sync. To keep
them — a `severity_floor: high` on `routing`, a per-field strategy — set
`managed: false` on that suite's entry; sync then prints what it would have
written instead of writing it.

> **Note:** `structural.length` is intentionally **not** in the
> scaffolded config. Agent runs frequently produce empty `final_text`
> (the model returned only tool calls), which makes the length
> evaluator score 0/0 across every routine row — pure noise. Add it
> back manually only for prompts that produce text.

Nothing in `evalshift.yaml` wires a toolset to a prompt — dispatch reads it
off each golden-suite *example* instead (`toolset_ref` or inline `tools`, see
[Suite ground truth](#suite-ground-truth) below), so the same prompt can
legitimately dispatch some examples with tools and others without, in one
run. `tools.yaml` above is just this project's human-readable record of what
those tools are; it accepts either Anthropic-shape (`name` / `description` /
`input_schema`) or OpenAI-shape (`{ "type": "function", "function": {...}
}`) entries — `evalshift run` serialises whatever a toolset resolves to in
the right shape per provider.

## Suite ground truth

```jsonl
{"id": "ex_security_01", "inputs": {"query": "..."}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team"}], "toolset_ref": "sha256:1a2b3c..."}
{"id": "ex_text_01", "inputs": {"query": "what is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true, "tools": []}
```

Every example carries a toolset — `toolset_ref` (a pointer to a
`<base>/toolsets/<hex>.json` sidecar; what `capture promote`/`sync`
write) or inline `tools` (what a hand-authored suite uses; `[]` is a
real "no tools offered" value), exactly one of the two. `expected_no_tools`
means "tools were offered and none were called" — it is never set when
the example's toolset is empty, so it never asserts something no model
could have failed.

You rarely write these by hand — `evalshift capture promote` /
`capture sync` derive `expected_tools` (and friends) from recorded
production behaviour; `--strict-args`, `--names-only`, and
`--tool-count` control how strict the derived expectations are. See
[Configuration → Capture lifecycle](configuration.md#capture-lifecycle).

### Agent rounds and what a replay can reproduce

A captured agent turn is usually a **loop**: the model calls tools, reads the
results, calls more tools, then answers. `evalshift capture promote` groups
those calls into rounds and, by default, keeps only **round 1** as
`expected_tools`. Every round is preserved on the case as
`expected_tool_rounds`.

This is not a simplification — it is the only honest yardstick. `evalshift run`
issues **one** model call per example and does not feed tool results back, so a
candidate model cannot reach round 2. Scoring it against round-2 calls records a
regression that no model could avoid.

Promote with `--rounds all` to flatten every round into `expected_tools`. Do
that only when you have a reason to score the whole trace — for example when
comparing against an externally-produced multi-round trace.

### Where `expected` text comes from

`example.expected` is recovered from the capture's `final_output` event when
there is one, and otherwise from the last `model_call` that produced text — on
an agent turn that is the reply the user saw, after the tool round-trips.
Captures with neither are promoted without text ground truth and
`capture sync` says so.

The fallback exists because only the SDK's LangChain adapter emits
`final_output`; the manual capture API has no way to. Without it every
manually instrumented project promoted `expected: null` while the reply sat in
the last `model_call` the whole time. A non-string `output` is left alone
rather than stringified into ground truth nothing produced.

### Wrapper arguments are unwrapped

A capture SDK that decorates a Python function records that *function's*
parameters. An agent whose tools are `def archive_project(tool_args: dict)`
records every call as `{"tool_args": {"project_name": "..."}}`, while the model
only ever saw the flat properties declared in the toolset it was actually
offered. No model can produce the recorded shape, so ground truth in it
scores 0 against every candidate.

Promotion undoes this — but only when the declared schema *confirms* it: the
wrapper key must not be a declared property and the inner keys must all be
declared ones. The schema comes from the capture's own recorded
`toolset_ref` sidecar (the toolset that call was actually offered), not any
project config — so this works with no `evalshift.yaml` in sight. Without a
resolvable sidecar there is nothing to check against, so the recording is
left exactly as captured; a wrong guess would silently rewrite your ground
truth.

## What does not belong in a golden suite

Ground truth is what the agent *did right*, and a capture is not automatically
that. Three shapes are actively filtered or flagged during promotion:

- **Errored turns are refused.** A capture whose trace carries an `error` event
  died before the agent finished — an upstream 400, a timeout, a crash. Promoted
  naively it records *no tool calls*, which would become the assertion
  "calling nothing is correct here". `capture promote` exits non-zero and
  `capture sync` skips it, both naming the first error. `--allow-errored`
  promotes it anyway; even then the case never gets `expected_no_tools: true`,
  because a turn that never ran is not evidence that inaction was right.
- **Captures missing a recorded toolset are refused, unconditionally.** Every
  model call is required to record the toolset it was offered; a capture whose
  first `model_call` has no `toolset_ref` — the SDK failed to write the sidecar,
  or predates per-call toolset capture — has nothing to carry. `--allow-errored`
  does not help here (it is a different failure). The error names the capture
  id; the fix is re-capturing with a current `evalshift-sdk`.
- **Duplicated turns are warned about.** A retried turn produces two captures
  with the same `(conversation_id, turn_index)` — usually the failed attempt
  and the retry. Both would be promoted and their reconstruction order is
  arbitrary. `capture sync` warns once per collision so you can delete the
  case file you don't want.
- **Failed tool results are warned about.** A turn whose recorded tool result
  carries an `error`, or the common `{"success": false}` convention, is still
  promoted — "the model correctly tried, the backend was down" is legitimate
  ground truth — but you are told, because a candidate model is now being
  scored on reproducing a call that failed in production.

## Multi-turn agents

A follow-up question ("what time works?" → "1pm") is now a first-class
**conversation turn**, not just a bare `expected_no_tools` example scored in
isolation. If your agent captures record `conversation_id` / `turn_index` /
a messages-list `model_call.input`, `evalshift capture sync` links the turns
together and `run` replays each one with its recorded history prefix. See
[Multi-turn conversations](conversations.md) for the full walkthrough
(SDK-side capture, promotion, and teacher-forced replay semantics).

## Picking an evaluator

| Evaluator              | Use when                                                |
| ---------------------- | ------------------------------------------------------- |
| `tool_selection`       | You care about *which* tools fire (most common).        |
| `tool_arguments`       | You care about *what* the model passes to each tool.    |
| `tool_trace_structure` | You care about call counts, parallelism, or refusals.   |

You can run multiple at once. Each becomes an independent comparison
in `analysis.json`, with the existing Benjamini-Hochberg correction
already adjusting for the multi-test count.

### Parallel fan-outs and call order

`conformance: expected` matches the expected calls **in order**, which is right
for a sequential plan (fetch, then act) and wrong for a parallel fan-out
(archive six projects). Under an in-order walk, a model that emits the same
calls in a different sequence — or that made one expected call first — can
score 0. When your expected calls are a fan-out, use `conformance: expected_set`:

```yaml
tool_selection:
  - name: routing_selection
    conformance: expected_set
```

It scores the same recall, counting duplicates (two expected `archive_project`
calls need two actual ones) and ignoring extra calls, without penalising a
permutation. Call counts are `tool_trace_structure`'s job, not this one's.

### Scoring free-text arguments

Free-text tool arguments — search queries, titles, descriptions — will never
match exactly between two models. The default `default_strategy: auto` handles
that without configuration:

1. Strings equal after normalizing case and whitespace score `1.0`.
   `"Find people"` vs `"find  people"` is not a wrong value.
2. Fields the example's toolset declares as identifiers, enums, booleans or
   `date` / `date-time` / `uuid` / `email` formats are scored `exact` — a
   reworded timestamp *is* wrong. Numbers go to `numeric`, objects and arrays
   to `subset`.
3. What is left is genuine free text, and it is graded: embedding similarity
   when an `evaluators.semantic` block lent a model, `difflib` ratio when not,
   so partial credit survives even with no embedding model configured.

Override a field explicitly when you know better:

```yaml
tool_arguments:
  - name: routing_args
    strategies:
      query: semantic
      order_ref: exact
```

`strategies` keys are **field names matched across every tool**, so pick names
that mean the same thing everywhere in your toolset. A `strategies` entry always
wins over `default_strategy`. `semantic` borrows the embedding model (and cache)
from your `evaluators.semantic` block — with no semantic evaluator configured
there is no model to borrow and the strategy degrades to `exact`. Set
`default_strategy: exact` on the evaluator to score every unlisted field by byte
equality, the pre-0.12 behaviour.

### Scoring arguments against ground truth

`tool_arguments` compares the target's arguments to the **source's** by default.
That answers "did the arguments change?", not "are the arguments right" — the
source's own score is 1.0 by construction, even when the source passed a value
that does not exist. For a capture-first suite, set `against: expected` so both
models are scored against the arguments your production agent actually recorded:

```yaml
tool_arguments:
  - name: routing_args
    against: expected
```

Only expectations that carry `arguments` are scored; name-only expectations are
`tool_selection`'s business. An expected call the model never made scores 0 — a
missing call cannot have correct arguments — and extra calls beyond the
expectation are ignored here. Which keys are compared follows each expectation's
`match_strategy`: `exact` also flags arguments the ground truth did not record,
while `subset` (what `capture promote` writes) scores the recorded keys only.

Check `evalshift doctor` first. If it warns about **tool argument shape**, your
recorded arguments use keys the declared schema does not have, and every
ground-truth comparison will score 0 for a reason that is not the model's fault.

A ground-truth field that **neither** model produced is dropped from that call's
denominator on both sides and disclosed as `unmeasured_fields` in the record's
per-call metadata. It is a stale expectation — an argument your agent used to
pass and no longer does — not a model defect, and scoring it would cap the call
below 1.0 for good. For the same reason `ARGUMENT_VALUE_DRIFT` is stamped only
when the target scores *below* the source: both models missing the same
expectation by the same margin is a fact about the suite.

#### Ground-truth provenance

`capture promote` / `capture sync` transcribe `expected_tools[].arguments`
verbatim from the source model's own recorded call and mark each expectation
`provenance: captured`. On such a row the source scores 1.0 **by construction**,
so `against: expected` quietly measures the same thing `against: source` does —
target deviation from source — while `source_score: 1.0` reads like evidence the
source was right. Nothing is wrong with the number, so the run says so instead of
changing it: when every scored row is `captured`, `migration_decision.json`
carries a recommendation naming the count and the caveat.

Set `provenance: reviewed` on a golden row once a human has checked its
arguments:

```jsonl
{"id": "ex_refund_01", "inputs": {"query": "..."}, "toolset_ref": "sha256:1a2b...", "expected_tools": [{"tool_name": "issue_refund", "arguments": {"order_id": "A-1", "amount_usd": 40.0}, "provenance": "reviewed"}]}
```

Scoring is identical either way. The disclosure goes silent as soon as one row
is `reviewed`: a suite someone has started checking is no longer uniformly
source-derived, and a blanket disclaimer over it would understate the rows they
did check.

### Optional tool parameters

A parameter that is `"required": []` in your tool schema can legitimately be
passed by one model and omitted by the other: `get_projects(status="active")`
and `get_projects()` are both valid calls. That scores 0.5, not 0.0 — a real
difference worth surfacing, not a total failure. Set
`optional_fields_scored: strict` to score presence exactly.

## Reading a tool run's report

`report.html` renders the two `tool_selection` axes as separate rows,
each labelled with its slug (`routing · tool_selection.divergence`) and
with what it compares. They are not interchangeable: divergence measures
your migration, conformance measures your suite.

Two places name the tools themselves, both read off the evaluator's own
record rather than re-derived from the call trace:

* the **per-example breakdown** gains a `Tools called (source → target)`
  column — one line per example, so a ten-example suite shows all ten
  rather than only the five worst;
* each **top regression** card states both sides' tools in its "why
  flagged" line.

The **Tool match** column is signed: ✗ means some tool evaluator scored
the target *below* the source, not that the target missed absolute
perfection. A pair on which both models miss your recorded ground truth
identically is not a ✗ — it is the same pair before and after, and what
is wrong is the ground truth. Look for **Ground truth missed by both**
in the evaluator table when that happens.

## Troubleshooting

* **A suite reports tool evaluator scores you didn't expect on some rows** —
  run `evalshift doctor`. It reports the toolset each configured suite
  carries and flags a suite whose examples carry more than one distinct
  toolset — legal (each example dispatches its own), but also the shape a
  wiring mistake takes, so it's worth confirming the split is intentional.
* **Bimodal score distribution** — tool evaluators often produce
  scores at exactly 0 or 1. The analysis layer's Shapiro-Wilk
  fallback routes these through Wilcoxon signed-rank automatically.
* **"no matched calls between source and target"** — the
  `tool_arguments` evaluator scores a regression when the target
  doesn't reuse any of the same tool names as the source. Check
  `tool_selection` first to triage.
