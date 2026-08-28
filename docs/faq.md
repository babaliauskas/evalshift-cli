# FAQ

## Does EvalShift send my prompts to EvalShift servers?

**Not during local runs.** `doctor`, `run`, `evaluate`, `analyze`, and
`report` operate locally. Every provider API call goes directly from your
machine to the LLM provider you configured (Anthropic, OpenAI, Google)
using your own API keys.

The local SQLite cache at `~/.evalshift/cache.db` only contains
provider responses for *your* prompts and inputs.

Hosted private-alpha uploads are explicit. `bundle` packages the completed
local run artifacts into `run_bundle.json.gz` without uploading them. `push`
and `all --push` upload that bundle to the hosted backend for your project.

One local stage does call a provider with your data beyond the run itself:
`report` generates the run-insights narrative, sending the worst regressions'
inputs and outputs to `defaults.insights_model`. That is your provider, not
EvalShift's, and `--no-insights` turns it off.

## What is the summary at the top of the report, and can I trust its numbers?

It is the run-insights narrative — plain-language prose written by
`defaults.insights_model` (falling back to `judge_model`) explaining the
verdict, the advisory signal, the economics and what changed behaviourally in
the worst regressions.

The prose is machine-written; the numbers are not. Every figure is computed
from the run and handed to the model pre-rendered as a string to copy
verbatim, and any numeric token in the output that was not supplied causes the
generation to be rejected and retried. After two bad generations the CLI ships
deterministic templated prose instead (shown as model `none`). So a figure in
that block is the same figure as in the tables below it, or it is not there at
all.

It costs one model call per run, is cached in `insights.json`, and skips
itself when no API key is configured. Disable it with
`evalshift report --no-insights` or `evalshift all --no-insights`.

## What happens if a single LLM call fails?

The orchestrator records the error in `raw.jsonl` (with `error="..."`)
and moves on. The run still completes; failed calls are recorded with
a neutral 0.5/0.5 score in the evaluation phase so the analysis can
account for them rather than silently dropping examples.

## What models does EvalShift support?

Anything LiteLLM supports. The `evalshift.models.registry` provides
friendly aliases and sane defaults for common models (Claude, GPT,
Gemini), but **the registry is advisory, not gating**. A model id
that isn't in the registry — for example a fresh preview from a
vendor playground — gets passed through to LiteLLM with a
prefix-inferred provider. LiteLLM is the source of truth at call
time.

## Can I resume a run after Ctrl+C / a crash?

Yes. `evalshift run --resume` finds the latest in-progress run for
the project, validates that the config + suite haven't changed since,
and continues from where it left off. Already-completed calls
(including ones that errored at the LLM layer) are skipped.

A config or suite change between attempts aborts the resume — start
a fresh run instead.

## How do I push a run to hosted EvalShift?

Sign in through the hosted web app, then approve CLI login in the browser:

```bash
evalshift login          # defaults to https://api.evalshift.dev
evalshift whoami
```

Set `project: org-slug/project-slug` in `evalshift.yaml` or pass
`--project org-slug/project-slug`, then run:

```bash
evalshift all --yes --push
```

See [Hosted alpha](hosted.md) for credential precedence, bundle contents,
and troubleshooting.

## Why does the `max cost` row in `evalshift all` look so much higher than the actual `Total cost` in the report?

The pre-flight figure is a **worst-case ceiling**, not a forecast.
`evalshift all` (and `evalshift run`) prices each call as if the model
emits its full registry `default_max_tokens` of completion (4096).
Real completions — especially agent-style runs that produce short
tool-call decisions — are usually far shorter than the cap, so the
actual `Total cost` in the report typically lands well below the
displayed ceiling.

The figure is conservative on purpose: the cost-confirmation prompt
(triggered above $10) wants to over-warn rather than under-warn. If
you see `≤ $0.17` and the run actually cost $0.03, that's expected.

## How do I lower the cost of a run?

* **Set the SQLite cache to be on** (it's the default). A re-run of
  the exact same configuration is free.
* **Use cheaper models.** The model registry assigns sensible
  defaults but you can drop everything to flash/mini/haiku tier.
* **Skip the LLM judge.** Structural and semantic evaluators are
  much cheaper. Drop the `evaluators.llm_judge` section to disable
  the judge entirely.
* **Cap with `max_cost_usd`** in `defaults` (a future tightening
  will hard-enforce; currently a soft ceiling).

## What does "passthrough" mean next to my model id in `evalshift test-call`?

It means the id you passed isn't in EvalShift's curated registry.
The id is sent to LiteLLM as-is (with provider prefix inferred from
the prefix). If LiteLLM doesn't know the model either, you'll get a
clean error from the provider when you make the call.

## Why does my Cohen's d show as 0 with severity "none"?

Two common causes:

1. **Every delta is identical.** When the variance in deltas is near
   zero, the test is skipped and severity defaults to `none`.
2. **Your sample size is too small.** With `n < 5` the test is
   skipped and severity is `insufficient` (not `none`).

If you expected a real signal, double-check your evaluator output
range — many "all the same" cases are evaluators returning a constant.

## Why is the migration-policy verdict `inconclusive`?

Three common causes, all by design:

1. **Every configured evaluator is advisory** (`blocking: false` — the
   fresh `evalshift init` state), so nothing gates quality. Promote
   evaluators to blocking as your suite grows. The cost and latency
   budgets still apply — they read the run's calls, so a breach there
   reports `fail`, not `inconclusive`.
2. **A rate budget was breached but the 95% Wilson interval can't
   confirm it** at this suite size — grow the suite.
3. **All comparisons were `insufficient`** (n < 5).

`analyze` and `all` print the specific reason and the recommended fix
under the verdict line, and record them in `migration_decision.json`
(`reason` / `recommendations`).

## Does EvalShift work with LangChain agents?

You don't need LangChain to use EvalShift. Each golden-suite example
carries its own toolset — a `toolset_ref` pointing at a sidecar, or an
inline `tools` list (Anthropic-shape or OpenAI-shape, either works). The
usual path is automatic: capture your agent with the `evalshift-sdk` (works
regardless of framework — LangChain, a manual loop, anything) and
`capture promote` / `capture sync` record the toolset it was actually
offered. Writing a suite by hand instead? Inline `tools:` directly on each
example — see [Agent migrations](agents.md#suite-ground-truth). If your
tools are defined as LangChain `Tool` objects, export them to JSON Schema
once for that inline list. Framework-side agent timelines can also be
scored via [external traces](traces.md).

## Does EvalShift evaluate multi-turn conversations?

Yes — one suite example per turn, each carrying a recorded `history`
prefix that is replayed teacher-forced: both models see byte-identical
context, and only the current turn's output is compared. See
[Multi-turn conversations](conversations.md). Full-conversation
re-driving (feeding the candidate's own replies into later turns) is
deliberately not supported — it breaks the paired-comparison contract.

## Where is `evalshift validate` / `evalshift test-call` in `--help`?

They're hidden — they're development aids, not part of the supported
user pipeline. Both still run if you invoke them by name.
