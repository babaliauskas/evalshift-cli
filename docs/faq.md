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

Create a hosted API token in the web app, then:

```bash
evalshift login --token <hosted-api-token> --host <hosted-api-url>
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
emits its full `default_max_tokens` of completion (1024 for most
models in the registry). Real completions — especially agent-style
runs that produce short tool-call decisions — are usually 5–10× shorter
than the cap, so the actual `Total cost` in the report typically lands
well below the displayed ceiling.

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

## Does EvalShift work with LangChain agents?

You don't need LangChain to use EvalShift. EvalShift reads tool
definitions from a yaml/json file (Anthropic-shape or OpenAI-shape —
either works). If your tools are defined as LangChain `Tool` objects,
export them to JSON Schema once and point `tools_path` at the result.

LangChain `AgentExecutor` *auto-detection* (read tools straight from
the agent code) is on the roadmap.

## Does EvalShift evaluate multi-turn conversations?

Not yet. The pipeline scores one assistant turn against another for
the same user input. Multi-turn evaluation (where each turn might
produce its own tool calls) is on the roadmap.

## Where is `evalshift validate` / `evalshift test-call` in `--help`?

They're hidden — they're development aids, not part of the supported
user pipeline. Both still run if you invoke them by name.
