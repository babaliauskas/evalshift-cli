# FAQ

## Does AIMigrate send my prompts to your servers?

**No.** Every API call AIMigrate makes goes directly from your machine
to the LLM provider you configured (Anthropic, OpenAI, Google) using
your own API keys. AIMigrate has no hosted backend in the MVP.

The local SQLite cache at `~/.aimigrate/cache.db` only contains
provider responses for *your* prompts and inputs.

## What happens if a single LLM call fails?

The orchestrator records the error in `raw.jsonl` (with `error="..."`)
and moves on. The run still completes; failed calls are recorded with
a neutral 0.5/0.5 score in the evaluation phase so the analysis can
account for them rather than silently dropping examples.

## What models does AIMigrate support?

Anything LiteLLM supports. The `aimigrate.models.registry` provides
friendly aliases and sane defaults for common models (Claude, GPT,
Gemini), but **the registry is advisory, not gating**. A model id
that isn't in the registry — for example a fresh preview from a
vendor playground — gets passed through to LiteLLM with a
prefix-inferred provider. LiteLLM is the source of truth at call
time.

## Can I resume a run after Ctrl+C / a crash?

Yes. `aimigrate run --resume` finds the latest in-progress run for
the project, validates that the config + suite haven't changed since,
and continues from where it left off. Already-completed calls
(including ones that errored at the LLM layer) are skipped.

A config or suite change between attempts aborts the resume — start
a fresh run instead.

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

## What does "passthrough" mean next to my model id in `aimigrate test-call`?

It means the id you passed isn't in AIMigrate's curated registry.
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

## Where is `aimigrate validate` / `aimigrate test-call` in `--help`?

They're hidden — they're development aids that we plan to relocate
under a hidden `--debug` group at the v0.1.0 cut. Both still run if
you invoke them by name.
