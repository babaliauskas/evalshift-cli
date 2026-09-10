# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.14.1] - 2026-09-10

### Fixed

- `capture sync` now pairs a history tool result that was recorded without a
  `tool_call_id` with the preceding assistant turn's next unanswered tool
  call, in order, instead of assigning it a positional `_pos<N>` id. Gemini
  puts no ids on function responses, so every promoted Gemini conversation
  with a tool round carried an id nothing could match and LiteLLM rejected
  the replay of every later turn on both models (`Missing corresponding tool
  call for tool response message`) before it reached the provider. Re-run
  `evalshift capture sync` to regenerate affected suites. A result with no
  preceding call to answer still gets `_pos<N>` with a warning.
- LiteLLM's "Give Feedback / Get Help" banner, which the library prints on
  every failed call outside any logger, no longer floods the pipeline output.

## [0.14.0] - 2026-09-10

### Added

- `defaults.samples_per_example` (default `1`, max `20`): send every
  `(prompt, example)` to each model N times. Each sample is its own live call
  (`raw.jsonl` rows carry `sample_index`; the cache key includes it, so a
  single-sample run keeps every existing cache entry and resume skips per
  sample). `evaluate` scores source sample *i* against target sample *i* and
  folds the samples of one example into one `scores.jsonl` row: the mean
  `source_score` / `target_score` / `delta` over the samples that scored, the
  per-sample lists and the population `delta_variance` under
  `metadata.samples`, and the explanation prefixed `mean of k samples`. The
  paired tests still run over examples, so `n` is unchanged. The report shows
  an `N samples per example` pill, `report.json` carries
  `samples_per_example`, the non-determinism banner suggests the setting on a
  single-sample run, and example rows (report, bundle, insights, `inspect`)
  show sample 0. The cost estimate and pre-flight call count multiply by N.
- Teacher-forced multi-round replay. `capture promote` / `capture sync
  --rounds all` now carry the recorded tool results on the promoted example as
  `tool_result_fixtures` — one inner list per covered round, positionally
  aligned with `expected_tool_rounds`, each entry `{tool_name, result, error}`
  — pairing each round's calls with that round's `tool_result` events by
  `call_id` first and by tool name within the round second. `evalshift run`
  then replays such an example round by round: round *k* is sent the prompt
  (and any `history` prefix) followed by the *recorded* rounds `1..k-1` as
  assistant tool calls and `tool` results, so source, target and the recording
  all see identical context; the candidate's own calls are never fed back. The
  replay covers every round the fixtures cover plus the round after it, which
  — when every tool round is covered — is the answer round, where the recorded
  agent called nothing and produced its final text. Fixture coverage stops at
  the first round with a call that has no recorded result, with a warning
  naming the rounds the replay will cover. One `raw.jsonl` row per example per
  model as before: tokens, cost and latency summed, `text` the last round's
  answer, the `ToolTrace` carrying `round_count` and a `round_index` on every
  call. A model error in round *k* fails the example as `round k/n: …`.
  `tool_selection`, `tool_arguments` and `tool_trace_structure` score each
  round against its own ground truth (conformance against
  `expected_tool_rounds[k]`, "called nothing" for the answer round; divergence
  and argument pairing within a round) and record the mean over replayed
  rounds with per-round detail under `metadata.rounds`, so
  `max_tool_divergence` counts an example as diverged when any round diverged.
  The HTML report and `report.json` show one line per round in the tools
  column and prefix trace-diff items with `Round k:`; bundle trace events now
  carry the real `round`. The cost pre-flight counts one call per replayed
  round. Suites without the field (every suite written before it, and every
  `--rounds first` promotion) replay single-shot exactly as before.
- `cache_key` accepts a `round_index` (hashed only when set), so tool-call
  caching can land later without a cache migration; the tool path still
  bypasses the cache.

- Runs now record which generation parameters the models cannot honour, instead
  of `drop_params: True` making them vanish. A promoted capture can pin the
  generation config its original call used (`temperature`, `top_p`,
  `response_format` / `response_mime_type` / `response_schema`, `max_tokens` /
  `max_output_tokens`, `tool_choice` / `tool_config`, `parallel_tool_calls`),
  and the replay sends it — but LiteLLM's `drop_params` lets a model that never
  accepted one of those answer anyway, minus the constraint, so the arm
  measured a model change *plus* a missing constraint with nothing saying so.
  At run start EvalShift now asks LiteLLM (`models.capabilities.unsupported_params`,
  the generalisation of the existing `honors_temperature` probe) which of the
  parameters the suite actually recorded each arm supports, mapping provider
  spellings to their OpenAI names first. Anything a model positively lacks
  lands in `state.json` under `dropped_params` (`model id → [param, …]`), is
  logged once per (model, parameter) at `WARNING` rather than per call, reaches
  `report.json` as `dropped_params`, and renders as a **Constraints not
  honoured** banner beside the sampling banner in the HTML report. Uncertainty
  reads as "supported" — an exception, a `None`, or an empty answer records
  nothing — on the same reasoning as the sampling probe: a false banner on
  every report costs more than one missed warning. `temperature` stays with
  `non_deterministic_models`, which owns its own banner and probes
  unconditionally. Calls are unaffected: `drop_params` is still on, so nothing
  that used to succeed now fails.
  Alongside the probe, a small hard-coded table
  (`models.capabilities._KNOWN_LITELLM_GAPS`) covers what the probe cannot see:
  parameters LiteLLM *reports as supported* and then discards while building
  the provider's request body. Verified against litellm 1.100.0, that is two
  Gemini cases — `parallel_tool_calls` (filtered out against
  `GenerationConfig`'s fields) and a tool's `strict` flag (dropped by
  `_map_function`, since Gemini function declarations have no strict mode) —
  and both now land in `dropped_params` for a Gemini arm whose suite recorded
  them, where previously they only produced a per-dispatch log line. The tool
  flag is recorded under the pseudo-parameter name **`tools.strict`**, because
  it is a field on the `tools` array rather than a generation parameter. Probe
  and table merge into one sorted list per model. The two dispatch-time
  warnings in `models/client.py` are gone, since the run-start record now says
  the same thing and reaches the report and the policy; a `tool_choice` on a
  tool-less example is still stripped with a warning at dispatch and is
  deliberately *not* recorded as a dropped parameter — it is a fact about the
  suite, not about either target.
- `migration_policy.fail_on_dropped_params` (bool, default `false`) turns that
  record into a gate: when set and `dropped_params` is non-empty, the verdict
  is `fail` with a reason naming each model and parameter, whatever the scores
  said. For suites where the constraint *is* the contract — captures that
  pinned `response_format` measure nothing useful against a target that will
  not produce structured output. Top-level only (a model either accepts a
  parameter or does not, which no slice can vary), and runs recorded before
  `dropped_params` existed are never failed by it.
- Trace models accept `requested_tool_calls` on a `model_call` event — the tool
  calls the model asked for *in its response*, as
  `{name, arguments, call_id}` entries. It sits alongside the two notions that
  already existed and is none of them: `toolset_ref` / `tools_offered` is what
  was **offered** to the model, the `tool_call` / `tool_result` events are what
  the app **executed**, and this is what was **requested**. `null` (the default)
  means the trace predates the field; `[]` means the model requested no tools.
  The capture reader gates on the schema *major* only, so a capture written at
  the SDK's new `2.1.0` schema loads unchanged. Trace models are `extra="forbid"`,
  so the CLI has to accept the field before any SDK writes it.
- `capture promote` / `capture sync` now use those model-requested calls as the
  tool-call ground truth whenever a capture carries them, and record which
  yardstick a case used as `promotion_source` (`"requested"` | `"executed"`,
  default `"executed"`) on the promoted-case file. The executed `tool_call`
  events have already passed through the application — its filtering, retries,
  re-ordering, and its own function signatures — so they show what the *app*
  did, while a golden case has to state what a *model* should produce. On the
  requested path each `model_call` is one round (rounds that requested nothing
  are dropped, exactly as tool-less executed rounds are) and arguments are
  carried verbatim: wrapper unwrapping never runs on them, because nothing
  stands between the model and its own requested call. Every `model_call` in a
  run has to carry the field for the capture to be scored against requested
  calls — pass `[]` for a round in which the model requested no tools, since
  `null` means *not recorded* rather than *nothing requested*. Fallback to the
  executed calls is silent for a capture that predates the field, and warns for
  one where only *some* `model_call` events carry it (the whole capture falls
  back rather than mixing yardsticks). When requested and executed calls
  disagree, the requested ones win and promotion warns, naming the tools on
  both sides.
- Replay now carries the tool-choice constraints production used, instead of
  debug-logging and dropping them. A recorded `tool_choice` reaches the target
  in whichever of the three spellings the capture holds — an OpenAI string
  (`"auto"` / `"none"` / `"required"`) or object, an Anthropic object
  (`{"type": "auto"|"any"|"tool", "name"?, "disable_parallel_tool_use"?}`), or
  Gemini's `tool_config` (`function_calling_config.mode`, with
  `allowed_function_names`) — all normalised to one OpenAI-style intent plus a
  `parallel_tool_calls` bool, which LiteLLM then maps onto each provider's own
  shape (Anthropic's `tool_choice` object carrying `disable_parallel_tool_use`,
  Gemini's `toolConfig`). Normalising rather than passing through is what lets a
  capture recorded against one provider replay meaningfully against a target on
  another. An Anthropic `disable_parallel_tool_use: true` becomes
  `parallel_tool_calls: false`; an explicit top-level `parallel_tool_calls`
  wins over the inferred one.
- `ToolSpec` gained `strict`, so a toolset sidecar or inline `tools` entry
  carrying `strict: true` (top-level in the canonical/Anthropic shape,
  `function.strict` in the OpenAI shape) is replayed instead of being rejected
  as an unknown key. Both serialisers emit it only when set, so non-strict tools
  serialise byte-identically to before and toolset fingerprints are unchanged.
- Generation-config keys the runner cannot translate are now logged at
  **warning** rather than debug — once per distinct key set, so a whole-suite
  replay says it once. Constraints a target provider genuinely cannot express
  are named rather than dropped in silence: a Gemini target warns for
  `parallel_tool_calls` (`generateContent` has no such switch) and for a tool's
  `strict` flag (Gemini function declarations have no strict mode), once per
  model and key. A `tool_choice` that reaches an example with no toolset is
  dropped with a warning — there is nothing to constrain.
- `doctor` row `evalshift-sdk`: reports the SDK version the `evalshift` import
  name resolves to in this environment; `warn` (never a failure) when the SDK
  is missing, fails to import, or is shadowed by an older CLI's leftover files
  or a local `evalshift/` directory.
- Judge-family warning (`evalshift_cli.models.family`): when an `llm_judge`
  `judge_model` resolves to the same provider as `defaults.source_model` or
  `target_model`, `doctor` prints a warn-level `judge family` row (one per
  distinct judge; `ok` "from a third family" when none overlaps; no row when
  either arm is unset) and `validate` prints the same line after its success
  line — LLM judges prefer their own relatives' output (self-preference bias),
  so verdicts lean toward that arm. Advisory, never a failure: `init` scaffolds
  a same-provider judge on purpose. The report repeats the note as a third
  banner, only for judges that actually contributed `scores.jsonl` rows, and
  `report.json` carries it as `judge_family_overlap`. Provider `other` (an id
  the registry cannot place) never matches.
- CI pin-drift check (`evalshift_cli.utils.ci_pin`): `capture sync`, `init` (without
  `--ci`), `doctor` (new `ci pin` row), and `validate` now parse
  `.github/workflows/*.yml` for `babaliauskas/evalshift-action` steps and warn
  when the `evalshift-version` pin is older than the local CLI (`stale`),
  absent (`unpinned` — the action default may lag), or newer than the local CLI
  (`ahead`), printing the exact `evalshift-version: "<v>"` line to set. Advisory
  only: the CLI never edits a workflow and exit codes are unchanged. Rationale:
  `extra="forbid"` config means the reader in CI must be at least as new as the
  writer locally.
- `packaging>=23` is now a declared dependency (version comparison).
- `examples/capture-first/` — the first example to use the managed `suites:`
  block. It checks in the whole capture-first flow: an SDK-instrumented agent,
  the three captures and the toolset sidecar it recorded, the promoted cases and
  `golden.jsonl` that `evalshift capture sync` wrote, and the unedited
  `evalshift.yaml` from `init` with sync's derived `tool_selection` /
  `tool_arguments` block filled in. Its own `.gitignore` shows how a project
  commits `.evalshift/suites/` and `.evalshift/toolsets/` while keeping runs and
  the cache ignored.
- Promoted case files now record what the captured run cost: `cost_usd` (the
  run's `model_call` events summed) and `cost_source`. The SDK never prices
  anything — a `model_call`'s `cost_usd` is `0.0` unless the app's own
  instrumentation set it, and the provider client wrappers record tokens but no
  cost by design — so `capture promote` / `capture sync` now price each event
  that recorded tokens but no cost from litellm's price table for its own
  `model_id`, tagging the case `cost_source: "estimated"`. A recorded non-zero
  cost is kept as recorded (`"recorded"`), never re-estimated. A model litellm
  does not price — local / self-hosted, the normal case for open-source models
  — stays at `0.0` with no tag and no warning; the pricer is never called for
  it, so `capture sync` neither prints litellm's provider banner nor opens a
  socket to a local Ollama daemon. Existing case files parse unchanged
  (both fields default), and the `golden.jsonl` example never carries the
  figure.

### Fixed

- LiteLLM warnings no longer print in the middle of the `evalshift all`
  pipeline block (and again in the deferred-warnings section) on
  litellm >= 1.100. That release routes records below WARNING to `sys.stdout`
  by re-pointing its handler's stream per record; because only `sys.stderr`
  and `sys.__stderr__` counted as console streams, a single INFO record left
  the handler unrecognised and `deferred_console_warnings()` stopped
  detaching it. `sys.stdout`/`sys.__stdout__` now count too.

### Changed

- `--rounds all` no longer flattens every recorded round into `expected_tools`.
  `expected_tools` is now `expected_tool_rounds[0]` under both settings, and
  `--rounds all` means teacher-forced multi-round replay instead (see *Added*).
  The flattened list was only ever right for comparing against an externally
  produced multi-round trace, which the `agent_trace` evaluator does from
  imported traces. `--tool-count` under `--rounds all` pins the total over the
  rounds the replay reaches rather than over every recorded round.
- **Breaking (packaging):** the CLI's import package is now `evalshift_cli`.
  The distribution (`evalshift`) and the `evalshift` command are unchanged.
  The import name `evalshift` belongs to the capture SDK, which the CLI now
  declares as a dependency (`evalshift-sdk>=0.3.0`), so both install into one
  environment and `pip install evalshift` brings the SDK with it — the
  two-virtualenv rule is gone from every install page. What a user can
  notice: `python -m evalshift` is now `python -m evalshift_cli`, and scripts
  that imported CLI internals (`from evalshift.models.client import …`) must
  import from `evalshift_cli`. No shim is possible — shipping any `evalshift/`
  file would recreate the collision — so this ships as a minor bump (0.14.0).
  Design: `docs/superpowers/specs/2026-09-09-namespace-collision-design.md`.
- Docs: the capture guides (`README.md`, `DOCS.md`, `docs/sdk.md`,
  `docs/getting-started.md`, `llms-full.txt`) now cover the SDK 0.4.0 provider
  client wrappers (`wrap_openai` / `wrap_anthropic` / `wrap_genai`), the
  `requested_tool_calls` promotion path and `capture sync --rounds`, and no
  longer tell users to install `evalshift-sdk` in a separate venv — the CLI
  depends on it since the `evalshift_cli` rename. The capture-first example
  README explains why its cases are `promotion_source: executed`.
- Documented the config version policy: `version: 1` bumps only for breaking
  changes; additive fields ride on the CLI version and the CI pin check is the
  mechanism that keeps CI's reader at least as new as the local writer.

## [0.13.1] - 2026-08-28

### Changed

- `evalshift init --ci` now scaffolds a production-shaped GitHub Actions
  workflow instead of a single-suite example: dynamic suite discovery under
  `.evalshift/suites/` (a project with no suites yet skips green), one matrix
  job per suite, a single `evalshift gate` join check for branch protection,
  `fail-on: policy` (the action's real default), `evalshift-version` pinned to
  the scaffolding CLI, a provider API key matching `--provider`, PR-only
  run cancellation so base-branch baselines survive, and the full setup
  checklist documented in the generated file's header.

## [0.13.0] - 2026-08-28

Initial public release.
