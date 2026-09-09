# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
- `doctor` row `evalshift-sdk`: reports the SDK version the `evalshift` import
  name resolves to in this environment; `warn` (never a failure) when the SDK
  is missing, fails to import, or is shadowed by an older CLI's leftover files
  or a local `evalshift/` directory.
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

### Fixed

- LiteLLM warnings no longer print in the middle of the `evalshift all`
  pipeline block (and again in the deferred-warnings section) on
  litellm >= 1.100. That release routes records below WARNING to `sys.stdout`
  by re-pointing its handler's stream per record; because only `sys.stderr`
  and `sys.__stderr__` counted as console streams, a single INFO record left
  the handler unrecognised and `deferred_console_warnings()` stopped
  detaching it. `sys.stdout`/`sys.__stdout__` now count too.

### Changed

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
