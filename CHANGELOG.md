# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
