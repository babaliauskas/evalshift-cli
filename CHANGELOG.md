# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI pin-drift check (`evalshift.utils.ci_pin`): `capture sync`, `init` (without
  `--ci`), `doctor` (new `ci pin` row), and `validate` now parse
  `.github/workflows/*.yml` for `babaliauskas/evalshift-action` steps and warn
  when the `evalshift-version` pin is older than the local CLI (`stale`),
  absent (`unpinned` — the action default may lag), or newer than the local CLI
  (`ahead`), printing the exact `evalshift-version: "<v>"` line to set. Advisory
  only: the CLI never edits a workflow and exit codes are unchanged. Rationale:
  `extra="forbid"` config means the reader in CI must be at least as new as the
  writer locally.
- `packaging>=23` is now a declared dependency (version comparison).

### Changed

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
