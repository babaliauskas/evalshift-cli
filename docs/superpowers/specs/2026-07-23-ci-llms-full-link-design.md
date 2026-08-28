# CI llms-full.txt documentation link — design

Date: 2026-07-23
Status: approved

## Goal

`evalshift init` currently advertises two hosted llms.txt references to coding agents
(`cli-llms-full.txt`, `sdk-llms-full.txt`). Add a third for the EvalShift GitHub Action:
`https://www.evalshift.dev/ci-llms-full.txt`, following the exact same pattern end to end —
source doc in the owning repo, hosted copy in the web app, link wired by `init`.

## Changes by repo

### 1. evalshift-action (source of truth)

- Author `llms-full.txt` at the repo root: a dense single-file reference for AI tools.
  - Header: name, canonical hosted copy URL (`https://evalshift.dev/ci-llms-full.txt`),
    pinned CLI version, marketplace usage (`uses: babaliauskas/evalshift-action@v0`).
  - What the action does, step by step: setup-python → pip install pinned `evalshift` →
    `scripts/evalshift_action.py` (run pipeline, push hosted run, PR comment, gate).
  - Full inputs table (from `action.yml`): `token`, `host`, `config`, `suite`,
    `evalshift-version`, `python-version`, `fail-on`, `branch`, `base-branch`,
    `create-project`, `comment`, `github-token`.
  - Full outputs table: `run_url`, `diff_url`, `run_id`, `regression_count`, `conclusion`.
  - `fail-on` gating semantics: `never` / `regression` / `any-slice-regression`.
  - Branch auto-detection behavior and overrides.
  - Copy-paste PR-gate workflow example with `EVALSHIFT_TOKEN` secret.
  - Cross-links to `cli-llms-full.txt` and `sdk-llms-full.txt`.
- Content sourced from `action.yml`, `README.md`, `scripts/evalshift_action.py`. No behavior
  claims not backed by those files.

### 2. evalshift-client (hosting)

- Copy the authored doc to `public/ci-llms-full.txt` (same mechanism as the cli/sdk copies).
- Add a third entry to `public/llms.txt` under "For AI coding tools":
  `[Complete GitHub Action (CI) reference](https://evalshift.dev/ci-llms-full.txt)`.

### 3. evalshift-cli (this repo)

- `src/evalshift/cli/commands/_agents.py`:
  - Add `CI_DOCS_URL: Final = "https://www.evalshift.dev/ci-llms-full.txt"`.
  - Render as a third bullet ("EvalShift GitHub Action (CI)") in both documentation lists:
    the `AGENT_INSTRUCTIONS` `## Documentation` section and `_render_pointer_block`.
  - Extend the fetch-guard sentence in both places to cover CI / GitHub Action /
    workflow tasks.
  - Export `CI_DOCS_URL` in `__all__`.
- Tests (`tests/unit/test_agents.py`): TDD — assert `CI_DOCS_URL` appears in
  `AGENT_INSTRUCTIONS` and in the pointer block, alongside the existing cli/sdk assertions.
- Docs travel with behavior:
  - `DOCS.md`: `--wire-agents` bullet mentions the three hosted references.
  - `llms-full.txt` (repo root): mention CI reference URL where sdk reference is mentioned.
  - No `docs/` page changes: no per-topic page documents the agent wiring or the hosted
    llms references today (verified by grep), and `docs/github-action.md` already covers
    the action itself.
  - `CHANGELOG.md` under `## [Unreleased]`.

## Non-goals

- No changes to action behavior, no new init flags, no renaming of existing URLs.
- No automation for syncing `evalshift-action/llms-full.txt` → `evalshift-client/public/`
  (matches current manual pattern for cli/sdk).

## Risks

- Link 404s until the client repo deploys — acceptable, all three repos updated together.
- Doc drift between action repo and hosted copy — same accepted risk as existing pattern.
