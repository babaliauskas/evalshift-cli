# GitHub Action

The reusable EvalShift GitHub Action runs the CLI in CI, pushes the run to
hosted EvalShift, compares it with a compatible base-branch run, and updates
the pull request with a regression summary.

## Scaffold the workflow

From a new EvalShift project:

```bash
evalshift init --ci      # capture-first project
```

This writes `.github/workflows/evalshift.yml` — a production-shaped,
self-documenting workflow (its header comment carries the full setup
checklist) with three jobs:

- **`discover`** — lists every committed suite under
  `.evalshift/suites/*/golden.jsonl`. A suite added by `evalshift capture
  sync` is evaluated on the next run with no workflow edit; a project with no
  suites yet skips green with a notice instead of failing.
- **`eval <suite>`** — a matrix job per discovered suite (the action
  evaluates one suite per invocation). Runs `fail-on: policy`, so the verdict
  is the hosted re-score against the `migration_policy` block in
  `evalshift.yaml`. `evalshift-version` is pinned to the CLI that scaffolded
  the project: the CLI that *reads* the config in CI must be at least as new
  as the CLI that *wrote* it locally (`extra: forbid` rejects newer keys), and
  the CLI warns when the pin falls behind — see [Pin drift](#pin-drift). `max-parallel` defaults to 1 —
  raise it toward your hosted plan's in-flight-run ceiling (Free 1, Pro 5,
  Team 10). The PR comment is posted by the first matrix job only: the
  comment marker is a constant, so multiple suites would overwrite one
  another's summary.
- **`evalshift gate`** — the join job to require in branch protection. It
  fails if any suite failed and passes when evaluation was skipped (fork PR
  — no secret access, no suites committed, or `EVALSHIFT_TOKEN` not set
  yet). Require this check, not the per-suite jobs (dynamic names) and not
  the `evalshift/regression` commit status (with several suites the last
  writer wins that status).

The workflow keys off `${{ secrets.<PROVIDER>_API_KEY }}` for the provider
chosen at `init` time; add further keys under the eval job's `env:` if your
judge or embedding models live in another family. Suites must be committed
for CI to see them — keep runtime data ignored and un-ignore just the suites:

```gitignore
.evalshift/*
!.evalshift/suites/
!.evalshift/toolsets/
```

Runs on pushes to the main branch create the base-branch baselines pull
requests diff against; the workflow's `concurrency` block therefore cancels
superseded runs on PRs only, never on main.

The scaffold refuses to overwrite existing files. If you already have
an `evalshift.yaml`, run `evalshift init --ci --directory` somewhere scratch
and copy `.github/workflows/evalshift.yml` across instead of overwriting
your project.

## Required secrets

- `EVALSHIFT_TOKEN`: a **service account key** from the web app (Settings → API
  tokens → Service accounts), scoped to `run:create` + `run:read`. Not a personal
  token — that one dies with its owner's membership and takes the pipeline with
  it. A scoped key cannot auto-create the hosted project (`project:create` is
  owner-only), so create the project once in the web app and set
  `create-project: false`.
- Provider API keys used by the source, target, judge, or embedding models.

Do not hard-code tokens in workflow YAML. Keep them in GitHub encrypted secrets
— repository or, better for production repos, environment secrets. Never expose
them to a `pull_request_target` workflow: that trigger runs the base repo's
workflow with secrets in scope against fork code.

Rotate on a schedule: rotate the key in the web app (the old one keeps working
for a 24-hour grace window), update the GitHub secret, confirm a green run, then
let the old key expire.

## Project config

The Action expects the CLI project to know where hosted runs should land:

```yaml
project: acme/model-migration
thresholds:
  pass_rate_min: 0.95
```

The default workflow runs the local pipeline, finds the latest run id, and
pushes that run to hosted EvalShift.

The action passes the hosted token through environment variables so command
output does not expose it.

## Baselines and PR comments

On pull requests, the Action:

- Pushes the candidate run.
- Looks for the latest compatible run on the base branch.
- Fetches the hosted diff if a baseline exists.
- Creates or updates one PR comment marked by EvalShift.
- Sets commit status `evalshift/regression`.

If no compatible baseline exists, the comment explains that the run was pushed
but there is no baseline yet. Gating passes in that case.

## `fail-on` modes

| Mode | Behavior |
| --- | --- |
| `policy` (default) | Ask hosted EvalShift for the migration-policy verdict — the run re-scored against the `migration_policy` limits in `evalshift.yaml`. `fail` fails; `pass`/`conditional_pass` pass. If the policy check is unreachable, falls back to `regression` gating and says so. |
| `never` | Do not fail the workflow for hosted regressions. |
| `regression` | Fail when the hosted diff reports one or more regressed examples. |
| `any-slice-regression` | Fail when any slice pass rate moves down. |

The Action can still fail for setup errors, provider auth errors, invalid
config, upload failures, or finalize failures.

## Inputs

Common inputs:

| Input | Default | Description |
| --- | --- | --- |
| `token` | required | Hosted EvalShift API token. |
| `host` | hosted default | Hosted API base URL. |
| `config` | `evalshift.yaml` | Config path. |
| `suite` | `golden.jsonl` | Suite path (one suite per invocation). |
| `fail-on` | `policy` | Gate mode — see the table above. |
| `evalshift-version` | action default (may lag) | Exact CLI version installed from PyPI. Always set it: it must be at least as new as the CLI that writes your `evalshift.yaml` (reader ≥ writer). `init --ci` pins it to the scaffolding CLI. |
| `create-project` | `true` | Allow project auto-create when permissions allow it. |
| `comment` | `true` | Post or update the PR comment on pull requests. |

See the action repository README for the full input list.

## Pin drift

`evalshift.yaml` rejects unknown keys (`extra: forbid`), so a config written by a
newer CLI can fail outright on an older one. The rule is **reader ≥ writer**:
the version the action installs in CI must be at least as new as the CLI you
run `capture sync` and `init` with locally. Upgrading locally without bumping
`evalshift-version` is the common way to break this.

The CLI checks for it wherever it writes or validates config — `capture sync`,
`init` (without `--ci`, next to a workflow it did not write; `init --ci` pins
the scaffolding CLI itself and does not warn about the file it just wrote),
`doctor` (a `ci pin` row), and `validate`. It parses every `.github/workflows/*.yml` for
`babaliauskas/evalshift-action` steps and compares their `evalshift-version`
with its own:

```text
⚠ CI installs evalshift 0.12.1 (.github/workflows/evalshift.yml, job evalshift) but the local CLI is 0.13.1 — an older CLI rejects config keys a newer one writes.
  Fix: set `evalshift-version: "0.13.1"` on the babaliauskas/evalshift-action step.
```

| Status | Trigger | Fix |
| --- | --- | --- |
| stale | a literal pin is older than the local CLI | set `evalshift-version: "<local>"` |
| unpinned | a step has no `evalshift-version` (action default applies and may lag) | add the pin |
| ahead | every pin is newer than the local CLI | `pip install -U evalshift` |

Equal pins, `${{ }}` expressions, unparseable values, and an editable install
without metadata are silent. The check is advisory only: the CLI never edits
your workflow and never changes an exit code, and in CI it is a no-op by
construction — the running CLI *is* the pin. Config `version: 1` is not
bumped for additive fields; see the
[config version policy](configuration.md#config-version-policy).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No PR comment | Missing `pull-requests: write` or `issues: write`. | Add both permissions to the workflow. |
| No commit status | Missing `statuses: write`. | Add the permission. |
| Push fails with missing project | Project does not exist and token cannot auto-create it. | Create the project in the web app or use an org-scoped owner token for first setup. |
| No compatible baseline | Base branch has not pushed a compatible run yet. | Merge or run EvalShift on the base branch once. |
