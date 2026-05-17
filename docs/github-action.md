# GitHub Action

The reusable EvalShift GitHub Action runs the CLI in CI, pushes the run to
hosted EvalShift, compares it with a compatible base-branch run, and updates
the pull request with a regression summary.

## Scaffold the workflow

From a new EvalShift project:

```bash
evalshift init --ci
```

This writes `.github/workflows/evalshift.yml` with:

```yaml
permissions:
  contents: read
  pull-requests: write
  issues: write
  statuses: write

jobs:
  evalshift:
    runs-on: ubuntu-latest
    env:
      EVALSHIFT_NONINTERACTIVE: "1"
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    steps:
      - uses: actions/checkout@v4

      - name: EvalShift hosted regression check
        uses: babaliauskas/evalshift-action@v0
        with:
          token: ${{ secrets.EVALSHIFT_TOKEN }}
          fail-on: regression
```

Adjust provider secrets to match the models in your config.

`init` refuses to overwrite existing files. If you already have an
`evalshift.yaml`, copy the workflow shape below into
`.github/workflows/evalshift.yml` instead of overwriting your project.

## Required secrets

- `EVALSHIFT_TOKEN`: hosted API token from the web app. Prefer a project-scoped
  token for CI when the project already exists.
- Provider API keys used by the source, target, judge, or embedding models.

Do not hard-code tokens in workflow YAML. Keep them in GitHub repository
secrets.

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
| `suite` | `golden.jsonl` | Suite path. |
| `fail-on` | `regression` | Hosted regression gate mode. |
| `create-project` | `true` | Allow project auto-create when permissions allow it. |
| `comment` | `true` | Post or update the PR comment on pull requests. |

See the action repository README for the full input list.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No PR comment | Missing `pull-requests: write` or `issues: write`. | Add both permissions to the workflow. |
| No commit status | Missing `statuses: write`. | Add the permission. |
| Push fails with missing project | Project does not exist and token cannot auto-create it. | Create the project in the web app or use an org-scoped owner token for first setup. |
| No compatible baseline | Base branch has not pushed a compatible run yet. | Merge or run EvalShift on the base branch once. |
