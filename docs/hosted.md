# Hosted alpha

Hosted EvalShift is optional. Local commands stay local until you explicitly
package and push a completed run.

Use hosted when you want shared run history, web viewing, diff links, and
GitHub pull request comments.

## Login

Create a hosted API token in the web app, then store it locally:

```bash
evalshift login --token <hosted-api-token> --host <hosted-api-url>
evalshift whoami
```

`login` verifies the token with `GET /me`, then writes credentials to
`~/.evalshift/credentials` with owner-only file permissions.

Credential precedence is:

1. Explicit CLI flags: `--host` and `--token`.
2. Environment variables: `EVALSHIFT_HOST` and `EVALSHIFT_TOKEN`.
3. Stored credentials: `~/.evalshift/credentials`.

The default host is local development (`http://localhost:8000`). Use
`--host` or `EVALSHIFT_HOST` for staging or production.

## Configure a project

Hosted push needs a project path in `org-slug/project-slug` form. Put it in
`evalshift.yaml`:

```yaml
project: acme/model-migration
thresholds:
  pass_rate_min: 0.95
  regression_max: 0
```

Or pass it for a single command:

```bash
evalshift push <run-id> --project acme/model-migration
```

`thresholds` are hosted project settings. They are not part of the frozen run
bundle manifest. If you push thresholds, the hosted backend requires project
owner rights and returns the canonical thresholds for that project.

## Bundle and push

`bundle` packages a completed local run from `.evalshift/runs/<run-id>/`:

```bash
evalshift bundle <run-id>
```

The output is `.evalshift/runs/<run-id>/run_bundle.json.gz`.

Push a local run:

```bash
evalshift push <run-id>
```

Push a prebuilt bundle:

```bash
evalshift push --bundle .evalshift/runs/<run-id>/run_bundle.json.gz
```

Run the whole local pipeline, then push:

```bash
evalshift all --yes --push
```

On success, `push` prints only the hosted run URL. If the backend already has
an available run with the same id, the CLI treats that as idempotent success
and prints the existing URL.

## Project auto-create

If the hosted project is missing, `push` can create it automatically when:

- The token can see the org.
- The token has permission to create projects in that org.
- `--create-project` is enabled, which is the default.

Disable auto-create when you want strict CI behavior:

```bash
evalshift push <run-id> --no-create-project
```

Project-scoped tokens cannot auto-create projects.

## Privacy model

Local run and report commands do not upload prompts, suites, outputs, scores,
or reports to EvalShift-operated services.

`bundle` creates a local archive. `push` and `all --push` upload the completed
bundle. That bundle contains the run manifest, raw examples and outputs,
scores, analysis, and `report.html`. Review your suite and report contents
before pushing sensitive data.

Do not put hosted API tokens or provider API keys in config files. Use the
credential file locally and repository secrets in CI.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing hosted token` | No flag, env var, or credentials file token is available. | Run `evalshift login --token <hosted-api-token> --host <hosted-api-url>` or set `EVALSHIFT_TOKEN`. |
| `host uses plain http` warning | The host is non-local HTTP. | Use HTTPS for non-local hosts. |
| `hosted project is required` | No `project` in config and no `--project` flag. | Add `project: org/project` or pass `--project`. |
| `project was not found` | The project does not exist and auto-create is disabled or not allowed. | Ask an owner to create it, use an org-scoped owner token, or enable auto-create. |
| Threshold warning | Local `thresholds` differ from hosted canonical thresholds. | Pull the current project thresholds from the web app or ask an owner to sync them. |
