# Getting started

This page walks you from a fresh shell to a working `evalshift.yaml` in
under a minute.

## 1. Install

EvalShift is a Python 3.14+ package. We recommend [`uv`][uv] for dependency
management, but `pip` works too.

```bash
# uv (recommended)
uv pip install evalshift

# or pip
pip install evalshift
```

Verify the install:

```bash
evalshift --version
```

## 2. Set provider API keys

EvalShift calls Anthropic, OpenAI, and Google directly using your own keys.
Local runs do not send prompts or outputs to an EvalShift-operated server.
Provider responses are cached locally in `~/.evalshift/cache.db`.

Set whichever providers you intend to use:

```bash
export ANTHROPIC_API_KEY=<anthropic-api-key>
export OPENAI_API_KEY=<openai-api-key>
export GOOGLE_API_KEY=<google-api-key>
```

## 3. Scaffold a starter project

```bash
mkdir my-eval
cd my-eval
evalshift init
```

You can also pick a migration profile and scenario pack:

```bash
evalshift init --profile cost-reduction --pack tool-calling-agent
```

Profiles scaffold a `migration_policy` block that powers the verdict in
`analyze`, `all`, and `report`.

This writes four files (a runnable customer-support agent project):

| File              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `evalshift.yaml`  | Run configuration (prompts, evaluators, slices).      |
| `prompts.py`      | Agent system prompt discovered by AST-walking.        |
| `tools.yaml`      | Six tool specs the agent can call.                    |
| `golden.jsonl`    | 40 example suite rows across 5 slices.                |

If any of those files already exist, `init` refuses to clobber them. Pass
`--force` to overwrite, or `--directory my-eval/` to scaffold into a
different folder.

## 4. Verify your environment

```bash
evalshift doctor
```

You'll see a short table:

* Green ✓ — check passes.
* Yellow ✗ — informational warning (e.g. an unset API key, or no
  `evalshift.yaml` here yet). Doctor still exits 0.
* Red ✗ — hard failure (e.g. an `evalshift.yaml` that doesn't validate).
  Doctor exits 1.

If everything is green or yellow, you're ready to run.

## 5. Run the pipeline

The fast path is one command:

```bash
evalshift all --from gemini-2.5-flash --to gemini-3.1-flash-lite-preview --yes --open
```

This runs `doctor → run → evaluate → analyze → report` under a single
Rich Live region with a progress bar for the run stage and a final
verdict block (`✓ significantly better` / `✗ regressed` / `~ no
significant change`). The HTML report opens in your browser.

If you want to drive each stage by hand (useful when re-running just
one stage after fixing config, or in CI where you stage artefacts):

```bash
evalshift run --from gemini-2.5-flash --to gemini-3.1-flash-lite-preview
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

`evalshift all` accepts every flag the underlying commands do
(`--from/--to`, `--config`, `--suite`, `--offline`, `--fixtures`,
`--yes`, `--resume`, `--gate`, `--policy-gate`, `--open`).

## 6. Optional: push to hosted EvalShift

Hosted private alpha adds shared run history, web viewing, diffs, and
GitHub PR comments. Create an API token in the hosted web app, then log in:

```bash
evalshift login --token <hosted-api-token> --host <hosted-api-url>
evalshift whoami
```

Add a hosted project path to `evalshift.yaml`:

```yaml
project: acme/model-migration
thresholds:
  pass_rate_min: 0.95
```

Then push a completed run:

```bash
evalshift all --yes --push
```

Or package and push manually:

```bash
evalshift bundle <run-id>
evalshift push <run-id>
```

See [Hosted alpha](hosted.md) and [GitHub Action](github-action.md) for CI
setup and privacy details.

[uv]: https://docs.astral.sh/uv/
