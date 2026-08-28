# Getting started

This page walks you from a fresh shell to a working `evalshift.yaml` in
under a minute.

## 1. Install

EvalShift is a Python 3.11+ package. We recommend [`uv`][uv] for dependency
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

The capture SDK is a separate package that goes in **your agent's** virtualenv,
not this one — both use the top-level import name `evalshift`, so they must not
share an environment:

```bash
uv pip install evalshift-sdk
```

It is optional for this walkthrough — you can hand-write `golden.jsonl`
instead (see step 5) — but it is how real projects build their suite
fastest.

## 2. Set provider API keys

EvalShift calls Anthropic, OpenAI, and Google directly using your own keys.
Local runs do not send prompts or outputs to an EvalShift-operated server.
Provider responses are cached locally in `~/.evalshift/cache.db`.

Set whichever providers you intend to use:

```bash
export ANTHROPIC_API_KEY=<anthropic-api-key>
export OPENAI_API_KEY=<openai-api-key>
export GEMINI_API_KEY=<gemini-api-key>
```

## 3. Scaffold your project

```bash
mkdir my-eval
cd my-eval
evalshift init
```

You can also pick a migration profile:

```bash
evalshift init --profile cost-reduction
```

The default `model-upgrade` profile scaffolds a `migration_policy` block
that powers the verdict in `analyze`, `all`, and `report`.

This writes a single, minimal, capture-first `evalshift.yaml`: a
passthrough `replay` prompt, advisory semantic + LLM-judge evaluators, an
empty managed `suites:` block for `capture sync` to fill, and the migration
policy. `init` refuses to clobber an existing `evalshift.yaml`; pass
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

## 5. Record what your agent actually does

`init`'s `suites:` block starts empty — `run` needs a golden suite to
dispatch against. Instrument your agent with [evalshift-sdk][sdk]
(installed in step 1):

```python
from evalshift import capture


@capture.agent(suite="support_agent", redact=True)
def handle(message: str) -> str: ...
```

Then exercise the agent with capture turned on:

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py   # writes .evalshift/captures/
```

Captures are off unless `EVALSHIFT_CAPTURE=1` is set, so the decorator can
stay in production code. Full contract: [Capture SDK](sdk.md). If you can't
instrument the agent, write `golden.jsonl` by hand instead — see
[Configuration](configuration.md).

## 6. Promote captures into a suite

```bash
evalshift capture sync
```

`capture sync` promotes every recorded capture into
`.evalshift/suites/<suite>/golden.jsonl` and injects the matching
`suites:` block into `evalshift.yaml`. See
[Configuration](configuration.md) for the full capture lifecycle.

## 7. Run the pipeline

The fast path is one command:

```bash
evalshift all --suite-name support_agent --to <candidate-model> --yes --open
```

This runs `doctor → run → evaluate → analyze → report` under a single
Rich Live region with a progress bar for the run stage and a final
verdict block. Warnings raised along the way (LiteLLM deprecation
notices, insights retries) are held back and printed as one `⚠` section
directly under the pipeline block; errors are never deferred.
`run`/`all` estimate worst-case cost up front and prompt for
confirmation above $10 (skip with `--yes`).

If you want to drive each stage by hand (useful when re-running just
one stage after fixing config, or in CI where you stage artefacts):

```bash
evalshift run --suite-name support_agent --to <candidate-model>
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

`evalshift all` accepts every flag the underlying commands do
(`--from/--to`, `--config`, `--suite`, `--suite-name`, `--yes`, `--resume`,
`--gate`, `--policy-gate`, `--open`, `--push`).

## 8. Optional: import agent traces

If your own agent runtime already records source and target timelines,
attach them to a completed run before `evaluate`:

```bash
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

Configure `evaluators.agent_trace` to compare tool order, argument
drift, extra dangerous actions, and missing verification steps. See
[Agent traces](traces.md) for the JSONL schema.

## 9. Optional: push to hosted EvalShift

Hosted private alpha adds shared run history, web viewing, diffs, and
GitHub PR comments. Sign in through the hosted web app, then approve CLI login
in the browser:

```bash
evalshift login --host <hosted-api-url>
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
[sdk]: https://github.com/babaliauskas/evalshift-sdk
