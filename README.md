# EvalShift

> Run your prompts on two LLMs and find out, with statistical confidence, what regressed.

[![CI](https://github.com/babaliauskas/evalshift-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/babaliauskas/evalshift-cli/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

EvalShift is a local-first CLI that helps engineering teams migrate safely
between LLM versions (for example `gemini-2.5-flash` → `gemini-3.1-flash-lite-preview`).
Point it at your prompts and a golden suite of inputs; it runs both models,
scores the outputs with structural / semantic / LLM-as-judge / tool-call
evaluators, and produces a single-file HTML report with **defensible
statistics**: paired tests, Cohen's d, 95% CIs, and Benjamini-Hochberg
correction across every (prompt x evaluator x slice) comparison.

Local runs stay on your machine by default. Hosted private-alpha commands are
available when you explicitly log in and push a run.

## Status

**Alpha.** Every command in the pipeline is shipped and the test suite is at
95%+ coverage. APIs may still change as feedback comes in.

## Install

Requires Python 3.14+.

```bash
# Recommended
uv pip install evalshift     # or: pip install evalshift
```

From source (for contributors):

```bash
git clone https://github.com/babaliauskas/evalshift-cli.git
cd evalshift-cli
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

```bash
# 1. Scaffold a starter project. Writes evalshift.yaml + prompts.py +
#    tools.yaml + a 40-row golden.jsonl for a customer-support agent.
mkdir my-eval && cd my-eval
evalshift init

# 2. Set whichever provider keys you'll use
export GOOGLE_API_KEY=<google-api-key>   # or ANTHROPIC_API_KEY / OPENAI_API_KEY

# 3. Run the whole pipeline in one command (doctor → run → evaluate
#    → analyze → report). Pass --open to launch the report.
evalshift all --yes --open
```

`evalshift all` drives the full five-stage pipeline under a single
Rich Live region — stacked status rows, an inline progress bar for
the run stage, and a final verdict block that tells you whether the
candidate is significantly better, regressed, or showed no
significant change.

If you want to drive each stage by hand (useful in CI, or when
re-running just one stage after fixing config):

```bash
evalshift doctor
evalshift run --yes
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

Every artefact lives under `.evalshift/runs/<run-id>/` — `state.json`,
`raw.jsonl`, `scores.jsonl`, `analysis.json`, `report.json`,
`report.html`. None of it leaves your machine unless you opt in to hosted
upload commands.

## Hosted private alpha

Hosted EvalShift adds shared run history, web viewing, diffs, and GitHub PR
comments. It is optional: local CLI usage does not require an account.

```bash
# Create a hosted API token in the web app, then store it locally.
evalshift login --token <hosted-api-token> --host <hosted-api-url>
evalshift whoami

# Add a hosted project to evalshift.yaml:
# project: acme/model-migration
# thresholds:
#   pass_rate_min: 0.95

# Run locally, then package and push the result.
evalshift all --yes --push
```

You can also drive the hosted steps manually:

```bash
evalshift bundle <run-id>
evalshift push <run-id>
evalshift push --bundle .evalshift/runs/<run-id>/run_bundle.json.gz
```

Credential precedence is explicit CLI flags, then `EVALSHIFT_HOST` /
`EVALSHIFT_TOKEN`, then `~/.evalshift/credentials`.

## GitHub Action

`evalshift init --ci` scaffolds a workflow that runs EvalShift on pull
requests, pushes the run to hosted EvalShift, compares against the latest
compatible base-branch run, posts or updates one PR comment, and sets the
`evalshift/regression` commit status.

Required setup:

```bash
evalshift init --ci
```

Then add repository secrets for `EVALSHIFT_TOKEN` and the provider keys your
models use. The generated workflow uses:

```yaml
uses: babaliauskas/evalshift-action@v0
```

See [`docs/github-action.md`](docs/github-action.md) for workflow permissions,
`fail-on` modes, and baseline behavior.

## Agent migrations

Migrating an agent (a prompt that uses tools)? EvalShift detects
regressions in *which* tools the new model calls, *what* arguments it
passes, and *how* it sequences them. The killer scenario: a routing
agent that silently stops calling `notify_security_team` after the
migration — text-only eval reports green, EvalShift marks it CRITICAL.

The default `evalshift init` scaffold *is* an agent project — six tools
plus a 40-row golden suite. Just run the quick-start above and the
tool-call evaluators kick in automatically.

See [`docs/agents.md`](docs/agents.md) for the full walkthrough and
the [`examples/agent/`](examples/agent/) directory for a runnable
customer-support example.

## What the report looks like

Generate a deterministic example locally — no API keys required:

```bash
scripts/run_showcase.sh --offline --only pass-clean --open
```

That runs the [`examples/showcase/pass-clean/`](examples/showcase/pass-clean/)
scenario with the bundled `fixtures.jsonl`, writes a single-file HTML report
under `.evalshift/runs/<run-id>/report.html`, and opens it in your browser.

The HTML report (single file, no external assets, works offline) has:

* **Executive summary** — one row per prompt with a severity badge.
* **Per-prompt deep dive** — aggregate stats, per-slice breakdown,
  top-5 worst regressions side-by-side.
* **Methodology appendix** — every test, p-value, effect size, and
  CI is documented.

## Why local-first?

Your prompts and suite stay local for `doctor`, `run`, `evaluate`, `analyze`,
and `report`. The only outbound calls in local mode are to the LLM providers
you configure (Anthropic, OpenAI, Google) using your own API keys.

`bundle` packages completed local artifacts into `run_bundle.json.gz` without
uploading them. `push` and `all --push` upload that bundle to the hosted
backend associated with your token.

## Documentation

* [Getting started](docs/getting-started.md) — install + first run walkthrough
* [Hosted alpha](docs/hosted.md) — login, bundle, push, thresholds, privacy
* [GitHub Action](docs/github-action.md) — PR comments + hosted regression gate
* [Configuration reference](docs/configuration.md) — every `evalshift.yaml` field
* [Evaluators](docs/evaluators.md) — when to use which family
* [Methodology](docs/methodology.md) — the statistical machinery
* [FAQ](docs/faq.md) — common questions

## Non-goals

* General-availability hosted service or billing
* Hosted provider-key storage
* Multi-criterion judge in a single call
* Custom evaluator plugin system
* Comparing more than 2 models in one run
* Auto-detection of LangChain / LlamaIndex prompt patterns

## License

[AGPL-3.0-or-later](LICENSE). Free for any use, including commercial, provided
that derivative works — including network-hosted services — are released under
the same license.

Versions `0.3.0` and earlier (published on PyPI before this change) remain
available under the MIT License terms they were released with.

Commercial licenses without the AGPL share-back requirement are available;
contact <l.babaliauskas@gmail.com>.
