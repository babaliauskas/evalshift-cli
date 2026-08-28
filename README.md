# EvalShift

Open-source LLM migration and regression testing for AI agents.

[![CI](https://github.com/babaliauskas/evalshift-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/babaliauskas/evalshift-cli/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**The SDK captures what your agent really did. The CLI replays it against a
candidate model and tells you what broke. The hosted app keeps the history.**

```
evalshift-sdk    captures real agent behavior in production
      ↓
evalshift CLI    replays it against a candidate model — scores, stats, report
      ↓
hosted (opt-in)  run history, diffs, PR gates
```

Migrating between LLM versions (say `gemini-2.5-flash` →
`gemini-3.1-flash-lite-preview`) means guessing which behaviors changed.
EvalShift removes the guess. It runs both models over the same golden suite,
scores the outputs with structural / semantic / LLM-as-judge / tool-call
evaluators, and produces a single-file HTML report with **defensible
statistics**: paired tests, Cohen's d, 95% CIs, and Benjamini-Hochberg
correction across every (prompt x evaluator x slice) comparison.

An eval is only worth the examples in it. That is why the
[capture SDK](https://github.com/babaliauskas/evalshift-sdk) is part of the
product rather than an add-on: it records real production runs — model calls,
tool calls, final outputs — to disk, and `evalshift capture sync` promotes them
into golden suites. Hand-written suites are fully supported too, but captured
traffic is the recommended starting point.

Local runs stay on your machine by default. Hosted private-alpha commands are
available when you explicitly log in and push a run.

## How EvalShift fits together

Four pieces, released and documented independently:

| Piece | What it does for you | Reference |
| --- | --- | --- |
| **SDK** — PyPI `evalshift-sdk` | Records what your agent actually did in production — model calls, tool calls, final output — as capture files on disk. Those captures become your golden suite. | [docs/sdk.md](docs/sdk.md) |
| **CLI** — this repo, PyPI `evalshift` | Replays the suite on two models, scores, analyses, reports, bundles, pushes. | [DOCS.md](DOCS.md) |
| **GitHub Action** — `babaliauskas/evalshift-action@v0` | Runs the pipeline on pull requests, pushes the run, posts one PR comment, sets the `evalshift/regression` status. | [docs/github-action.md](docs/github-action.md) |
| **Hosted server** — `api.evalshift.dev`, web app at `evalshift.dev` | Optional. Stores pushed run bundles, diffs them across branches, drives PR comments and gating. | [docs/hosted.md](docs/hosted.md) |

The SDK and the CLI never call each other — the interface is files under
`.evalshift/captures/`, so either works without the other. Because both use the
top-level import name `evalshift`, install them in **separate virtual
environments**: the SDK in your agent's, the CLI wherever you run evaluations.

## For AI coding agents

Point your coding agent at the dense, single-file reference for the piece it is
working on:

- EvalShift CLI: <https://www.evalshift.dev/cli-llms-full.txt>
  (source of truth: [llms-full.txt](llms-full.txt) in this repo)
- EvalShift SDK: <https://www.evalshift.dev/sdk-llms-full.txt>
- EvalShift GitHub Action (CI): <https://www.evalshift.dev/ci-llms-full.txt>

## Status

**Alpha.** Every command in the pipeline is shipped and the test suite covers
92% of the source. APIs may still change as feedback comes in.

## Install

Requires Python 3.11+.

```bash
# Recommended
uv pip install evalshift     # or: pip install evalshift
```

And, in your agent's virtualenv — a **separate** one, see above — the capture
SDK that feeds the CLI its suites:

```bash
uv pip install evalshift-sdk     # or: pip install evalshift-sdk
```

From source (for contributors):

```bash
git clone https://github.com/babaliauskas/evalshift-cli.git
cd evalshift-cli
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

### Use it on your agent

This is the workflow: record what your agent really does, then hold a
candidate model to it.

```bash
evalshift init                    # minimal capture-first evalshift.yaml
```

Instrument the agent with [evalshift-sdk](https://github.com/babaliauskas/evalshift-sdk)
— installed in the agent's own virtualenv, stdlib-only, Python 3.10+:

```python
from evalshift import capture


@capture.tool(name="issue_refund")
def issue_refund(order_id: str) -> dict: ...


@capture.agent(suite="support_agent", redact=True)
def handle(message: str) -> str: ...
```

`redact=` is required on every entry point that opens a capture session (SDK
0.3.0+) — `@capture.agent`, `capture.agent_session`,
`capture.agent_session_async`, `EvalShiftCallbackHandler`: `True` masks emails,
API keys and bearer tokens before anything reaches disk, `False` records
verbatim, or pass your own `(value) -> value` callable. `@capture.tool` takes no
`redact` of its own — tool spans are masked by the redactor of the agent session
they run inside.

Nothing is recorded unless `EVALSHIFT_CAPTURE=1` is set, so the decorators are
safe to leave in production permanently:

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py   # writes .evalshift/captures/
evalshift capture sync                     # captures → golden suites + wired config
evalshift all --suite-name support_agent --to <candidate-model>
```

See [docs/sdk.md](docs/sdk.md) for the full capture contract. Can't instrument
the agent? A hand-written `golden.jsonl` works just as well — see
[Getting started](docs/getting-started.md).

### Driving the pipeline

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
`raw.jsonl`, `scores.jsonl`, `analysis.json`, `migration_decision.json` (when
the config sets a `migration_policy`), `report.json`, `report.html`, and
`insights.json` (when insights ran). None of it leaves your machine unless you
opt in to hosted upload commands.

## Hosted private alpha

Hosted EvalShift adds shared run history, web viewing, diffs, and GitHub PR
comments. It is optional: local CLI usage does not require an account.

```bash
# Sign in through the hosted web app, then approve CLI login in the browser.
# Defaults to https://api.evalshift.dev; pass --host to target another server.
evalshift login
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

### What gets uploaded

Nothing, until you run `push` (or `all --push`) — and the CLI itself has no
telemetry, analytics, or crash reporting. A push uploads one file,
`run_bundle.json.gz`, whose full field-by-field contract is documented in
[docs/hosted.md — Privacy model](docs/hosted.md#privacy-model--exactly-what-uploads).
The short version:

* **Uploads**: the run manifest (model ids, suite name, git SHA/branch/PR
  number, content hashes, CLI version); per-example rows — the example's
  template `inputs` and `expected` output verbatim, both models' full outputs,
  tool-call traces (names and arguments), scores, cost and latency; aggregate
  statistics, the analysis, the migration decision, economics, and the
  machine-written insights narrative.
* **Never uploads**: provider API keys, prompt bodies and system prompts,
  suite conversation histories, tool definitions/schemas, `raw.jsonl`, the
  response cache, captures, and `report.html`. Prompt and dataset content is
  replaced by SHA-256 hashes so diffs still align across runs.
* **Can still be sensitive**: inputs, expected outputs, model outputs, and
  traces carry whatever content your suite or your models put in them. Redact
  at capture time (see the SDK's redaction boundary) and inspect before
  pushing: `evalshift bundle <run-id>` writes the exact bytes a push would
  upload — `gunzip -c .evalshift/runs/<run-id>/run_bundle.json.gz | jq .`.

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

Each golden-suite example carries its own toolset — recorded automatically
by `capture promote` / `capture sync` from your production captures, or
inlined by hand for a hand-authored suite.

See [`docs/agents.md`](docs/agents.md) for the full walkthrough and
the [`examples/agent/`](examples/agent/) directory for a runnable
customer-support example.

## What the report looks like

Every run writes a single-file HTML report to
`.evalshift/runs/<run-id>/report.html` — see [Use it on your
agent](#use-it-on-your-agent) above, or the runnable walkthrough in
[`examples/agent/`](examples/agent/). The report (single file, no external
assets, works offline) has:

* **Migration verdict** — the policy decision up top: which budgets failed, the
  top regression causes, and the recommendation.
* **Executive summary** — one row per prompt with a severity badge.
* **What changed, in plain language** — the verdict, the economics and the
  behavioural drift explained by `defaults.insights_model`. Every figure in it
  is copied from the computed statistics, never generated. Needs a provider
  key; skip it with `--no-insights`.
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

## Wiring the agent references into your project

The three references are listed at the top of this README. `evalshift init`
wires these links into your project automatically: it writes
`EVALSHIFT.md` and points existing agent files (`AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.cursorrules`, `.github/copilot-instructions.md`) at it, creating
`AGENTS.md` if none of those files exist. Disable with `--no-wire-agents`. In
*this* repo the same three links live in
[AGENTS.md](AGENTS.md).

## Documentation

* [DOCS.md](DOCS.md) — consolidated single-file reference for everything below
* [Getting started](docs/getting-started.md) — install + first run walkthrough
* [Configuration reference](docs/configuration.md) — every `evalshift.yaml` field
* [Evaluators](docs/evaluators.md) — when to use which family
* [Agent migrations](docs/agents.md) — tool-call evaluation, per-example toolsets
* [Multi-turn conversations](docs/conversations.md) — teacher-forced replay
* [Agent traces](docs/traces.md) — bring-your-own agent timelines
* [Capture SDK](docs/sdk.md) — instrument your agent, promote captures to suites
* [Methodology](docs/methodology.md) — the statistical machinery
* [Hosted alpha](docs/hosted.md) — login, bundle, push, thresholds, and the
  privacy model: exactly what data uploads and what never leaves your machine
* [GitHub Action](docs/github-action.md) — PR comments + hosted regression gate
* [FAQ](docs/faq.md) — common questions
* [llms-full.txt](llms-full.txt) — dense single-file reference for AI coding
  tools, hosted at <https://www.evalshift.dev/cli-llms-full.txt>

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
