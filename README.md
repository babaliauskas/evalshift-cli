# AIMigrate

> Run your prompts on two LLMs and find out, with statistical confidence, what regressed.

[![CI](https://github.com/babaliauskas/AIMigrate/actions/workflows/ci.yml/badge.svg)](https://github.com/babaliauskas/AIMigrate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

AIMigrate is a local-first CLI that helps engineering teams migrate safely between
LLM versions (e.g. `claude-4.5-sonnet` → `claude-5-sonnet`). Point it at your
prompts and a golden suite of inputs; it runs both models, scores the outputs
with structural / semantic / LLM-as-judge evaluators, and produces a single-file
HTML report with **defensible statistics**: paired tests, Cohen's d, 95% CIs,
and Benjamini–Hochberg correction across every (prompt × evaluator × slice)
comparison.

## Status

**Alpha.** Every command in the pipeline is shipped; the test suite is at 95%+
coverage; the package isn't yet on PyPI.

## Install

```bash
# Clone and install in dev mode
git clone https://github.com/babaliauskas/AIMigrate.git
cd AIMigrate
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

## Quick start

```bash
# 1. Scaffold a starter project (writes aimigrate.yaml + prompts.py + golden.jsonl)
mkdir my-eval && cd my-eval
aimigrate init

# 2. Verify your environment
export GOOGLE_API_KEY=...   # or ANTHROPIC_API_KEY / OPENAI_API_KEY
aimigrate doctor

# 3. Run both models against the suite
aimigrate run --yes --from gemini-2.5-flash --to gemini-2.5-pro

# 4. Score outputs and analyse
aimigrate evaluate <run-id>
aimigrate analyze <run-id>

# 5. Generate the single-file HTML report
aimigrate report <run-id> --open
```

Every artefact lives under `.aimigrate/runs/<run-id>/` — `state.json`,
`raw.jsonl`, `scores.jsonl`, `analysis.json`, `report.json`,
`report.html`. None of it leaves your machine.

## What the report looks like

The HTML report (single file, no external assets, works offline) has:

* **Executive summary** — one row per prompt with a severity badge.
* **Per-prompt deep dive** — aggregate stats, per-slice breakdown,
  top-5 worst regressions side-by-side.
* **Methodology appendix** — every test, p-value, effect size, and
  CI is documented.

## Why local-first?

Your prompts and your suite never leave your machine. The only outbound calls
are to the LLM providers you configure (Anthropic, OpenAI, Google) using your
own API keys. There is no AIMigrate cloud.

## Documentation

* [Getting started](docs/getting-started.md) — install + first run walkthrough
* [Configuration reference](docs/configuration.md) — every `aimigrate.yaml` field
* [Evaluators](docs/evaluators.md) — when to use which family
* [Methodology](docs/methodology.md) — the statistical machinery
* [FAQ](docs/faq.md) — common questions
* [`MVP_TODO.md`](MVP_TODO.md) — the build checklist (every box ticked)

## Non-goals (for v0.1)

* Hosted backend / web UI
* Multi-criterion judge in a single call
* Custom evaluator plugin system
* Comparing more than 2 models in one run
* Auto-detection of LangChain / LlamaIndex prompt patterns

These are deferred to v0.2+; see the PDF spec in the repo for the
full deferred-features list.

## License

[MIT](LICENSE) — free for any use.
