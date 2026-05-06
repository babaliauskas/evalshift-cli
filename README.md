# AIMigrate

> Run your prompts on two LLMs and find out, with statistical confidence, what regressed.

[![CI](https://github.com/babaliauskas/AIMigrate/actions/workflows/ci.yml/badge.svg)](https://github.com/babaliauskas/AIMigrate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#status)

AIMigrate is a local-first CLI that helps engineering teams migrate safely between LLM
versions (e.g. `claude-4.5-sonnet` → `claude-5-sonnet`). Point it at your prompts and a
golden suite of inputs; it runs both models, scores the outputs with structural,
semantic, and LLM-as-judge evaluators, and produces a single-file HTML report with
**defensible statistics**: paired tests, Cohen's d, 95% CIs, and Benjamini–Hochberg
correction across every (prompt × evaluator × slice) comparison.

## Status

**Pre-alpha — under active development.** This README will fill in as features land.
See [`CHANGELOG.md`](CHANGELOG.md) for what's shipped and the build plan in the repo
for what's next.

## Install (once published)

```bash
uv pip install aimigrate     # or: pip install aimigrate
```

## Quick start

The commands marked **(implemented)** work today; the rest are landing
incrementally — track progress in [`MVP_TODO.md`](MVP_TODO.md).

```bash
aimigrate init               # (implemented) scaffold aimigrate.yaml + example suite
aimigrate doctor             # (implemented) verify environment + config
aimigrate run \              # (in progress, Phase 4)
  --from claude-4.5-sonnet \
  --to   claude-5-sonnet \
  --suite golden.jsonl
aimigrate report <run-id>    # (planned, Phase 7) write a single-file HTML report
```

See [`docs/getting-started.md`](docs/getting-started.md) for the full walkthrough.

## Why local-first?

Your prompts and your suite never leave your machine. The only outbound calls are to
the LLM providers you configure (Anthropic, OpenAI, Google) using your own API keys.

## License

[MIT](LICENSE) — free for any use.
