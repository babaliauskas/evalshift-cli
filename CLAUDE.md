# CLAUDE.md

Operating instructions for Claude Code in this repo. This file is about *how to work here* —
what the product does and how the CLI behaves is documented in [DOCS.md](DOCS.md) (single
consolidated reference; per-topic pages under `docs/`). Read the relevant DOCS.md section when a
task touches CLI behavior instead of re-deriving it from source. Do not duplicate documentation
into this file.

## Project in one line

EvalShift: local-first CLI for safe LLM model migrations — runs two models against a golden
suite, scores with structural/semantic/judge/tool-call evaluators, emits an HTML report with
paired stats. Alpha, published on PyPI as `evalshift`. Python **3.11+**, `mypy --strict` clean.

## Sibling pieces and their references

Four pieces ship separately; each owns one dense LLM reference. Fetch the matching one before
writing code or config for it — don't infer the API from memory. Orientation for any AI tool
opening this repo lives in [AGENTS.md](AGENTS.md); the ecosystem table is in
[DOCS.md](DOCS.md#ecosystem-and-ai-tool-references).

| Piece | Reference |
| --- | --- |
| CLI (this repo) | <https://www.evalshift.dev/cli-llms-full.txt> — generated from [llms-full.txt](llms-full.txt) here |
| SDK (`evalshift-sdk`, capture) | <https://www.evalshift.dev/sdk-llms-full.txt> — owned by the SDK repo |
| GitHub Action (CI) | <https://www.evalshift.dev/ci-llms-full.txt> — owned by the action repo |
| Hosted server (`api.evalshift.dev`) | `../evalshift-server/BUNDLE_SPEC.md` (upload contract) + [docs/hosted.md](docs/hosted.md) |

## Commands

Setup (uses `uv`):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install   # wires the commit hooks *and* the pre-push CI mirror
```

Day-to-day:

```bash
pytest                                      # full suite (cov on by default)
pytest -m "not integration"                 # unit tests only
pytest tests/unit/test_orchestrator.py::test_name -x
ruff check . && ruff format .
mypy --strict src/evalshift
make ci                                     # exactly what CI runs (also the pre-push hook)
pre-commit run --all-files                  # commit-stage hooks over the tree
```

CLI entry point: `evalshift = "evalshift.cli.main:app"`. Pytest markers: `integration`, `slow`.

## Workflow — which skills, when

Use these skills at these moments (announce, then follow them exactly):

| Moment | Skill |
| --- | --- |
| Before building/changing any feature or behavior | `superpowers:brainstorming`, then `superpowers:writing-plans` for multi-step work |
| Implementing any logic or bugfix | `superpowers:test-driven-development` — test first, watch it fail, then code |
| Any bug, test failure, or unexpected behavior | `superpowers:systematic-debugging` — root cause before fixes |
| Before claiming done / committing / opening a PR | `superpowers:verification-before-completion` — run `pre-commit run --all-files`, show output |
| After completing a significant change | `superpowers:requesting-code-review` |
| Locating code / bounded 1–2 file edits / diff review in long sessions | `caveman:cavecrew` subagents (investigator / builder / reviewer) to keep main context lean |

Best practices that are always on:

- **Read before you write.** Open the file you'll modify, its test file, and one existing
  example of the pattern you're about to add. Match the local idiom.
- **Tests mirror source**: new module under `src/evalshift/...` → test file under
  `tests/unit/...`. Integration tests in `tests/integration/`.
- **Green means all four**: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` —
  CI runs them on every PR. Never claim success without running them.
- **Conventional Commits** for messages; update `CHANGELOG.md` under `## [Unreleased]` for any
  user-visible change.
- **Docs travel with behavior**: a user-visible change updates `DOCS.md` + `llms-full.txt`
  (repo root) and the matching `docs/` page in the same change.

## Repo map

Modules under `src/evalshift/` — one-line orientation, read the module for detail:

| Module | What lives there |
| --- | --- |
| `cli/` | Typer app; one file per subcommand under `cli/commands/`. |
| `config/` | Pydantic models for `evalshift.yaml`. |
| `suite/` | Golden `*.jsonl` loader + models. |
| `parsers/` | Prompt sourcing (`manual`, `python_string` AST parser). |
| `models/` | LiteLLM client, model registry. |
| `cache/` | Async SQLite response cache (SHA-256 key, 7-day TTL). |
| `runner/` | Orchestrator, `Call`/`RunState`, checkpoint/resume, run retention. |
| `evaluators/` | One file per evaluator family + shared tool-call infra. |
| `analysis/` | `statistics.py` (paired tests, BH-FDR, severity) + `slicing.py`. |
| `reports/` | JSON payload + single-file Jinja HTML report. |
| `utils/` | Cost estimation, suite×prompt template validation. |

Deep references: statistical contract → `docs/methodology.md`; agent/tool evals →
`docs/agents.md`; pipeline, artefacts, config schema → `DOCS.md`.

## Hard rules

- **Config is public API.** All config models use `extra="forbid"` — keep it that way; typos in
  user YAML must fail loudly. Any new top-level `evalshift.yaml` field needs entries in
  `docs/configuration.md` and `DOCS.md`.
- **Never execute user code.** The `python_string` prompt parser AST-extracts only; it
  rejects dynamic forms (f-strings, concatenation, `.format()`). Preserve this.
- **Reuse the stage cores.** New commands that chain pipeline stages call `run_evaluate` /
  `run_analyze` / `run_report` / `run_orchestrator` (see `cli/commands/all.py` for the pattern)
  — never re-implement stage logic. Raise the existing typed exceptions (`ConfigError`,
  `CheckpointError`, `NoEvaluatorsError`, `NoPairsError`, `MissingScoresError`,
  `EmptyScoresError`) so callers pretty-print uniformly.
- Every module starts with `from __future__ import annotations`; public functions/classes get
  Google-style docstrings.
- `.evalshift/` is generated runtime data, gitignored — never commit it, never hand-edit it.
