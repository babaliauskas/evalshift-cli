# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

EvalShift is a local-first CLI for safe LLM model migrations (e.g. `Codex-4.5-sonnet` → `Codex-5-sonnet`).
It runs the same prompts on two models against a golden suite, scores outputs with structural / semantic /
LLM-as-judge / tool-call evaluators, and emits an HTML report with paired stats (Cohen's d, 95% CIs,
Benjamini–Hochberg FDR correction). Status is **alpha**; not yet on PyPI.

Python **3.14+ only**. The repo is `mypy --strict` clean and CI runs `ruff check`, `ruff format --check`,
`mypy --strict`, and `pytest` on every PR (see `.github/workflows/ci.yml`).

## Common commands

Setup uses `uv`:
```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Day-to-day:
```bash
pytest                                      # full suite (cov enabled by default)
pytest -m "not integration"                 # unit tests only
pytest tests/unit/test_orchestrator.py      # one file
pytest tests/unit/test_orchestrator.py::test_name -x   # one test, fail-fast
ruff check . && ruff format .               # lint + auto-format
mypy --strict src/evalshift                 # type-check
pre-commit run --all-files                  # everything CI runs
```

`pyproject.toml` wires `--cov=evalshift --cov-report=term-missing` into pytest by default and registers
two markers: `integration` and `slow`.

The CLI entry point is `evalshift = "evalshift.cli.main:app"`; once installed, run `evalshift --help`.

## End-to-end pipeline (the user-facing flow this code implements)

The CLI is a five-stage pipeline. Each stage writes one artefact under
`.evalshift/runs/<run-id>/`, and the next stage reads it. Stages are independently re-runnable.

```
init  →  doctor  →  run         →  evaluate      →  analyze         →  report
                    raw.jsonl      scores.jsonl     analysis.json     report.html
                    state.json
```

* **`init`** scaffolds `evalshift.yaml`, `prompts.py`, `tools.yaml`, and a 40-row `golden.jsonl`
  for a customer-support agent (the v0.2 default scaffold *is* an agent project).
* **`doctor`** validates the local config + API keys.
* **`run`** dispatches calls. The async **orchestrator** in `src/evalshift/runner/orchestrator.py`
  is the heart of this stage: it parses prompts, validates each example against each prompt,
  estimates cost (prompts the user above $10 unless `--yes`), opens the SQLite cache, builds a
  work list of `(prompt × example × {source,target})`, and processes under a concurrency
  semaphore. Checkpoints every 50 completions; resume support is built-in.
* **`evaluate`** scores `raw.jsonl` against the configured evaluators.
* **`analyze`** runs paired stats per `(prompt, evaluator, slice)` triple. See
  `docs/methodology.md` for the full statistical contract — Shapiro-Wilk normality screen,
  paired-t or Wilcoxon fallback, Cohen's d with 95% CI, BH-FDR correction at α=0.05, then
  severity classification.
* **`report`** renders a single-file HTML report (no external assets, works offline).
* **`all`** drives `doctor → run → evaluate → analyze → report` end to end under a single
  `rich.live.Live` region. Lives in `src/evalshift/cli/commands/all.py`. Uses the reusable
  cores `run_evaluate` / `run_analyze` / `run_report` (see Architecture below) plus
  `run_orchestrator` with its `on_progress` callback so the Live UI updates per call.

`validate` and `test-call` are hidden debug commands.

### Reusable stage cores

Each downstream Typer command (`evaluate`, `analyze`, `report`) is a thin wrapper around a
sync core function that returns a typed result dataclass:

* `run_evaluate(*, run_id, config_path, runs_base, console, quiet=False) -> EvaluateResult`
* `run_analyze(*, run_id, config_path, runs_base) -> AnalyzeResult`
* `run_report(*, run_id, config_path, runs_base) -> ReportResult`

`run_orchestrator(...)` similarly returns `RunResult`, and exposes an
`on_progress: Callable[[ProgressEvent], None] | None` hook plus `preflight_cost(...)` for
callers (currently `evalshift all`) that want to drive their own progress UI / cost preflight
without invoking the standalone Typer commands.

When adding a new top-level command that needs to chain stages, prefer these cores over
re-implementing their logic — and always raise the existing typed exceptions (`ConfigError`,
`CheckpointError`, `NoEvaluatorsError`, `NoPairsError`, `MissingScoresError`,
`EmptyScoresError`) so callers can pretty-print them uniformly.

## Architecture

Modules under `src/evalshift/`:

| Module        | Responsibility |
| ------------- | -------------- |
| `cli/`        | Typer app (`cli/main.py`) and one file per subcommand under `cli/commands/`. |
| `config/`     | Pydantic models for `evalshift.yaml`. **`extra="forbid"` everywhere** — typos in user YAML must fail loudly. New top-level fields here become public API and need a doc entry in `docs/configuration.md`. |
| `suite/`      | Loader + Pydantic models for the golden `*.jsonl` suite. |
| `parsers/`    | Prompt sourcing. Two modes: `manual` (inline `content`) and `python_string` (AST-walks a `.py` file for a module-level string assignment). The `python_string` parser **does not execute user code** and explicitly rejects f-strings, concatenation, `.format()`, and other dynamic forms. |
| `models/`     | Provider-agnostic LLM client (`client.py`, wraps `litellm`), canonical model registry with aliases (`registry.py`), and `replay_client.py` for `--offline` runs that replay canned `fixtures.jsonl` responses. |
| `cache/`      | Async SQLite cache keyed by SHA-256 over canonicalised JSON of `(model, prompt, inputs, temperature, max_tokens)`. Default 7-day TTL. |
| `runner/`     | The orchestrator + `Call` / `RunState` models + checkpoint/resume logic. |
| `evaluators/` | One file per evaluator family: `structural`, `semantic`, `llm_judge`, plus the v0.2 tool-call evaluators (`tool_selection`, `tool_arguments`, `tool_trace_structure`) and their shared `tool_models` / `tool_parser` / `tool_loader` infra. |
| `analysis/`   | `statistics.py` (paired tests, BH correction, severity) + `slicing.py`. |
| `reports/`    | `json.py` builds the report payload; `html.py` renders the single-file Jinja template under `reports/templates/`. |
| `utils/`      | `cost.py` (token-based cost estimation; `tiktoken`) and `templating.py` (suite × prompt compatibility validation). |

### v0.2 agent-eval addition

A prompt becomes "agent-style" when its `prompts[].tools_path` is set (see `docs/agents.md`).
The orchestrator then dispatches via `ModelClient.complete_with_tools`, each `Call` row carries
a parsed provider-agnostic `ToolTrace`, and the tool-call evaluators score against per-example
`expected_tools` / `expected_no_tools` / `expected_parallel` ground truth.

`tools.yaml` accepts both Anthropic-shape (`name` / `description` / `input_schema`) and
OpenAI-shape (`{"type":"function","function":{...}}`) entries — the model client serialises
to the right shape per provider.

### Offline mode

`evalshift run --offline --fixtures fixtures.jsonl` swaps in `ReplayClient`. Fixtures match by
canonical model id + a substring of the rendered prompt; exactly one fixture must match.
Used by `examples/showcase/` and `scripts/run_showcase.sh` for deterministic demos without
API keys. `scripts/capture_fixtures.py` records new fixtures from a live run.

## Conventions

* All modules use `from __future__ import annotations`.
* Public functions and classes get Google-style docstrings.
* Tests mirror the source layout under `tests/unit/` (and `tests/integration/` for end-to-end).
  Add a test file alongside any new module.
* Commit messages follow Conventional Commits.
* Update `CHANGELOG.md` under `## [Unreleased]` for any user-visible change.
* Generated runtime data lives under `.evalshift/` and is gitignored — never commit it.
