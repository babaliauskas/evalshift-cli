# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — Tool-call evaluation

Adds agent-migration support: detects regressions in *which* tools the
new model calls, *what* arguments it passes, and *how* it sequences
them. v0.1 backward compatible — every existing example keeps working
without changes.

### Added

- `aimigrate.evaluators.tool_models`: provider-agnostic `ToolSpec`,
  `ToolCall`, `ToolTrace`. `ToolSpec.to_anthropic` / `to_openai` /
  `from_dict` adapters keep wire-format details out of user code.
- `aimigrate.evaluators.tool_parser`: `parse_response_to_trace`
  dispatcher with three provider parsers. Handles the LiteLLM
  Anthropic-normalised-to-OpenAI shape transparently. Malformed
  argument JSON marked `_parse_error` instead of crashing.
- `aimigrate.evaluators.tool_loader`: `load_tools(yaml or json)` with
  the same plain/rich error rendering pattern as `ConfigError`.
- `aimigrate.models.client.complete_with_tools` and
  `ToolCompletionResult` — tool-aware call path that doesn't replace
  the existing `complete` / `CompletionResult`.
- `aimigrate test-call --tools <path>` smoke command extension.
- `scripts/smoke_live_tools.py` for fixture capture (manual; not in CI).
- Three new evaluators wired into `aimigrate evaluate`:
  - `ToolSelectionEvaluator` (modes: exact / set / first / expected)
  - `ToolArgumentsEvaluator` (per-field strategies: exact / subset /
    numeric / semantic; greedy nearest-index call matching)
  - `ToolTraceStructureEvaluator` (call_count / parallelism /
    refusal_alignment / expected_count_alignment; refusal mismatches
    force severity_floor: high)
- Suite extension: optional `expected_tools` (with per-call
  `match_strategy`), `expected_tool_count`, `expected_no_tools`,
  `expected_parallel`. Mutual-exclusion checks where appropriate.
- Config extension: `prompts[].tools_path` makes a prompt agent-style;
  `evaluators.{tool_selection, tool_arguments, tool_trace_structure}`
  blocks added.
- Orchestrator dispatch: agent prompts route through
  `complete_with_tools`; the resulting `Call.trace` round-trips via
  pydantic so `raw.jsonl` Just Works.
- HTML report extension: per-prompt "Top regressions" section now
  renders side-by-side trace diffs for tool-evaluator regressions
  with green / yellow / red colour coding for matching / different-
  args / missing-extra calls. CSS stays inlined.
- `aimigrate doctor` warns about common agent-config mistakes:
  prompts with `tools_path` but no tool evaluators, and suite
  examples with `expected_tools` but no agent prompts.
- Docs: `docs/agents.md` walkthrough, plus configuration / evaluators
  / faq updates.
- `examples/agent/` runnable customer-routing example with 12 golden
  rows across security / routine / text-only slices.

### Quality

- 512+ tests, including end-to-end integration tests covering the
  PRD §9.3 failure-mode grid. ruff + format + mypy --strict clean.
- v0.1 backward compatibility verified: existing 462 tests still
  green; `examples/simple/` unchanged.

## [0.1.0]

First alpha release. Every command in the planned MVP pipeline is shipped.

### Live commands

| Command                | What it does                                                            |
| ---------------------- | ----------------------------------------------------------------------- |
| `aimigrate doctor`     | Check Python version, provider API keys, and `aimigrate.yaml` validity. |
| `aimigrate init`       | Scaffold a starter project (yaml + prompts.py + golden.jsonl).          |
| `aimigrate run`        | Call source + target models on every (prompt, example) → `raw.jsonl`.   |
| `aimigrate evaluate`   | Score every (source, target) pair → `scores.jsonl`.                     |
| `aimigrate analyze`    | Paired tests + BH correction → `analysis.json`.                         |
| `aimigrate report`     | Single-file HTML report → `report.html` + `report.json`.                |
| `aimigrate cache clear`| Wipe the local SQLite response cache.                                   |
| `aimigrate validate`   | (hidden dev) Verify config + suite + prompts are compatible.            |
| `aimigrate test-call`  | (hidden dev) One-shot smoke call to confirm provider connectivity.      |

### Added — Phase 0–1

- `pyproject.toml` (Python ≥3.14, hatchling, MIT, pinned deps), `ruff.toml`, `mypy.ini`, pytest config, `.pre-commit-config.yaml`, GitHub Actions CI.
- Full `src/aimigrate` package skeleton across cli/config/parsers/models/suite/runner/evaluators/analysis/reports/cache/utils with module docstrings and a `py.typed` marker.
- `aimigrate.config.models`: pydantic v2 schema for `aimigrate.yaml` (prompts, evaluators, slices, defaults) with strict validation, `extra='forbid'`, and detection-mode field invariants.
- `aimigrate.config.loader`: `load_config()` plus a structured `ConfigError` with both plain-text and Rich panel rendering.
- `aimigrate doctor`: reports Python version, provider API keys, and `aimigrate.yaml` validity. Exits 1 only on hard failures.
- `aimigrate init`: scaffolds `aimigrate.yaml` + `prompts.py` + `golden.jsonl`. `--force` overwrite, `--directory` target dir.

### Added — Phase 2 (suite + prompt loading)

- `aimigrate.suite` package: pydantic `SuiteExample` + `Suite` models, JSONL loader with line-numbered errors, blank-line tolerance, multi-error collection.
- `aimigrate.parsers` package: `ManualParser` (inline content) and `PythonStringParser` (AST-walks `.py` for module-level string literals; rejects f-strings, concatenation, function calls, attribute access, name references — never runs user code).
- `aimigrate.utils.templating`: `extract_variables`, `render` with strict missing-var detection, and a bulk `validate_suite_against_prompts` pre-flight check.
- `aimigrate validate` (hidden): end-to-end pre-flight that confirms config + suite + prompts are mutually compatible.

### Added — Phase 3 (model client + cache)

- `aimigrate.models.registry`: advisory, not gating. `resolve_model()` accepts any model id, falling back to prefix-inferred provider when not in the curated registry. `get_model()` strict variant kept for tests.
- `aimigrate.cache`: SQLAlchemy 2.0 + async `CacheStore` (sha256 keys, 7-day TTL) at `~/.aimigrate/cache.db`. Added `greenlet` runtime dep.
- `aimigrate cache clear` CLI command (under a `cache` sub-app).
- `aimigrate.models.client`: async `ModelClient` wrapping `litellm.acompletion` with uniform error mapping (`RateLimitError` / `AuthError` / `ModelError`), full-jitter exponential backoff retries (auth short-circuits), and per-call cost + token bookkeeping.
- `aimigrate.utils.cost`: `estimate_run_cost(...)` for pre-flight cost estimation with defensive fallbacks for missing LiteLLM pricing/token-counter data.
- `aimigrate test-call` (hidden): one-shot smoke test renderer with response/tokens/cost/latency Rich panel.

### Added — Phase 4 (run orchestrator)

- `aimigrate.runner` package: pydantic `RunState` + `Call` models, atomic `state.json` checkpointing, append-only `raw.jsonl`, crash-safe iterator, `validate_resume` with hash-drift detection.
- `run_orchestrator`: async loop with `asyncio.Semaphore(concurrency)`, cache-check → live-call → record per (prompt, example, role), Rich progress bar, $10 cost gate (skip with `--yes`), 50-call checkpoint cadence.
- `aimigrate run` command: `--from`, `--to`, `--config`, `--suite`, `--resume`, `--yes`. Outputs go to `.aimigrate/runs/<run-id>/`. Per-call errors are recorded but don't fail the run.

### Added — Phase 5 (evaluators)

- `aimigrate.evaluators` package: `Evaluator` Protocol, `PairedScore`, `EvalRecord`.
- Structural: `JsonSchemaEvaluator`, `RegexEvaluator`, `LengthEvaluator` (1.0 inside bounds, distance-decayed outside).
- Semantic: `CosineSimilarityEvaluator` (target preservation framing — source = 1.0, target = cosine to source).
- LLM judge: `PairwiseJudgeEvaluator` with order-randomization and defensive JSON parsing.
- `aimigrate evaluate <run-id>` command: pairs source/target calls, runs every configured evaluator, writes `scores.jsonl`.

### Added — Phase 6 (statistics)

- `aimigrate.analysis.slicing`: `build_slices` groups records by tag (with implicit `"all"` slice); `aggregates()` computes per-slice n/mean/std.
- `aimigrate.analysis.statistics`: per-comparison Shapiro-Wilk → paired t-test or Wilcoxon signed-rank, Cohen's d, 95% CI (analytical for t-test, bootstrap for Wilcoxon), Benjamini-Hochberg correction across the whole run, severity classification (critical/high/medium/low/improved/none/insufficient).
- `aimigrate analyze <run-id>` command: writes `analysis.json` and prints a Rich summary table.

### Added — Phase 7 (report)

- `aimigrate.reports.json`: `build_report_payload` + `ReportData` stitching state + raw + scores + analysis into a single payload. Persists `report.json`.
- `aimigrate.reports.templates/report.html.j2` + `report.css`: self-contained HTML (CSS inlined, zero external assets, no JS) with executive summary, per-prompt aggregate + slice tables, top-5 regressions side-by-side, methodology appendix.
- `aimigrate report <run-id>` command, with optional `--open` to launch in the user's default browser.

### Added — Phase 8 (polish + OSS)

- Documentation site (MkDocs Material) covering Getting Started, Configuration, Evaluators, Methodology, FAQ.
- `examples/simple/` runnable example.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, GitHub issue + PR templates.
- README polished with badges, status, full pipeline walkthrough, non-goals.

### Quality

- 371+ tests, ≥95% line coverage, `mypy --strict` clean, `ruff check` clean, `ruff format` clean.
- Mocked CI throughout (no API keys needed); `scripts/smoke_live.py` deferred for the user's manual live verification.
