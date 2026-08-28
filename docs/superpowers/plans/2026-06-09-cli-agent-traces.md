# CLI Agent Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLI-only bring-your-own-agent trace import, comparison, debugging, and report support.

**Architecture:** Add a new `evalshift.traces` package for strict trace models, JSONL loading, pairing, and diffing. Add an `agent_trace` evaluator family that consumes imported `traces.jsonl` pairs and emits existing `EvalRecord`s so `evaluate -> analyze -> report` remains the pipeline. Extend debug commands and report payload/rendering to prefer imported agent traces when present.

**Tech Stack:** Python 3.14, Typer, Pydantic v2, Rich, pytest, existing EvalShift runner/evaluator/report modules.

---

### Task 1: Trace Models And Loader

**Files:**
- Create: `src/evalshift/traces/__init__.py`
- Create: `src/evalshift/traces/models.py`
- Create: `src/evalshift/traces/loader.py`
- Test: `tests/unit/test_trace_models.py`

- [ ] **Step 1: Write failing trace model tests**

Create `tests/unit/test_trace_models.py` with tests for model validation, event ordering, duplicate `sequence_index`, invalid tool-result references, JSONL line-numbered load errors, and pair indexing.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_trace_models.py -q`
Expected: FAIL because `evalshift.traces` does not exist.

- [ ] **Step 3: Implement trace models and loader**

Implement strict Pydantic models with discriminated event types, `TRACES_FILENAME = "traces.jsonl"`, `load_traces_jsonl`, `write_traces_jsonl`, `index_traces`, and `pairs_for_prompt_examples`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_trace_models.py -q`
Expected: PASS.

### Task 2: Trace Import Command

**Files:**
- Create: `src/evalshift/cli/commands/traces.py`
- Modify: `src/evalshift/cli/main.py`
- Test: `tests/unit/test_trace_import_command.py`

- [ ] **Step 1: Write failing CLI import tests**

Create tests that scaffold a completed run, import source/target trace JSONL files, assert `.evalshift/runs/<run-id>/traces.jsonl` is written, assert bad example ids fail, and assert `--strict` fails when a pair is missing.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_trace_import_command.py -q`
Expected: FAIL because `evalshift traces` is not registered.

- [ ] **Step 3: Implement import command**

Add `traces_app = typer.Typer(...)`, `import_traces(...)`, run artifact validation via `read_state` and `iter_calls`, source/target role validation, normalized write, and Rich summary output.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_trace_import_command.py -q`
Expected: PASS.

### Task 3: Agent Trace Evaluator

**Files:**
- Modify: `src/evalshift/config/models.py`
- Modify: `src/evalshift/evaluators/failures.py`
- Create: `src/evalshift/evaluators/agent_trace.py`
- Modify: `src/evalshift/cli/commands/evaluate.py`
- Test: `tests/unit/test_config_models.py`
- Test: `tests/unit/test_agent_trace_evaluator.py`
- Test: `tests/unit/test_evaluate_command.py`

- [ ] **Step 1: Write failing config and evaluator tests**

Add tests for parsing `evaluators.agent_trace`, scoring missing verification before dangerous tools, argument drift, extra dangerous tools, and evaluate failure when `agent_trace` is configured without imported traces.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_config_models.py tests/unit/test_agent_trace_evaluator.py tests/unit/test_evaluate_command.py -q`
Expected: FAIL because config/evaluator support does not exist.

- [ ] **Step 3: Implement config, failure constants, evaluator, and evaluate dispatch**

Add `AgentTraceEvaluatorConfig`, failure constants, `AgentTraceEvaluator.score_trace_pair`, and a separate agent-trace branch in `run_evaluate` that loads `traces.jsonl` only when configured.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_config_models.py tests/unit/test_agent_trace_evaluator.py tests/unit/test_evaluate_command.py -q`
Expected: PASS.

### Task 4: Trace Diff And Debug Commands

**Files:**
- Create: `src/evalshift/traces/diff.py`
- Modify: `src/evalshift/cli/commands/debug_artifacts.py`
- Modify: `src/evalshift/cli/commands/diff.py`
- Modify: `src/evalshift/cli/commands/inspect.py`
- Modify: `src/evalshift/cli/commands/replay.py`
- Test: `tests/unit/test_trace_diff.py`
- Test: `tests/unit/test_debug_commands.py`

- [ ] **Step 1: Write failing diff/debug tests**

Add tests for missing/extra/reordered trace diff items, argument field deltas, `diff case` preferring trace timelines, `inspect case` showing a trace summary, and `replay case --trace` printing normalized JSON.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_trace_diff.py tests/unit/test_debug_commands.py -q`
Expected: FAIL because trace diff/debug support does not exist.

- [ ] **Step 3: Implement trace diff and debug rendering**

Add reusable diff models and renderers, load traces from debug helpers, and wire debug commands to fall back to existing text behavior when traces are absent.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/unit/test_trace_diff.py tests/unit/test_debug_commands.py -q`
Expected: PASS.

### Task 5: Report And Docs

**Files:**
- Modify: `src/evalshift/reports/json.py`
- Modify: `src/evalshift/reports/templates/report.html.j2`
- Modify: `src/evalshift/reports/templates/report.css`
- Modify: `docs/configuration.md`
- Modify: `docs/getting-started.md`
- Create: `docs/traces.md`
- Create: `examples/agent-traces/evalshift.yaml`
- Create: `examples/agent-traces/golden.jsonl`
- Create: `examples/agent-traces/source-traces.jsonl`
- Create: `examples/agent-traces/target-traces.jsonl`
- Test: `tests/unit/test_reports.py`
- Test: `tests/integration/test_agent_trace_pipeline.py`

- [ ] **Step 1: Write failing report/integration tests**

Add tests that report payload includes `source_agent_trace`, `target_agent_trace`, and `trace_diff` for `agent_trace` regressions, HTML renders trace rows, and `traces import -> evaluate -> analyze -> report` works on fixtures.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/test_reports.py tests/integration/test_agent_trace_pipeline.py -q`
Expected: FAIL because report/docs fixtures are not implemented.

- [ ] **Step 3: Implement report serialization/rendering and docs/examples**

Load `traces.jsonl` in report payload, attach trace data to top regressions, add timeline rendering, document CLI trace usage, and add a minimal example project.

- [ ] **Step 4: Run targeted tests to verify pass**

Run: `uv run pytest tests/unit/test_reports.py tests/integration/test_agent_trace_pipeline.py -q`
Expected: PASS.

### Task 6: Final Verification

**Files:**
- Verify whole repo.

- [ ] **Step 1: Run unit/integration targets**

Run: `uv run pytest tests/unit/test_trace_models.py tests/unit/test_trace_import_command.py tests/unit/test_agent_trace_evaluator.py tests/unit/test_trace_diff.py tests/unit/test_debug_commands.py tests/integration/test_agent_trace_pipeline.py -q`
Expected: PASS.

- [ ] **Step 2: Run lint, format check, and type check**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/evalshift
```

Expected: all PASS.

- [ ] **Step 3: Update changelog if needed**

Add a concise `CHANGELOG.md` entry under `## [Unreleased]` for CLI agent trace import/evaluation.
