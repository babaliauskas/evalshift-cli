# CLI Agent Traces Design

## Summary

Phase 2 adds CLI-only support for bring-your-own-agent trace evaluation. EvalShift should accept externally recorded source and target agent traces, normalize them into local run artifacts, compare the traces with the same evaluator/analyze/report pipeline used today, and surface actionable trace diffs in debug commands and the offline HTML report.

This phase does not add SDKs, hosted features, web dashboards, production observability, or an agent runtime. EvalShift remains the local migration safety layer.

## Goals

- Define a first-class agent trace schema for local CLI artifacts.
- Import and validate external trace JSONL files for completed runs.
- Compare source and target agent timelines per `(prompt_id, example_id)`.
- Emit normal `EvalRecord` rows so existing analysis, migration policy, and reports continue to work.
- Extend `diff case`, `inspect`, `report.json`, and `report.html` to explain trace regressions.
- Preserve backward compatibility for existing prompt/output and current `ToolTrace` workflows.

## Non-Goals

- No Python SDK.
- No TypeScript SDK.
- No hosted/web implementation.
- No OpenTelemetry exporter implementation.
- No agent builder, workflow runner, or tool execution runtime.
- No generic observability ingestion outside migration test runs.

## Current Repo Context

The repo already has provider-level tool-call traces:

- `src/evalshift/evaluators/tool_models.py` defines `ToolCall` and `ToolTrace`.
- `src/evalshift/runner/models.py` stores `Call.trace` in `raw.jsonl` for tool-aware prompts.
- `tool_selection`, `tool_arguments`, and `tool_trace_structure` compare provider-parsed tool traces.
- `reports/json.py` and `report.html.j2` can render side-by-side tool trace diffs.
- `inspect`, `diff case`, and `replay case` provide recorded-artifact debugging.
- Migration policy and failure taxonomy work already exists in `analysis/policy.py` and evaluator metadata.

Phase 2 should extend these capabilities to multi-event agent traces without replacing them.

## Trace Artifact Model

Add a new package:

```text
src/evalshift/traces/
  __init__.py
  models.py
  loader.py
  diff.py
```

`models.py` defines strict Pydantic models:

- `AgentTrace`
- `TraceEvent`
- `ModelCallEvent`
- `ToolCallEvent`
- `ToolResultEvent`
- `RetrievalEvent`
- `GuardrailEvent`
- `FinalOutputEvent`
- `ErrorEvent`

Each imported trace belongs to one side of one example:

```json
{
  "run_id": "r_20260609_ab12cd",
  "prompt_id": "support_agent",
  "example_id": "refund_017",
  "role": "target",
  "events": []
}
```

Each event uses a discriminated `type` field and stable ordering:

```json
{
  "type": "tool_call",
  "name": "issue_refund",
  "arguments": {"ticket_id": "T-1032", "amount": 42.0},
  "sequence_index": 3,
  "timestamp": "2026-06-09T12:00:00Z",
  "metadata": {"step": "refund"}
}
```

Required event fields:

- `type`
- `sequence_index`
- `timestamp`
- `metadata`

Event-specific fields:

- `model_call`: `model_id`, `input`, `output`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`
- `tool_call`: `name`, `arguments`, `call_id`, `parent_call_id`
- `tool_result`: `name`, `call_id`, `result`, `error`
- `retrieval`: `source`, `query`, `documents`
- `guardrail`: `name`, `verdict`, `reason`
- `final_output`: `text`
- `error`: `message`, `category`

Validation rules:

- `sequence_index` values are unique and non-negative within a trace.
- Events are normalized into ascending `sequence_index` order.
- `tool_result.call_id`, when present, should reference a previous `tool_call.call_id`.
- `role` must be `source` or `target`.
- `prompt_id` and `example_id` must match examples in the run being imported.
- Unknown fields are rejected so trace contract drift is visible.

## Import Command

Add a `traces` Typer sub-app:

```bash
evalshift traces import <run-id> --source source-traces.jsonl --target target-traces.jsonl
```

Command behavior:

1. Load `.evalshift/runs/<run-id>/state.json` and `raw.jsonl`.
2. Validate both JSONL files line by line.
3. Ensure every imported trace references known `prompt_id` and `example_id` values from the run.
4. Ensure imported `role` values match the side implied by `--source` or `--target`.
5. Write a normalized append-free artifact:

```text
.evalshift/runs/<run-id>/traces.jsonl
```

6. Print an import summary:

```text
Imported traces for run r_...
source: 40 traces
target: 40 traces
missing pairs: 0
artifact: .evalshift/runs/<run-id>/traces.jsonl
```

Failure behavior:

- Invalid JSONL reports file path and line number.
- Schema validation errors are grouped and rendered similarly to config/suite errors.
- Missing source/target pairs warn by default and fail only when `--strict` is passed.
- Import never mutates `raw.jsonl`, `scores.jsonl`, or `analysis.json`.

## Trace Evaluation

Add an evaluator family:

```yaml
evaluators:
  agent_trace:
    - name: trace_safety
      check_tool_order: true
      check_arguments: true
      check_missing_verification: true
      verification_tools: ["check_refund_policy", "verify_order_status"]
      dangerous_tools: ["issue_refund", "delete_record", "send_email"]
```

Add config models in `src/evalshift/config/models.py`:

- `AgentTraceEvaluatorConfig`
- `evaluators.agent_trace: list[AgentTraceEvaluatorConfig] | None`

Evaluation behavior:

- `run_evaluate` loads `traces.jsonl` when `agent_trace` evaluators are configured.
- Each source/target trace pair produces an `EvalRecord`.
- Records are written to existing `scores.jsonl`; no new analysis stage is needed.
- Existing `analyze`, migration policy, and report severity classification remain the statistical layer.

Scoring dimensions:

- Tool sequence similarity.
- Missing expected verification before dangerous tool calls.
- Extra dangerous tool calls.
- Argument value drift for same-name matched tool calls.
- Tool result error drift.
- Final output presence and refusal-like error changes.

Failure categories:

- `TOOL_SELECTION_DRIFT`
- `TOOL_ORDER_DRIFT`
- `ARGUMENT_VALUE_DRIFT`
- `DANGEROUS_ACTION_DRIFT`
- `MISSING_VERIFICATION_STEP`
- `UNNECESSARY_TOOL_CALL`
- `TOOL_RESULT_DRIFT`
- `TRACE_SCHEMA_FAILURE`

These should integrate with the existing failure taxonomy metadata used by reports.

## Trace Diffing

Add reusable trace diff logic in `src/evalshift/traces/diff.py`.

Output model:

- Matched events.
- Missing source events.
- Extra target events.
- Reordered tool calls.
- Argument field deltas.
- Verification gaps.
- Dangerous action flags.

Use this model in:

- `evalshift diff case <run-id> <example-id>`
- report payload assembly
- HTML rendering
- tests

The CLI diff should render a compact timeline:

```text
source                         target
1 model_call claude...          1 model_call gpt...
2 tool_call check_policy        - missing
3 tool_call issue_refund        2 tool_call issue_refund  ARGUMENT_VALUE_DRIFT
4 final_output                  3 final_output
```

## Report Changes

Extend `reports/json.py`:

- Load `traces.jsonl` if present.
- Attach `source_agent_trace` and `target_agent_trace` to top regression rows.
- Attach computed `trace_diff` to rows from `agent_trace` evaluators.
- Include aggregate top regression causes from agent trace metadata.

Extend `report.html.j2`:

- Render trace timelines for agent trace regressions.
- Show missing/extra/reordered events with clear labels.
- Show argument drift inline with field names and source/target values.
- Keep existing `ToolTrace` rendering for current tool-call evaluators.

HTML remains single-file, no external assets, no JavaScript.

## Debug Command Changes

`evalshift inspect`:

- `evalshift inspect <run-id> --failed` should include agent trace failures.
- `evalshift inspect case <run-id> <example-id>` should show trace summary when available.

`evalshift diff case`:

- Prefer agent trace diff when `traces.jsonl` has a pair for the case.
- Fall back to current text diff when no agent trace exists.

`evalshift replay case`:

- Keep current recorded-output behavior.
- Add `--trace` to print the imported target/source trace as normalized JSON.

## Compatibility

Existing users without `traces.jsonl` are unaffected.

Current provider-level `ToolTrace` stays in `Call.trace`. Agent traces are broader user-provided timelines. Where useful, the current `ToolTrace` can be adapted into an `AgentTrace` internally for diff rendering, but that adapter should not change the persisted `raw.jsonl` format.

`agent_trace` evaluators require imported traces. If configured and `traces.jsonl` is missing, `evaluate` should fail with a clear message:

```text
agent_trace evaluators require imported traces.
Run: evalshift traces import <run-id> --source ... --target ...
```

## Testing Strategy

Unit tests:

- Trace model validation and normalization.
- JSONL loader line-numbered errors.
- Import command success, missing pairs, strict failure, and unknown example failure.
- Trace diff matching, missing, extra, reorder, argument drift, and verification gap cases.
- Agent trace evaluator scoring and failure metadata.
- Config model parsing for `evaluators.agent_trace`.
- Debug command rendering/fallback behavior.
- Report payload and HTML rendering with agent traces.

Integration tests:

- Fixture run with imported source/target traces.
- `traces import -> evaluate -> analyze -> report`.
- Existing simple and tool pipeline tests stay green.

Verification commands:

```bash
pytest tests/unit/test_trace_models.py
pytest tests/unit/test_trace_import_command.py
pytest tests/unit/test_agent_trace_evaluator.py
pytest tests/unit/test_reports.py
pytest tests/integration/test_agent_trace_pipeline.py
ruff check .
ruff format --check .
mypy --strict src/evalshift
```

## Documentation

Add:

- `docs/traces.md`
- Configuration reference section for `evaluators.agent_trace`
- Getting-started note for CLI-only bring-your-own-agent traces
- Example trace JSONL files under `examples/agent-traces/`

Docs should make the product boundary explicit: EvalShift evaluates traces supplied by the user's agent; it does not run the agent.

## Rollout

Implement in four focused increments:

1. Trace schema and loader.
2. Trace import command and artifact storage.
3. Agent trace evaluator and config.
4. Debug/report rendering and example docs.

Each increment should preserve existing CLI behavior and keep the full test suite green.
