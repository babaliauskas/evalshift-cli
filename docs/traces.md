# Agent Traces

EvalShift can compare externally recorded agent timelines. The traces come from
your own runtime — EvalShift never runs your agent — and are imported into a
completed local run, after which `evaluate`, `analyze`, and `report` work as
usual. Import is local, but imported traces are not local-only: they travel to
hosted EvalShift inside the run bundle if you `push` (see [Hosted](#hosted)).

## Import

Run the normal model calls first so EvalShift has a run id and source/target
call pairs:

```bash
evalshift run --yes
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

`--source` and `--target` are both required. Add `--strict` to fail the import
when any completed run pair lacks a trace pair; without it those pairs are only
counted in the printed `missing pairs` line.

```bash
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl \
  --strict
```

The import command writes:

```text
.evalshift/runs/<run-id>/traces.jsonl
```

## JSONL Shape

Each line is one trace for one `(prompt_id, example_id, role)`:

```json
{"run_id":"r_20260609_trace1","prompt_id":"support_agent","example_id":"refund_017","role":"source","events":[{"type":"tool_call","sequence_index":0,"timestamp":"2026-06-09T12:00:00Z","metadata":{},"name":"check_refund_policy","arguments":{}},{"type":"tool_call","sequence_index":1,"timestamp":"2026-06-09T12:00:01Z","metadata":{},"name":"issue_refund","arguments":{"ticket_id":"T-1032"}}]}
```

The line itself carries `run_id`, `prompt_id`, `example_id` (all non-empty),
`role` (`source` or `target`), and `events` (defaults to `[]`).

Every event has `type`, `sequence_index` (integer, ≥ 0), `timestamp`, and
`metadata` (defaults to `{}`). Per type, on top of those:

| `type` | Required | Optional (default) |
| --- | --- | --- |
| `model_call` | `model_id` (non-empty) | `input`, `output` (`null`), `input_tokens`, `output_tokens`, `latency_ms` (`0`), `cost_usd` (`0.0`) |
| `tool_call` | `name` (non-empty) | `arguments` (`{}`), `call_id`, `parent_call_id` (`null`) |
| `tool_result` | `name` (non-empty) | `call_id`, `result`, `error` (`null`) |
| `retrieval` | `source` (non-empty) | `query` (`""`), `documents` (`[]`) |
| `guardrail` | `name` (non-empty), `verdict` (`pass` / `fail` / `warn` / `skipped`) | `reason` (`null`) |
| `final_output` | — | `text` (`""`) |
| `error` | `message` (non-empty) | `category` (`null`) |

`timestamp` must carry a UTC offset — `2026-06-09T12:00:00Z` or
`2026-06-09T14:00:00+02:00`. An offset timestamp is converted to UTC on import;
a naive one (no offset at all) is rejected with the file and line that carried
it. Assuming UTC for a naive value would silently relabel a trace recorded in
another zone, and the run bundle would then carry that as fact — the hosted
bundle contract requires UTC, so the ambiguity has to be resolved by the person
who has the trace, not by the CLI.

Numeric fields cannot be negative. Trace models are strict, like the rest of the
config contract: an unknown key anywhere in the line is an error, not a
warning.

EvalShift sorts events by `sequence_index` and rejects duplicate indices. A
`tool_result` with a `call_id` must match a `tool_call` that carried the same
`call_id` earlier in the trace.

## Evaluator

```yaml
evaluators:
  agent_trace:
    - name: trace_safety
      check_tool_order: true
      check_arguments: true
      check_missing_verification: true
      verification_tools: ["check_refund_policy"]
      dangerous_tools: ["issue_refund"]
```

The evaluator emits normal `scores.jsonl` records with failure categories such
as `TOOL_ORDER_DRIFT`, `ARGUMENT_VALUE_DRIFT`, `DANGEROUS_ACTION_DRIFT`, and
`MISSING_VERIFICATION_STEP`.

Debug commands become trace-aware:

```bash
evalshift diff case <run-id> <example-id>
evalshift inspect case <run-id> <example-id>
evalshift replay case <run-id> <example-id> --model target --trace
```

## Hosted

`evalshift bundle` / `evalshift push` carry imported traces into the run bundle
as one event stream per model side, so the hosted run-detail page renders the
timeline and not just the text. A new round starts at each `model_call`; events
before the first one stay in round 0.

Not everything travels verbatim. `model_call` `input` and `output` payloads are
excluded, and oversized content is shortened rather than dropped: a serialized
`tool_result.result` over 16 KB becomes a truncated preview, and a stream over
256 KB keeps its leading events and is flagged `truncated`. The full bundle
contract is in [Hosted alpha](hosted.md#bundle-and-push).
