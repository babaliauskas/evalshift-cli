# Agent example (v0.2)

A small customer-routing agent that EvalShift's v0.2 tool-call evaluators
exercise end to end.

## What's here

| File              | Purpose                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| `evalshift.yaml`  | Run config with a single agent prompt + the `tool_selection` evaluator.|
| `prompts.py`      | The agent's system prompt, referenced via `python_string`.             |
| `tools.yaml`      | Three tools: `search_orders`, `notify_security_team`, `send_email`.    |
| `golden.jsonl`    | 12 examples mixing security, routine, and text-only cases.             |

## Run it

```bash
cd examples/agent
export GOOGLE_API_KEY=...           # or any provider you set up
evalshift run --yes --from gemini-2.5-flash --to gemini-2.5-pro
RUN_ID=$(ls .evalshift/runs/ | head -1)
evalshift evaluate $RUN_ID
evalshift analyze $RUN_ID
evalshift report $RUN_ID --open
```

`evalshift run` notices `tools_path` on the prompt and dispatches via
`ModelClient.complete_with_tools` — each `Call` row in `raw.jsonl` carries
a parsed `ToolTrace`. `evalshift evaluate` then runs `tool_selection`
against the per-example `expected_tools` ground truth.

## What the example demonstrates

* **Tool-call regression detection.** If you set `target_model` to a
  weaker model, the report will flag examples where the target stops
  calling `notify_security_team` on security-flagged inputs.
* **Slice analysis.** The `security` and `routine` slices report
  per-segment regressions, so a model that's fine on routine traffic
  but worse on security can't hide.
* **Text-only fallback.** The `text_only` examples check that the
  model knows when *not* to call a tool — also a real regression mode.
