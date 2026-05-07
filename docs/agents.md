# Agent migrations (v0.2)

AIMigrate v0.2 extends the v0.1 pipeline to compare **agent behaviour** —
which tools the model called, what arguments it passed, and how it
sequenced them — across two model versions.

The killer scenario it catches:

> A team migrates a customer-support agent from Gemini 2.5 Flash to
> 3.1 Flash-Lite. The new model silently stops calling
> `notify_security_team` on sensitive requests. Text-only eval
> reports green; v0.2 marks it CRITICAL and blocks the migration.

## What's new in v0.2

* **`ToolTrace` data model**: provider-agnostic, populated from
  Anthropic / OpenAI / Gemini responses.
* **Three new evaluators**: `tool_selection`, `tool_arguments`,
  `tool_trace_structure`.
* **Suite extension**: optional `expected_tools`, `expected_tool_count`,
  `expected_no_tools`, `expected_parallel` per example.
* **Config extension**: `prompts[].tools_path` makes a prompt
  agent-style.
* **HTML report**: side-by-side trace diffs in place of text panes
  for tool-evaluator regressions.

## Walkthrough

`aimigrate init` ships a complete agent project as the default
scaffold — six customer-support tools (`search_orders`,
`lookup_customer`, `issue_refund`, `update_order_status`,
`send_email`, `notify_security_team`) and a 40-row golden suite
across five slices (`security`, `routine`, `refund`,
`customer_lookup`, `text_only`):

```bash
mkdir my-agent-eval && cd my-agent-eval
export GOOGLE_API_KEY=...                 # or OPENAI_API_KEY / ANTHROPIC_API_KEY

aimigrate init                            # writes aimigrate.yaml + prompts.py +
                                          #        tools.yaml + golden.jsonl
aimigrate doctor
aimigrate run --yes                       # uses Gemini defaults from the yaml
RUN_ID=$(ls .aimigrate/runs/ | head -1)
aimigrate evaluate $RUN_ID
aimigrate analyze $RUN_ID
aimigrate report $RUN_ID --open
```

The same files exist as a checked-in reference at `examples/agent/` if
you want to read or copy them without scaffolding.

`aimigrate run` notices `tools_path` on the prompt and dispatches via
`ModelClient.complete_with_tools` — each `Call` row in `raw.jsonl`
carries a parsed `ToolTrace`. `aimigrate evaluate` then runs
`tool_selection` (and any other configured tool evaluators) against the
per-example `expected_tools` ground truth.

## Configuration

A minimal agent config:

```yaml
version: 1
prompts:
  - id: routing
    detection: python_string
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]
    tools_path: tools.yaml          # makes this an agent prompt

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-3.1-flash-lite-preview
  judge_model: gemini-2.5-pro

evaluators:
  tool_selection:
    - name: routing_selection
      mode: expected                # match against example.expected_tools
      severity_floor: high

  # Optional — enable as needed:
  # tool_arguments:
  #   - name: routing_args
  # tool_trace_structure:
  #   - name: routing_structure
```

> **Note:** `structural.length` is intentionally **not** in the
> scaffolded config. Agent runs frequently produce empty `final_text`
> (the model returned only tool calls), which makes the length
> evaluator score 0/0 across every routine row — pure noise. Add it
> back manually only for prompts that produce text.

The tools file (`tools.yaml`) accepts either Anthropic-shape
(`name` / `description` / `input_schema`) or OpenAI-shape
(`{ "type": "function", "function": {...} }`) entries. `aimigrate run`
serialises them in the right shape per provider.

## Suite ground truth

```jsonl
{"id": "ex_security_01", "inputs": {"query": "..."}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team"}]}
{"id": "ex_text_01", "inputs": {"query": "what is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true}
```

## Picking an evaluator

| Evaluator              | Use when                                                |
| ---------------------- | ------------------------------------------------------- |
| `tool_selection`       | You care about *which* tools fire (most common).        |
| `tool_arguments`       | You care about *what* the model passes to each tool.    |
| `tool_trace_structure` | You care about call counts, parallelism, or refusals.   |

You can run multiple at once. Each becomes an independent comparison
in `analysis.json`, with the existing Benjamini-Hochberg correction
already adjusting for the multi-test count.

## Troubleshooting

* **"prompt has tools_path but no tool_* evaluators are configured"** —
  add at least one `tool_*` block under `evaluators:` or remove the
  `tools_path`. `aimigrate doctor` warns about this.
* **Bimodal score distribution** — tool evaluators often produce
  scores at exactly 0 or 1. The analysis layer's Shapiro-Wilk
  fallback routes these through Wilcoxon signed-rank automatically.
* **"no matched calls between source and target"** — the
  `tool_arguments` evaluator scores a regression when the target
  doesn't reuse any of the same tool names as the source. Check
  `tool_selection` first to triage.
