# Capture SDK

The [evalshift-sdk](https://github.com/babaliauskas/evalshift-sdk) is a separate
package (`pip install evalshift-sdk`, import name `evalshift`) that you install
**inside your agent process**. It records what your agent actually did — model
calls, tool calls, the final output — as JSON capture files on disk. The CLI
promotes those captures into golden suites.

This page covers the CLI side of that contract. The full SDK guide lives in the
SDK repo: [DOCS.md](https://github.com/babaliauskas/evalshift-sdk/blob/main/DOCS.md),
dense LLM reference at <https://www.evalshift.dev/sdk-llms-full.txt>.

## The contract

```
your agent (evalshift-sdk)  →  .evalshift/captures/<suite>/cap_<hex>.json
                            →  evalshift capture sync
                            →  .evalshift/suites/<suite>/golden.jsonl + evalshift.yaml
```

- **Disk is the only interface.** The SDK never imports or calls the CLI, and
  the CLI never imports the SDK. Either one works without the other.
- **Separate virtual environments.** Both packages use the top-level import
  name `evalshift`. Install the SDK in your agent's venv and the CLI wherever
  you run evaluations — never in the same environment.
- **Off by default.** Nothing is recorded unless `EVALSHIFT_CAPTURE=1` is set,
  so the instrumentation is safe to leave in production code permanently.
- **No network.** The SDK writes local files only. Python 3.10+, stdlib-only.

## 1. Instrument the agent

Three primitives cover most agents: `@capture.agent` marks the agent boundary,
`@capture.tool` records a tool call, `record_model_call` records a completed
model call.

```python
from evalshift import capture, record_model_call


@capture.tool(name="issue_refund")
def issue_refund(order_id: str) -> dict:
    return {"status": "refunded", "order_id": order_id}


@capture.agent(suite="support_agent", redact=True)
def handle_ticket(query: str) -> str:
    response = client.messages.create(model="claude-sonnet-5", messages=messages)
    record_model_call(model_id="claude-sonnet-5", input=messages, output=response.text)
    ...
```

`redact` is a **required** keyword on `@capture.agent` — and on
`capture.agent_session`, `capture.agent_session_async` and
`EvalShiftCallbackHandler` — as of SDK 0.3.0. Captures hold the inside of a run
(tool arguments and results, model input and output), so every capture point has
to state its masking policy in the call itself. `True` applies the SDK's
`default_redactor` (emails, `sk-…`, `AKIA…`, `Bearer …`), `False` records
verbatim, and a `(value) -> value` callable does something custom; any other
value — `None` included — raises `TypeError`. Details:
[REDACTION.md](https://github.com/babaliauskas/evalshift-sdk/blob/main/docs/REDACTION.md).

Pass the **messages list** (not a bare string) as the model-call input where you
can: `capture sync` recovers conversation history verbatim from a messages list,
and only approximates it when all it has is a bare string. See
[Multi-turn conversations](conversations.md).

## 2. Record captures

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py
```

Each sampled invocation writes one file to
`.evalshift/captures/<suite>/cap_<hex>.json`. The directory stays bounded on its
own — identical-input runs are de-duplicated and each suite dir keeps the 200
newest captures. `EVALSHIFT_MAX_CAPTURES`, `EVALSHIFT_DEDUP`,
`EVALSHIFT_CAPTURE_TTL` and `EVALSHIFT_SAMPLE_RATE` tune that; they are read by
the SDK, in your agent's process.

## 3. Promote captures into suites

Run the CLI from the directory holding `.evalshift/` (or set `EVALSHIFT_DIR`):

```bash
evalshift capture list                  # what was recorded (--json for machine output)
evalshift capture sync                  # promote ALL captures → suites + wire evalshift.yaml
evalshift capture promote cap_ab12 --as refund_case_1
evalshift capture diff cap_ab12 cap_cd34
evalshift capture clean                 # delete already-promoted capture files
```

`capture sync` groups captures (by `conversation_id`/`turn_index` for
multi-turn), turns each into a suite example — first model input → `inputs`,
recorded tool calls → `expected_tools`, final output → `expected`, messages list
→ `history` — skips duplicate content across runs, writes
`.evalshift/suites/<suite>/golden.jsonl`, and rewrites the managed `suites:`
block in `evalshift.yaml`.

Strictness knobs for the derived tool expectations: `--strict-args`,
`--names-only`, `--tool-count`. `--tag` adds slice tags, `--print` previews the
config block without writing, `--keep-duplicates` disables dedup.

Full behaviour: [Configuration](configuration.md) and the
[Capturing from production](https://github.com/babaliauskas/evalshift-cli/blob/main/DOCS.md#capturing-from-production)
section of DOCS.md.

## 4. Evaluate a candidate against real behaviour

```bash
evalshift all --suite-name <suite> --to <candidate-model>
```

That is a normal EvalShift run — the only difference is that the suite came
from production traffic rather than hand-written examples.

## Related

- [Getting started](getting-started.md) — the capture-first `evalshift init` flow.
- [Multi-turn conversations](conversations.md) — how history is recovered.
- [Agent traces](traces.md) — for agents that run outside the SDK entirely
  (LangChain, another language): import full timelines with `evalshift traces import`.
