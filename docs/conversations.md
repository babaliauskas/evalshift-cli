# Multi-turn conversations

EvalShift can evaluate multi-turn agent conversations, not just single-shot
prompts. A conversation is captured as one file per turn by the
`evalshift-sdk`; `evalshift capture sync` links the turns back together into
golden suite examples that carry a **teacher-forced** conversation prefix,
and `run` replays each turn against a candidate model with that prefix sent
verbatim.

## How the SDK records turns

Schema `1.1.0` of the capture envelope (the `evalshift-sdk` package, a
separate install from this CLI) added three optional fields so a multi-turn
conversation — one capture file per turn — can be linked back together:

| Field               | Type          | Meaning |
| ------------------- | ------------- | ------- |
| `conversation_id`   | `str \| None` | Shared identifier for every capture belonging to the same conversation. `None` for a standalone (single-turn) capture. |
| `turn_index`        | `int \| None` | This capture's position in the conversation (0-based). |
| `parent_capture_id` | `str \| None` | The `capture_id` of the immediately preceding turn's capture, if any. |

Record one capture per turn, passing fresh values each time:

```python
from evalshift import capture, record_model_call

# One capture per turn — pass fresh conversation_id/turn_index/parent_capture_id each time.
with capture.agent_session(
    suite="support_agent",
    redact=True,  # required — mask PII before it reaches disk
    agent_input=messages,  # the full messages list — see below
    conversation_id="conv_abc123",
    turn_index=2,
    parent_capture_id="cap_prev_turn_id",
):
    record_model_call(model_id="claude-opus-4-8", input=messages, output=reply)
```

(`capture.agent_session_async(...)` is the `async with` equivalent, for async
agent code.)

### The messages-list convention for `model_call.input`

`capture.model_call(input=...)` (and `record_model_call(input=...)`) accept
`Any` — the SDK doesn't validate the shape. The **convention** for a
multi-turn agent is to pass the complete per-turn context as a list of
role-tagged messages, not just the latest message in isolation:

```python
messages = [
    {"role": "system", "content": "You are a scheduling assistant."},
    {"role": "user", "content": "Can we move my appointment?"},
    {"role": "assistant", "content": "Sure — what time works?"},
    {"role": "user", "content": "1pm"},  # the current turn's user message
]
```

i.e. the system prompt, every prior turn, and the current user message. This
is what lets the CLI recover both the conversation history and the
current-turn input from a single capture file, without needing the sibling
turn captures — see `capture promote`/`capture sync` below.

## Promotion: `capture sync`

`evalshift capture sync` groups captures by `conversation_id`, orders them by
`turn_index`, and builds one `SuiteExample` per turn:

* If a turn's own capture recorded a full messages list (the convention
  above), its `history` is recovered **verbatim** from that list: everything
  before the last `user` message becomes `history`, and the last `user`
  message becomes the current turn's `inputs`.
* If a turn's capture only recorded a bare string or dict (no messages
  list), its `history` is **reconstructed** from the group's prior turns:
  each prior turn contributes a `user` message (its recovered current-turn
  text) and an `assistant` message (its recorded final output). If any prior
  turn recovered a `system` message from its own recorded messages list, the
  reconstructed history is seeded with it — the **earliest** turn's system
  prompt wins, and `capture sync` warns if later turns recorded a different
  one.

Both paths produce a `SuiteExample.history` list that `run` will replay
ahead of the current turn, but they are not equivalent: verbatim recovery
reproduces exactly what the model saw, while reconstruction is an
approximation stitched from sibling captures — assistant replies come from
recorded final outputs (intermediate tool exchanges are absent), and the
system prompt is only recovered when at least one turn in the conversation
recorded a full messages list. Prefer the messages-list convention above
when you can.

```bash
evalshift capture sync                              # promote every capture,
                                                      #   wire suites: into evalshift.yaml
evalshift run --suite-name support_agent --yes       # score a candidate model against it
```

`evalshift capture promote` (promoting a **single** capture id) does *not*
do this cross-capture reconstruction — it only recovers `history` from that
one capture's own messages list, if any. Promoting a multi-turn conversation
one capture at a time with `promote` produces examples with gaps in their
history; use `capture sync` to promote a whole conversation together. If you
promote a single mid-conversation capture and unpromoted sibling turns exist,
`capture promote` prints a warning pointing you at `capture sync`.

## Teacher-forced replay semantics

When `run` dispatches an example whose `history` is not `None`, it sends the
recorded history prefix **verbatim**, followed by the current turn's
rendered prompt as the final `user` message — via
`ModelClient.complete_messages()` (or `complete_messages_with_tools()` for
agent-style prompts) instead of the plain single-string `complete()` path.
The candidate model never generates its own intermediate turns; each turn is
scored independently against the *same* recorded prefix the source
conversation actually took, and only the **current turn's** output is
compared between source and target.

Recorded **tool calls and tool results replay too**. An `assistant` turn's
`tool_calls` are sent in the OpenAI wire shape
(`tool_calls[].function.arguments` as a JSON string) and each `tool` message
is keyed by the `tool_call_id` it answers; LiteLLM translates that into each
provider's own form. Without this, a candidate model was asked to continue an
agent conversation with every tool call and result deleted from its context,
then scored against what a model *with* that context did.

This is deliberate: replaying the literal recorded prefix keeps every turn's
comparison a clean paired measurement (the two models see byte-identical
context up to the turn being scored), which is what the statistical pipeline
(paired tests, Cohen's d, BH correction — see `docs/methodology.md`) assumes.
Re-driving whole conversations — letting the candidate model's own replies
feed into the *next* turn's prompt — is not supported yet. It would mean two
models drift onto genuinely different conversations after turn 0, which
breaks the paired-comparison contract this tool is built on. It may come
later as an explicitly different (and differently-analyzed) mode.

## `SuiteExample` field reference

Three fields on a golden suite row carry conversation state (see
`docs/configuration.md` for the full suite schema):

| Field             | Type                    | Meaning |
| ----------------- | ----------------------- | ------- |
| `history`         | list of `{role, content, ...}` or `null` | Conversation prefix replayed verbatim before the current turn. `null` means single-turn (no message-mode dispatch). An empty list still triggers message-mode dispatch with no prefix. `role` is one of `system`, `user`, `assistant`, `tool`; at most one `system` message is allowed, and if present it must be first. |
| `history[].tool_calls` | list of `{id, name, arguments}` or absent | Tool calls the turn emitted. Only valid on an `assistant` message. |
| `history[].tool_call_id` | string or absent    | The `tool_calls[].id` this result answers. **Required** on a `tool` message, forbidden elsewhere — a result that can't be paired with its call is rejected at load time, not at dispatch. |
| `conversation_id` | string or `null`        | Id of the recorded conversation this example's turn came from. Provenance only — not used by dispatch. |
| `turn_index`      | integer or `null`       | Zero-based position of this turn within its conversation. Surfaced in the HTML report as a `turn N` badge. |

Single-turn suites (no `history`/`conversation_id`/`turn_index` fields at
all) parse and run exactly as before — these fields are purely additive.

## Limitations

* **A tool result with no recorded id gets a synthetic one.** `tool` messages
  are replayed, but only if they can be paired with the call they answer.
  When the recording carried no `tool_call_id`, promotion assigns a
  positional one (`_pos<N>`) and warns — the result stays in the prefix, but
  record the provider's call id for exact pairing.
* **Reconstructed history is an approximation.** When a turn's capture only
  recorded a bare string, its prefix is stitched from sibling captures:
  assistant replies are the recorded final outputs (any intermediate tool
  calls and results are absent), and the system prompt is seeded only if
  some turn in the conversation recorded a full messages list — the earliest
  turn's system prompt is used, with a warning if later turns disagree. A
  conversation where *no* turn recorded a messages list replays with no
  system prompt at all.
* **No full-conversation re-drive.** As above — each turn is scored against
  the recorded prefix, not a prefix generated by the candidate model's own
  prior replies.
* **`capture promote` on a single capture doesn't reconstruct cross-capture
  history.** Use `capture sync` (or promote every turn of a conversation in
  one `capture sync` pass) to get verbatim/reconstructed history spanning
  multiple capture files. A lone `capture promote` only ever looks at that
  one capture's own recorded messages list.
