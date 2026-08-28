# Temperature value rejection — runtime adaptation design

Date: 2026-08-23
Status: approved, not yet implemented

Companion to `2026-08-14-temperature-determinism-design.md`. That spec covers
*withdrawal* — a provider removing the `temperature` parameter, detected via the
`honors_temperature` probe in `models/capabilities.py`. This spec covers the failure
class that probe explicitly does not: a model that *advertises* `temperature` while
rejecting every value except its default.

## Problem

A CI run with `judge_model: gpt-5.6-terra` fails every judge call:

> `litellm.BadRequestError: OpenAIException - Unsupported value: 'temperature' does
> not support 0.0 with this model. Only the default (1) value is supported.`

EvalShift sends `temperature=0.0` on every call (`models/client.py`) and relies on
`drop_params=True` to survive models that reject non-default values. The comment at
`client.py:331-334` claims this works ("Reasoning-tier models reject temperature != 1;
drop_params tells LiteLLM to drop unsupported params instead of erroring"), and
`capabilities.py:20-23` repeats it ("that case is already handled by `drop_params`").

Both claims are false for the models that matter. Verified against litellm 1.98.0:

- **o-series names** (`o3`, `o4-mini`, …): litellm's `OpenAIOSeriesConfig` special-cases
  them and does drop `temperature != 1` under `drop_params`. The comment is true here.
- **gpt-5.x reasoning models**: litellm's `OpenAIGPT5Config` checks the model map's
  `supports_none_reasoning_effort` flag. For `gpt-5.6-terra` the flag is `True`, so
  litellm passes `temperature=0.0` through *untouched*, betting that the caller set
  `reasoning_effort: "none"`. EvalShift never sets `reasoning_effort`, the live API
  defaults to a reasoning tier, and the call 400s. Confirmed live:
  `get_optional_params(model="gpt-5.6-terra", temperature=0.0, drop_params=True)`
  returns `temperature: 0.0` in the outgoing params.
- **`honors_temperature`** returns `True` for these models — `temperature` *is* in
  `supported_openai_params`. The probe detects withdrawal, not value constraints, and
  says so in its docstring. Nothing detects value constraints.

A second defect compounds it: the 400 maps to a generic `ModelError` in
`_map_exception`, and `_dispatch_with_retry` retries it with **identical kwargs** up to
`max_attempts`. A deterministic failure burns the whole retry budget per call — the CI
log shows `attempt 2 failed (ModelError); retrying in 0.99s` for every example.

## Why not static knowledge

litellm's own model map is the thing that is wrong here — its
`supports_none_reasoning_effort: True` entry is precisely what routes the bad value
through. Extending the EvalShift registry with reasoning-model prefixes repeats the
mistake the 2026-08-14 spec rejected: passthrough ids (this repo's registry carries
only `gpt-4o` / `gpt-4o-mini` for OpenAI) would miss every future reasoning id until a
release catches up. A run-start probe call per model was also rejected: it costs a real
API call on every run forever, where runtime adaptation costs one failed call per
affected model per process.

## Decisions

| Question | Decision | Why |
| --- | --- | --- |
| Where to fix | `_dispatch_with_retry` in `models/client.py` | Single choke point under `complete`, `complete_messages`, `complete_with_tools`, `complete_messages_with_tools` — judge, replay, tools, and insights paths all inherit it |
| How to detect | At call time, from the provider's own rejection | The provider is the only source of truth; every static source (litellm map, registry) has been shown wrong |
| Detection predicate | Underlying exception is `litellm.BadRequestError` AND its message contains `temperature` (case-insensitive) AND `"temperature" in kwargs` | All three or no intercept. A temperature-flavored 400 on a call that never sent the parameter is somebody else's bug and must surface |
| Adaptation | Pop `kwargs["temperature"]`, redispatch immediately; does **not** consume a retry attempt | The adapted call is a different request, not a retry of the failed one. Attempt counting, backoff, `AuthError` short-circuit, and exhaustion behavior are unchanged |
| Memoization | Per-client set of rejected canonical model ids; later calls to a listed model omit `temperature` before dispatch | One failed call per model per process. Matches the log-dedupe precedent at `client.py:54` |
| Reporting | Client exposes the set read-only; the orchestrator merges it into `state.non_deterministic_models` (dedup, stable order) before the final state write and report build | Reuses the existing surface end-to-end: `runner/models.py:159`, HTML banner (`reports/html.py`), JSON (`reports/json.py`), economics note (`reports/economics.py`). One concept — "sampling was not controlled for these models" — one banner |
| Cache keys | Unchanged — keyed on the *requested* temperature | Same precedent as the Gemini ignored-temperature case: identical requests produce identical keys; the banner carries the honesty. Changing key composition would orphan every existing cache entry |
| Logging | `log.warning` once per model on first rejection | States the model, that `temperature` was withdrawn from its calls, and that outputs are non-deterministic |
| False comments | Rewrite `client.py:331-334` and `capabilities.py:20-23` | Both must describe the real mechanism: litellm saves o-series names only; value constraints on other models are handled by this adaptation |

## What this changes for users

A judge (or source/target/insights) model that rejects `temperature=0.0` now works:
first call to it fails once, the client adapts, the run completes. The report and JSON
carry the model in `non_deterministic_models`, so the loss of the control variable is
visible in the same banner Gemini withdrawal uses. Previously every call to such a
model failed after the full retry budget and, for a `blocking: true` judge, poisoned
the gate.

No config surface is added. No behavior changes for models that accept `temperature`.

## Out of scope

- **General deterministic-400 retry policy.** `BadRequestError`s that are not the
  temperature case still burn the retry budget. Real, separate change — noting it here
  so it is not silently forgotten.
- **Registry entries for gpt-5.x / o-series ids.** Complementary static data, not
  needed for correctness once adaptation exists.
- **Setting `reasoning_effort: "none"` to keep temperature control on gpt-5.1+.**
  Tempting, but it mutates the model's reasoning behavior under test — the same
  asymmetric-confound argument that rejected prompt injection in the 2026-08-14 spec.

## Tests (written first, per TDD)

Unit, `tests/unit/test_model_client.py` (or the module's existing home), with a faked
`litellm.acompletion`:

1. **Adapt and succeed** — first call raises `BadRequestError` naming temperature;
   client redispatches without `temperature`, returns the second response; model id
   lands in the rejected set; attempt counter unaffected (a subsequent transient error
   still gets the full budget).
2. **Preemptive omission** — second call to the same model through the same client
   never includes `temperature` in kwargs and makes exactly one dispatch.
3. **Non-temperature 400 unaffected** — `BadRequestError` with an unrelated message
   follows the existing retry-then-raise path.
4. **No-temperature call not intercepted** — temperature-naming 400 on kwargs without
   `temperature` raises; rejected set stays empty.
5. **AuthError short-circuit preserved.**
6. **Warning logged once** per model across repeated rejections.

Orchestrator level:

7. **Banner merge** — runtime-rejected model appears in `state.non_deterministic_models`
   exactly once (dedup against probe-detected), and in the JSON report output.

Integration:

8. **Judge through client** — `llm_judge` against a faked value-rejecting model
   produces a verdict instead of `EvaluatorError`.

Gate: `make ci` (ruff + `mypy --strict` + pytest) green before any done-claim.

## Documentation

- `DOCS.md` determinism section: value-rejection case added next to withdrawal.
- `CHANGELOG.md`: Fixed entry under Unreleased.
- `llms-full.txt`: regenerate/update if it states the `drop_params` claim.
