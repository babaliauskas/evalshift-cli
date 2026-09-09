# External Review Response — Findings and Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do one task at a time; each task is independently shippable.

**Goal:** Act on an external review of EvalShift (2026-09-08) that raised nine weaknesses. Each point was verified against the source of `evalshift-sdk` and `evalshift-cli`. This document records the verdict per point, then lays out the work in phases ordered by *how valid the point is* and *how much it matters*, cheapest-and-most-misleading fixes first.

**Scope:** Two repositories. Tasks are tagged `[cli]` or `[sdk]`. Cross-repo tasks list both.

**Tech Stack:** Python 3.10+ (sdk, stdlib-only runtime) / 3.11+ (cli), pydantic, litellm, pytest, `mypy --strict`, ruff.

---

## Part 1 — Findings

Verdicts, from most to least valid. Evidence lines are current as of 2026-09-08.

| # | Review point | Verdict | Importance | Phase |
|---|---|---|---|---|
| 4 | Capture and replay disagree on scope | **Valid, understated** | High — a documented workflow is broken | 0, 2 |
| 3 | Tool calls recorded only from executed functions | **Valid** | High — argument-drift scoring uses the wrong ground truth | 3 |
| 8 | LiteLLM hides provider edges (tool_choice, strict, structured output) | **Valid** | High — replays silently drop constraints | 4 |
| 1 | `evalshift` namespace collision | **Valid** (author already deferred it as D1-followup) | Medium-high — biggest setup friction, but breaking to fix | 6 |
| 2 | No provider client wrappers | **Valid on substance**; wrong that tokens gate promotion, wrong that LlamaIndex is on a roadmap | Medium-high — adoption | 5 |
| 6 | One sample per example, no power warning | **Half valid**: single-sample design is real; "no underpowered warning" is false | Medium | 7 |
| 7 | Judge self-preference unaddressed | **Half valid**: no code mitigation; warning already in scaffold + docs | Medium-low | 7 |
| 5 | Drift not correctness; 0.95 floor false alarms | **Half valid**: drift framing right; default init writes 0.75 and makes semantic/judge advisory | Low — already mitigated | 1 |
| 9 | Config carries two mental models | **Mostly invalid**: `prompts` is required and on the hot path; `captures` is not a config block. Real residue: one docs default mismatch, no `suites` example | Low | 0, 1 |

### Evidence summary per point

**#4 Capture vs replay.**
- SDK `docs/DECISIONS.md:31-34` says tool results MUST be stored as fixtures keyed by `call_id` + input hash for CLI replay; `trace/serialize.py` `build_fixture_table` implements it. No CLI code reads it (`grep fixture_table evalshift-cli/src` is empty).
- CLI runner makes one model call per example, no loop: `runner/orchestrator.py:95` (`WorkItem`: "a single LLM call"), `docs/agents.md:167-170`.
- Promotion defaults to round one: `captures/promote.py:91` `rounds: Literal["first","all"] = "first"`; `suite/models.py:196-199` says later rounds are "retained for teacher-forced multi-round replay" which does not exist.
- **Broken doc:** `evalshift-sdk/examples/support_agent/README.md:25` runs `evalshift run --offline --fixtures fixtures.jsonl`. Neither flag exists (`cli/commands/run.py`, `all.py`).
- README `README.md:253-256` claims detection of "*how* it sequences" tools; in the capture→run path this is in-order matching within one response only.

**#3 Executed-only tool calls.**
- Only recorder is `@capture.tool` → `capture/api.py:298-332` `_run_tool`, arguments bound from the Python signature (`_bind`, `:158-165`).
- `trace/models.py:24-45` `ModelCallEvent` has `input`/`output`/`tools_offered` but no requested-calls field. `grep 'tool_calls|AIMessage|function_call' evalshift-sdk/src` is empty.
- LangChain adapter records `on_tool_start`/`on_tool_end` (`adapters/langchain.py:493-528`), never `message.tool_calls` in `on_llm_end` (`:465-478`).
- CLI compensates with `promote.py:378-419` `_unwrap_recorded_arguments` ("No model can produce the recorded shape", `docs/agents.md:190-206`).

**#8 LiteLLM.**
- Single chokepoint `models/client.py:665` `litellm.acompletion`, with `drop_params: True` (`:483`, `:599`) — unsupported params vanish silently.
- No `tool_choice`, `parallel_tool_calls`, or `strict` anywhere in `evalshift-cli/src` (only unrelated `optional_fields_scored: "strict"`).
- SDK allow-list `capture/generation.py:28-36` `GENERATION_KEYS` records temperature, top_p, response_*, max_tokens — never tool_choice. CLI translator `runner/generation.py:18-20` `_HANDLED_KEYS` consumes four keys, debug-logs the rest.

**#1 Namespace.**
- `evalshift-sdk/pyproject.toml` `packages = ["src/evalshift"]`; `evalshift-cli/pyproject.toml` same. Both have a regular `evalshift/__init__.py`.
- `evalshift-sdk/docs/DECISIONS.md` D-pkg: "Co-installing ... clashes ... tracked as D1-followup (unify later: CLI depends on SDK, or a `[cli]` extra)."
- Two-venv instruction repeated in `README.md:50-53`, `:85`, `docs/getting-started.md:25-27`, `evalshift-sdk/README.md:41-42`.

**#2 Provider wrappers.**
- `evalshift-sdk/src/evalshift/adapters/` contains only `langchain.py`; one optional extra in `pyproject.toml:33-36`.
- Stdlib-only rule: `docs/DECISIONS.md` D-deps; `capture/toolset.py:19-20`.
- `record_model_call` (`capture/api.py:225-235`): `input_tokens`/`output_tokens`/`cost_usd` default 0, `latency_ms` None. No pricing table in SDK; LangChain adapter never sets `cost_usd` (`langchain.py:474-477`).
- Corrections: missing tokens never block promotion; missing `toolset_ref` does (`promote.py:206-215`, `blocked_reason="no_toolset"`). LlamaIndex appears once in the monorepo, in `README.md:331` **Non-goals**. `DECISIONS.md:4` references `IMPLEMENTATION_PLAN.md`, which does not exist.

**#6 Sampling.**
- No repeat option (`grep repeat|n_runs|trials` over config/runner/suite empty). `orchestrator.py:588-600` emits one source + one target item per pair. `config/models.py:603` `cache: bool = True`. `docs/methodology.md:469-472` states it.
- Dedup is by content key, deliberately **not** input_hash: `promote.py:990-1005`.
- Power warnings exist: `analysis/statistics.py:56-57` `MIN_N_FOR_TEST=5`, `MIN_N_RELIABLE=20`; `reports/html.py:82` `_SMALL_SAMPLE_THRESHOLD=10`; `policy.py:808-812` `inconclusive` when Wilson interval spans budget.

**#7 Judge.**
- `config/models.py:28` `DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite-preview"`. `init.py:47-57` scaffolds a same-provider judge on purpose (one API key).
- A/B randomisation: `llm_judge.py:132-136`. Family-bias warning already present: `init.py:140-141`, `docs/configuration.md:502`. No runtime check.
- Stale docstring `config/models.py:186` says "Defaults to a strong Anthropic model".

**#5 Drift vs correctness.**
- `evaluators/semantic.py:146` fails below `min_similarity` (default 0.9, `config/models.py:174`); target compared to source only. No text evaluator reads `SuiteExample.expected` (`suite/models.py:189-191`, "unused by most evaluators").
- Default init profile (`_scaffold.py:94-105`) writes `min_equivalence_rate: 0.75`; `init.py:72-81, 142-150` sets semantic and llm_judge `blocking: false`. The 0.95 floor is only `provider-switch` and the repo's stale root `evalshift.yaml:53-60`.
- Library default `SemanticEvaluatorConfig.blocking = True` (`config/models.py:177`) — hand-written configs that omit the key are blocking.

**#9 Config.**
- `config/models.py:626-640`: `prompts` required (`min_length=1`), `suites` optional dict. No `captures` block. `python_string` is a literal-only AST parser (`parsers/python_string.py:134-198`), never executes.
- `prompts` on hot path: `orchestrator.py:361-366`, `validate.py:69`, `doctor.py:149`. Init's `replay` passthrough prompt is what makes promoted captures replayable.
- **Real mismatch:** `docs/configuration.md:84-89` says init writes `max_cost_increase: 0.50`, `max_latency_increase: 2.0`; init and schema write 0.30/0.30. `llms-full.txt:909-911` shows a third, obsolete block (0.03/0/0.95/0.01/0.03).
- `examples/simple`, `examples/agent` use `python_string`; `examples/agent-traces` uses `manual`; none use `suites`.
- `docs/configuration.md:302` says "Three sub-keys" under evaluators; seven are documented.

---

## Part 2 — Plan

### Ordering rationale

Phases are ordered by (validity × user harm) ÷ cost. Phase 0 is all documentation that is currently *wrong* and cheap to fix. Phases 1–4 restore honesty and correctness in what exists. Phases 5–6 add capability or restructure packaging. Phase 7 holds the half-valid statistical points, which are already partly mitigated.

### Global constraints

- `mypy --strict` and ruff clean in both repos (`make ci`).
- TDD for every logic change: failing test first.
- Conventional Commits; one commit per task unless noted.
- SDK runtime stays stdlib-only (D-deps). Any provider integration is an import-guarded optional extra, like `adapters/langchain.py`.
- SDK capture schema changes bump `SCHEMA_VERSION` (`trace/schema.py:19`, currently `2.0.0`) and register a migration via `trace/migrate.py` `register_migration`. CLI must accept both old and new envelopes.
- Do not make previously-valid `evalshift.yaml` files fail validation.

---

## Phase 0 — Fix documentation that is wrong today

*Validity: full. Importance: high (users follow these). Cost: minutes each.*

### Task 0.1 `[sdk]` Remove the non-existent `--offline --fixtures` workflow

**Files:** `evalshift-sdk/examples/support_agent/README.md`, `evalshift-sdk/examples/support_agent/` (check for `fixtures.jsonl` and any script that references it).

- [x] **Step 1:** Read the example README end to end and list every command it tells the user to run.
- [x] **Step 2:** Run each command against the current CLI in a scratch venv; record which fail (`--offline`, `--fixtures` are known to).
- [x] **Step 3:** Rewrite the walkthrough to use commands that exist: `evalshift capture sync`, `evalshift run` with real keys, or the mocked integration harness at `evalshift-cli/tests/integration/replay_client.py`. Remove `fixtures.jsonl` if nothing consumes it.
- [x] **Step 4:** Add a short note that tool-result fixtures are captured but not yet consumed by replay (links to Phase 2).
- [x] **Step 5:** Commit: `docs(examples): replace non-existent --offline flags with a working walkthrough`.
  Done (sdk). Also found and fixed: example evalshift.yaml used removed keys `tools_path` and `tool_selection[].mode`, and lacked managed suites markers; `tools.yaml` removed with `fixtures.jsonl`; push host corrected to api.evalshift.dev.

### Task 0.2 `[cli]` Correct migration-policy defaults in the config reference

**Files:** `docs/configuration.md:77-89`, `llms-full.txt:905-915`, `DOCS.md` (verify only).

- [x] **Step 1:** Change `max_cost_increase: 0.50` → `0.30` and `max_latency_increase: 2.0` → `0.30` in `docs/configuration.md`, and reword the trailing comments (they explain the 50%/200% values).
- [x] **Step 2:** Replace the "what init writes" block in `llms-full.txt` with the current `INIT_PROFILE_POLICIES["model-upgrade"]` from `src/evalshift/cli/commands/_scaffold.py:94-105`.
- [x] **Step 3:** Add a unit test that renders `INIT_PROFILE_POLICIES["model-upgrade"]` and asserts each `key: value` line appears verbatim in both `docs/configuration.md` and `llms-full.txt` (pattern: `tests/unit/test_init.py:148-161` already pins init ↔ code; extend to docs).
- [x] **Step 4:** Commit: `docs(config): align migration_policy defaults with init and schema`.
  Done in cef4c85. Test also pins DOCS.md.

### Task 0.3 `[cli]` Regenerate the committed root `evalshift.yaml`

**Files:** `evalshift.yaml` (repo root).

- [x] **Step 1:** Confirm it is a stale init output (0.03/0/0.95 policy, no `blocking: false`).
- [x] **Step 2:** Decide: either regenerate with `evalshift init --provider <same>` and preserve any project-specific values, or delete it if it is only a fixture. Check `git log --follow evalshift.yaml` and grep tests/CI for references first.
- [x] **Step 3:** Commit: `chore: drop stale root evalshift.yaml (unused leftover init output)` — deleted rather than regenerated: no consumers, no project-specific values (a2c8f24).

### Task 0.4 `[cli]` Fix stale docstrings and doc counts

**Files:** `src/evalshift/config/models.py:186`, `docs/configuration.md:302`, `docs/sdk.md:123-124`.

- [x] **Step 1:** `config/models.py:186`: replace "Defaults to a strong Anthropic model" with the actual `DEFAULT_JUDGE_MODEL`, or reference the constant.
- [x] **Step 2:** `docs/configuration.md:302`: "Three sub-keys" → count the documented sub-keys and state that number, or drop the count.
- [x] **Step 3:** `docs/sdk.md:123-124`: LangChain is listed as "outside the SDK entirely" — correct to mention `EvalShiftCallbackHandler` (`evalshift-sdk[langchain]`).
- [x] **Step 4:** Commit: `docs: fix stale judge default, evaluator count, and LangChain adapter mention`.
  Done in 1861439.

### Task 0.5 `[sdk]` Fix dangling references in DECISIONS.md

**Files:** `evalshift-sdk/docs/DECISIONS.md:4`, `:31-34`.

- [x] **Step 1:** Line 4 references `IMPLEMENTATION_PLAN.md`, which does not exist. Remove the sentence or point at the real phase list.
- [x] **Step 2:** Lines 31-34 assert the CLI consumes the fixture table with a "halt-and-flag" default. Annotate: "Fixture table is written; CLI consumption is pending (see cli plan 2026-09-08-external-review-response.md Phase 2)."
- [x] **Step 3:** Commit: `docs(decisions): remove dangling plan reference, mark fixture consumption as pending`.
  Done (sdk, 647e09f). Left alone: DECISIONS.md:51 still says SCHEMA_VERSION 1.0.0 (actual 2.0.0) — separate stale note.

### Task 0.6 `[cli]` Add a `suites`-based example

**Files:** new `examples/captured-suite/` (name to taste), `README.md` examples list.

- [x] **Step 1:** Create an example whose `evalshift.yaml` is what `evalshift init` writes (passthrough `replay` prompt + managed `suites:` block) plus a small pre-promoted `golden.jsonl` under `.evalshift/suites/` (or wherever `capture sync` writes; confirm from `captures/promote.py`).
- [x] **Step 2:** Include one or two capture envelope files so `evalshift capture sync` can be demonstrated from scratch.
- [x] **Step 3:** Ensure `evalshift validate` and `evalshift doctor` pass on it; add to whatever test iterates examples (grep `examples/` in `tests/`).
- [x] **Step 4:** Commit: `docs(examples): add capture-first example using the suites block`.
  Done. Named `examples/capture-first/`. Envelopes generated with the real SDK; suite produced by the real `capture sync`; a test re-derives golden.jsonl from the committed captures.

### Task 0.7 `[cli]` Follow-up found during 0.6: `validate` has no `--suite-name`

**Files:** `src/evalshift/cli/commands/validate.py`, `cli/commands/_suites.py` (`resolve_suite_path`), tests, `docs/`.

`run`, `all`, `bundle` and `push` resolve suites through `_suites.resolve_suite_path`; `validate` hardcodes `--suite` defaulting to `./golden.jsonl`, so bare `evalshift validate` fails in every capture-first project.

- [ ] **Step 1:** Failing test: `evalshift validate --suite-name oncall_triage` in `examples/capture-first` exits 0; bare `evalshift validate` in a project whose only suite is in the managed block also resolves it (single-suite default) or lists the names.
- [ ] **Step 2:** Implement via `resolve_suite_path`; keep `--suite <path>` working.
- [ ] **Step 3:** Update `examples/capture-first/README.md` "Check it" section and `docs/getting-started.md`.
- [ ] **Step 4:** Commit: `feat(validate): accept --suite-name and resolve suites from the config`.

---

## Phase 1 — Honest framing in docs (no code)

*Validity: full for #4 marketing; partial for #5, #9. Importance: medium. Cost: low.*

### Task 1.1 `[cli]` Reword the "sequences" claim and state the single-round limit up front

**Files:** `README.md:253-256`, `docs/agents.md:3-5`, `docs/agents.md:167-170`, `llms-full.txt` (agent section).

- [ ] **Step 1:** Replace "how it sequences them" with what is actually tested: which tools, what arguments, order and parallelism *within the first tool-emitting round*.
- [ ] **Step 2:** Move the one-call-per-example statement from `docs/agents.md:167-170` to the top of the agents page, and add a one-liner in the README agent section.
- [ ] **Step 3:** Commit: `docs(agents): state first-round-only replay scope where the claim is made`.

### Task 1.2 `[cli]` Document drift-vs-correctness and the `prompts`/`suites` relationship

**Files:** `docs/evaluators.md` (semantic section, near `:57-61`), `docs/configuration.md` (top of `prompts` and `suites` sections), `docs/faq.md`.

- [ ] **Step 1:** Add a short "What 'expected' means" paragraph to the semantic evaluator docs: the yardstick is the source model, so a correct-but-reworded target reads as drift; that is why init ships it advisory; use the judge criterion for correctness.
- [ ] **Step 2:** Add two sentences to `docs/configuration.md` explaining that `prompts` is the template axis and `suites` the dataset axis, that both are always present, and that the init `replay` prompt is the passthrough that makes captured inputs replayable.
- [ ] **Step 3:** FAQ entry: "Why does a hand-written config block on semantic when init does not?" (library default `blocking: true`, `config/models.py:177`).
- [ ] **Step 4:** Commit: `docs: explain drift vs correctness and prompts vs suites`.

---

## Phase 2 — Make replay consume what capture records (#4)

*Validity: full. Importance: high. Cost: medium-high. Depends on Phase 0.1 wording.*

Goal: teacher-forced multi-round replay. For round *k* > 1, the candidate is given the recorded history through round *k-1* (including recorded tool results as fixtures) and asked for round *k*. Every round is scored against `expected_tool_rounds[k]`.

### Task 2.1 `[cli]` Spec first

**Files:** new `docs/superpowers/specs/2026-09-XX-teacher-forced-replay-design.md`.

- [ ] **Step 1:** Write the design: work-item shape (one `WorkItem` per round, or one per example with an inner loop), how recorded tool results are injected (provider-native `tool_result` messages built from `ToolResultEvent.result`, keyed by `call_id`), what happens when the candidate calls a tool with no fixture (halt-and-flag per SDK D-decision, or substitute a synthetic "unavailable" result), cache-key changes (round index), cost estimate changes (`utils/cost.py` multiplies by rounds), and policy/report changes (per-round divergence).
- [ ] **Step 2:** Decide the default: `rounds: first` stays default for cost; `rounds: all` opts into teacher forcing. Confirm `promote.py:91` already has the enum.
- [ ] **Step 3:** Review with maintainer; then continue.

### Task 2.2 `[cli]` Fixture loading

**Files:** `src/evalshift/captures/promote.py` (`_tool_rounds` `:457-483`, history recovery `:554-581`), `src/evalshift/suite/models.py` (`SuiteExample`), `tests/unit/test_promote.py`.

- [ ] **Step 1:** Failing test: promoting a two-round capture with `rounds: all` yields `expected_tool_rounds` of length 2 and a new `tool_result_fixtures: dict[call_id, result]` (or per-round list) on the example.
- [ ] **Step 2:** Implement; keep v0.1–v0.3 suites loading unchanged.
- [ ] **Step 3:** Commit: `feat(promote): carry recorded tool results as replay fixtures`.

### Task 2.3 `[cli]` Multi-round runner loop

**Files:** `src/evalshift/runner/orchestrator.py` (`_build_work_list` `:588-600`, dispatch `:1021-1038`), `src/evalshift/runner/models.py`, `src/evalshift/models/client.py`, `tests/unit/test_orchestrator.py`, `tests/integration/`.

- [ ] **Step 1:** Failing integration test using the mocked client: candidate emits tool call in round 1, receives fixture result, emits round-2 call; both rounds recorded on the result.
- [ ] **Step 2:** Implement loop with a hard cap equal to `len(expected_tool_rounds)`; unmatched fixture → record `fixture_missing` and stop the loop for that example.
- [ ] **Step 3:** Extend cache key with round index; extend cost pre-flight with round count.
- [ ] **Step 4:** Commit: `feat(runner): teacher-forced multi-round replay`.

### Task 2.4 `[cli]` Score and report per round

**Files:** `src/evalshift/evaluators/tool_selection.py`, `tool_trace_structure.py`, `analysis/policy.py`, `reports/html.py` + template, `docs/agents.md`, `docs/evaluators.md`.

- [ ] **Step 1:** Failing tests: tool_selection compares round *k* output to `expected_tool_rounds[k]`; report shows a per-round divergence row.
- [ ] **Step 2:** Implement; policy budget `max_tool_divergence` counts an example as diverged if any replayed round diverges (document this).
- [ ] **Step 3:** Update docs and revert the Phase 1.1 caveat to describe the new capability.
- [ ] **Step 4:** Commit: `feat(evaluators): per-round tool scoring for multi-round replay`.

---

## Phase 3 — Record the model's requested tool calls (#3)

*Validity: full. Importance: high. Cost: medium. Cross-repo; SDK first.*

### Task 3.0 `[cli]` CLI trace model accepts `requested_tool_calls` (must land before 3.1)

**Files:** `src/evalshift_cli/traces/models.py` (`ModelCallEvent`), `captures/reader.py` (version check), `tests/unit/test_trace_models.py`, `tests/unit/test_captures_reader.py`.

Found while preparing Phase 3: the CLI's trace models inherit `extra="forbid"`, and the SDK dataclasses mirror them field for field (guarded by `evalshift-sdk/tests/conformance/test_parity.py` against the vendored copy in `cli_models_vendored.py`). If the SDK starts writing a new field before the CLI accepts it, every existing CLI install rejects every new capture at load time. So the plan's "SDK first" is inverted for the schema field: the CLI model lands first and is the contract.

Contract (both repos, verbatim): `requested_tool_calls: list[RequestedToolCall] | None = None`, `RequestedToolCall = {name: str, arguments: dict[str, Any], call_id: str | None}`, positioned immediately after `tools_offered` so the parity test's field order holds.

- [x] **Step 1:** Failing tests: `ModelCallEvent` accepts and round-trips the field; absent → `None`; a `2.1.0` envelope passes `captures/reader.py` (it checks the major only, `_SUPPORTED_MAJOR = 2`; lock that in with a test).
- [x] **Step 2:** Add `RequestedToolCall` (strict) and the field.
- [x] **Step 3:** Commit: `feat(traces): accept requested_tool_calls on ModelCallEvent`.
  Done in 6b8f210 (cli). reader.py needed no change: major-only gate, now pinned by tests.

### Task 3.1 `[sdk]` Schema: add `requested_tool_calls` to `ModelCallEvent`

**Files:** `src/evalshift/trace/models.py:24-45`, `trace/schema.py`, `trace/serialize.py`, `trace/migrate.py`, `docs/SCHEMA.md`, `tests/`.

- [x] **Step 1:** Failing tests: a `ModelCallEvent` round-trips a `requested_tool_calls: list[{name, arguments, call_id}] | None` field; a `2.0.0` envelope loads with the field absent → `None`.
- [x] **Step 2:** Add the field (default `None` so old writers/readers coexist). Bump `SCHEMA_VERSION` to `2.1.0`; register a no-op-with-default migration. Update `schema.py:80` fixed field set.
- [x] **Step 3:** Document in `docs/SCHEMA.md` and `docs/DECISIONS.md` (new D-requested: "requested ≠ executed; both recorded").
- [x] **Step 4:** Commit: `feat(trace): record model-requested tool calls separately from executed ones`.
  Done in aad7a21 (sdk). Also updated docs/REDACTION.md field table and tests/test_smoke.py's pinned version.

### Task 3.2 `[sdk]` API: accept requested calls in `record_model_call` and the streaming recorder

**Files:** `src/evalshift/capture/api.py:225-250` (`record_model_call`), `:529-546` (`set_usage` area of `capture.model_call`), `DOCS.md:278-340`, `llms-full.txt`.

- [x] **Step 1:** Failing tests for `record_model_call(..., requested_tool_calls=[...])` and `rec.set_requested_tool_calls([...])`.
- [x] **Step 2:** Implement; redact arguments through the same redactor as `ToolCallEvent.arguments`.
- [x] **Step 3:** Add small stdlib-only helpers that extract requested calls from an already-serialised provider response dict (OpenAI `choices[0].message.tool_calls`, Anthropic `content[].type == "tool_use"`, Gemini `candidates[0].content.parts[].functionCall`). Pure dict walking, no provider import.
- [x] **Step 4:** Docs: explain the difference between "offered" (`tools=`), "requested" (new), and "executed" (`@capture.tool`).
- [x] **Step 5:** Commit: `feat(capture): requested_tool_calls on record_model_call and model_call recorder`.
  Done in ba8bffa + ff7a272 (sdk; helpers merged in d69d397, vendored `name` min_length synced in 19964ee). Helper: `evalshift.capture.requested.extract_requested_tool_calls`; returns `[]` for a recognised response with no calls and `None` for an unrecognised one.

### Task 3.3 `[sdk]` LangChain adapter: read `AIMessage.tool_calls`

**Files:** `src/evalshift/adapters/langchain.py:465-478` (`on_llm_end`), `tests/adapters/`.

- [x] **Step 1:** Failing test with a synthetic `LLMResult` whose generation message carries `tool_calls`.
- [x] **Step 2:** Populate `requested_tool_calls` from `generations[0][0].message.tool_calls` when present.
- [x] **Step 3:** Commit: `feat(langchain): capture requested tool calls from AIMessage`.
  Done in 836ea92 (sdk). Reuses api's normaliser; chat message with no tool_calls records `[]`, plain text generation leaves the field unset.

### Task 3.4 `[cli]` Promotion prefers requested calls as ground truth

**Files:** `src/evalshift/captures/models.py`, `captures/promote.py` (`_tool_rounds` `:457-483`, `_unwrap_recorded_arguments` `:378-419`), `docs/agents.md:190-206`, tests.

- [x] **Step 1:** Failing tests: when `requested_tool_calls` is present it becomes `expected_tools` / `expected_tool_rounds` verbatim and `_unwrap_recorded_arguments` is skipped; when absent, current executed-call behaviour is unchanged.
- [x] **Step 2:** Implement; add a `promotion_source: "requested" | "executed"` note to the promoted case metadata so reports can show which yardstick was used.
- [x] **Step 3:** Update `docs/agents.md` — the "No model can produce the recorded shape" section now applies only to legacy captures.
- [x] **Step 4:** Commit: `feat(promote): use model-requested tool calls as ground truth when captured`.

---
  Done in 76a8c43 (cli). `promotion_source` on PromotedCase and BuiltExample; mixed captures fall back to executed for the whole capture with a warning; requested-vs-executed disagreement warns, requested wins. Checked-in capture-first cases gained `promotion_source: executed`.

## Phase 4 — Stop LiteLLM from silently dropping migration-relevant params (#8)

*Validity: full. Importance: high. Cost: medium. Cross-repo.*

### Task 4.1 `[sdk]` Record `tool_choice`, `parallel_tool_calls`, and strictness

**Files:** `src/evalshift/capture/generation.py:28-36` (`GENERATION_KEYS`), `adapters/langchain.py:101` (`_generation_config`), `capture/toolset.py` (strict flag on OpenAI function shape survives normalisation?), tests, `DOCS.md`.

- [ ] **Step 1:** Failing tests: `sanitize_generation_config({"tool_choice": ..., "parallel_tool_calls": False})` keeps both; a toolset with `function.strict: true` fingerprints differently from one without.
- [ ] **Step 2:** Extend `GENERATION_KEYS`; verify `normalize_tools` does not prune `strict` (`DECISIONS.md:309-310` says it never prunes keys — add a test that locks that in).
- [ ] **Step 3:** Commit: `feat(capture): record tool_choice, parallel_tool_calls, and strict schemas`.

### Task 4.2 `[cli]` Carry those params through replay

**Files:** `src/evalshift/runner/generation.py:18-20` (`_HANDLED_KEYS`), `models/client.py:588-592` (tool serialisation), `evaluators/tool_models.py:60-77`, `tool_parser.py:39-64`, tests.

- [ ] **Step 1:** Failing tests: a captured `tool_choice` reaches the `litellm.acompletion` kwargs; `strict: true` survives `to_openai`; Anthropic equivalent (`tool_choice: {type: "tool", name}`) is produced for Anthropic targets.
- [ ] **Step 2:** Implement translation per provider prefix. Where a target provider cannot express the constraint, record it (next task) rather than drop it.
- [ ] **Step 3:** Commit: `feat(runner): pass tool_choice, parallel_tool_calls, and strict through replay`.

### Task 4.3 `[cli]` Surface dropped parameters instead of hiding them

**Files:** `src/evalshift/models/client.py:483, :599` (`drop_params`), `models/capabilities.py`, `runner/generation.py:58-60`, `reports/html.py` + template (extend the `non_deterministic_models` banner pattern at `report.html.j2:49-68`), `analysis/policy.py`, docs.

- [ ] **Step 1:** Failing tests: when `litellm.get_supported_openai_params` says the target lacks `tool_choice` (or `response_format`), the run records `dropped_params[model] = {...}` and the report shows a "Constraints not honoured by target" banner.
- [ ] **Step 2:** Implement using the existing capability probe (`capabilities.py:56-64`) before dispatch; keep `drop_params: True` so calls still succeed, but log at `warning` not `debug`.
- [ ] **Step 3:** Optional policy knob `fail_on_dropped_params: bool = False` (document in `docs/configuration.md`).
- [ ] **Step 4:** Commit: `feat(report): surface generation params the target model cannot honour`.

---

## Phase 5 — Thin provider client wrappers (#2)

*Validity: substantive. Importance: adoption. Cost: medium per provider. Depends on Phase 3.2 helpers.*

Constraint: stdlib-only runtime (D-deps). Each wrapper is an import-guarded optional extra like `adapters/langchain.py`, and wraps a *client instance* rather than monkeypatching the module.

### Task 5.1 `[sdk]` Design note

**Files:** `docs/DECISIONS.md` (new D-wrappers).

- [ ] **Step 1:** Record: wrappers are proxies over the user's client object (`wrap_openai(client)`, `wrap_anthropic(client)`, `wrap_genai(client)`); they populate `model_id`, `tools`, `input`, `output`, `input_tokens`, `output_tokens`, `latency_ms`, `generation_config` (incl. Phase 4 keys), and `requested_tool_calls` (Phase 3); `cost_usd` stays 0 in the SDK — pricing belongs to the CLI (`utils/cost.py`, litellm's price table) and is applied at promote/report time. Streaming: wrap the iterator; usage taken from the final chunk.
- [ ] **Step 2:** Decide extra names: `evalshift-sdk[openai]`, `[anthropic]`, `[google-genai]`.

### Task 5.2 `[sdk]` OpenAI wrapper

**Files:** new `src/evalshift/adapters/openai.py`, `pyproject.toml` extras, `tests/adapters/test_openai.py` (synthetic response objects; one `importorskip`-guarded real-client smoke test like langchain), `README.md`, `DOCS.md`.

- [ ] **Step 1:** Failing tests: sync `chat.completions.create`, async, and streaming each produce one `model_call` with usage, latency, offered tools, requested tool calls.
- [ ] **Step 2:** Implement as a proxy that forwards everything and intercepts only `chat.completions.create` / `responses.create`. Fail-open: any wrapper error logs and returns the real response.
- [ ] **Step 3:** Commit: `feat(adapters): OpenAI client wrapper`.

### Task 5.3 `[sdk]` Anthropic wrapper

- [ ] Same shape as 5.2 for `messages.create` and `messages.stream`. Tool calls from `content[].type == "tool_use"`. Commit: `feat(adapters): Anthropic client wrapper`.

### Task 5.4 `[sdk]` Google GenAI wrapper

- [ ] Same shape for `models.generate_content` (sync/async/stream). Reuse the existing duck-typed Gemini toolset handling in `capture/toolset.py`. Commit: `feat(adapters): google-genai client wrapper`.

### Task 5.5 `[cli]` Fill cost at promotion when the capture has tokens but no cost

**Files:** `captures/promote.py`, `utils/cost.py`, tests, `docs/agents.md`.

- [ ] **Step 1:** Failing test: capture with `input_tokens>0`, `cost_usd==0` is promoted with a cost estimate derived from litellm's price table for `model_id`, tagged `cost_source: "estimated"`.
- [ ] **Step 2:** Implement; leave recorded non-zero costs untouched.
- [ ] **Step 3:** Commit: `feat(promote): estimate cost from tokens when the SDK recorded none`.

---

## Phase 6 — Resolve the `evalshift` namespace collision (#1)

*Validity: full. Importance: high friction. Cost: breaking change, needs a major/minor bump and a deprecation window. Do after Phases 2–4 so the SDK schema is stable before the packaging move.*

### Task 6.1 Decide the shape

**Files:** `evalshift-sdk/docs/DECISIONS.md` (D1-followup → resolved), new spec in `evalshift-cli/docs/superpowers/specs/`.

Options already named by the author:
- **A. CLI depends on SDK.** `evalshift` (CLI) declares `evalshift-sdk` as a dependency; the SDK owns the top-level `evalshift` package; CLI code moves under `evalshift.cli`/`evalshift._cli` or a separate top-level `evalshift_cli` package with the `evalshift` console script. Pro: `import evalshift` unchanged for SDK users. Con: CLI must never break SDK's stdlib-only promise; two repos share one import root.
- **B. Rename the SDK import** to `evalshift_sdk` and keep a deprecated `evalshift` shim for one minor version. Pro: clean separation, no cross-repo package root. Con: every existing `from evalshift import capture` breaks after the shim window.
- **C. Single distribution, `[cli]` extra.** Merge repos; `pip install evalshift` is the SDK, `pip install "evalshift[cli]"` adds the CLI. Pro: simplest for users. Con: repo merge, AGPL (CLI) vs MIT (SDK) licensing has to be reconciled per subpackage.

- [x] **Step 1:** Picked **A** in its second form: CLI import package `evalshift_cli`, distribution name and console script unchanged, `evalshift-sdk>=0.3.0` declared as a runtime dependency. The first form (CLI code under the SDK's `evalshift` root) needs a PEP 420 namespace package, which the SDK's re-exporting `__init__.py` rules out; editable co-installs would need it too.
- [x] **Step 2:** Spec: `docs/superpowers/specs/2026-09-09-namespace-collision-design.md` (3dbf245). No shim is possible — any `evalshift/` file shipped by the CLI recreates the clash — so the window is the CHANGELOG entry plus a minor bump. Found seven two-venv passages in the CLI and three in the SDK (Findings #1 listed four).

### Task 6.2 Implement per the chosen spec

- [x] `[cli]` Package move `src/evalshift` → `src/evalshift_cli`; imports rewritten in `src/`, `tests/`, `scripts/`; tooling paths (mypy, ruff first-party, coverage, Makefile, CI, pre-commit); `evalshift-sdk` dependency; console script → `evalshift_cli.cli.main:app`; deferred-warnings printer matches its own records under the new root (1e1de65).
- [x] `[cli]` `doctor` row `evalshift-sdk`: ok with the SDK version; warn when missing, when the import fails, or when an older CLI's files or a local `evalshift/` directory shadow it; never fails the command (5da36c1).
- [x] `[cli]` Docs: README, getting-started, sdk.md, DOCS.md, AGENTS.md, llms-full.txt, faq.md, examples/capture-first; CHANGELOG Breaking + Added entries (d9c4eba, 1e1de65, 5da36c1).
- [x] `[sdk]` DECISIONS.md D-pkg and D1-followup marked resolved; README, DOCS.md, support_agent example; vendored-mirror header path; CHANGELOG. No code change (ea54c0a).
- [ ] `[cli]` Version bump to 0.14.0 — left for the maintainer's `chore(release): 0.14.0` commit, since CONTRIBUTING makes that bump the release itself. The CHANGELOG entry already names 0.14.0.

  Verified: ruff, format, `mypy --strict`, and 1893 tests green with `evalshift-sdk` 0.3.0 installed from PyPI; the built wheel ships 103 `evalshift_cli/` files, no `evalshift/` entry, and `Requires-Dist: evalshift-sdk>=0.3.0`; `evalshift doctor` shows the row; the capture-first agent and `capture sync` run from the single CLI venv; SDK tests 510 green.

---

## Phase 7 — Statistical rigour and judge hygiene (#6, #7, #5 residue)

*Validity: partial. Importance: medium. Cost: low–medium.*

### Task 7.1 `[cli]` Optional repeated sampling per example

**Files:** `config/models.py` (`defaults`), `runner/orchestrator.py:588-600` (`_build_work_list`), cache key `:911-920`, `analysis/statistics.py`, `reports/html.py` + template, `docs/methodology.md:469-472`, `docs/configuration.md`, tests.

- [ ] **Step 1:** Failing tests: `defaults.samples_per_example: 3` emits three source and three target items per pair, cache key includes the sample index, and the per-pair score is the mean with within-pair variance recorded.
- [ ] **Step 2:** Implement; default stays 1. Cost pre-flight multiplies by the sample count.
- [ ] **Step 3:** Report: show "samples per example: N" next to the pair count; when N == 1 and any model is non-deterministic, reuse the existing banner text.
- [ ] **Step 4:** Update `docs/methodology.md` (replace the "run multiple seeds upstream" advice).
- [ ] **Step 5:** Commit: `feat(runner): samples_per_example for repeated sampling`.

### Task 7.2 `[cli]` Runtime warning when the judge shares a family with source or target

**Files:** `cli/commands/doctor.py`, `cli/commands/validate.py`, `evaluators/llm_judge.py`, `reports/html.py` + template, tests.

- [ ] **Step 1:** Failing tests: `doctor`/`validate` emit a warning when `judge_model` resolves to the same provider prefix as `source_model` or `target_model`; the HTML report shows a one-line "judge shares a model family with the target" note.
- [ ] **Step 2:** Implement using `models/registry.py` provider resolution. Warning only; never fail.
- [ ] **Step 3:** Commit: `feat(doctor): warn when the judge shares a model family with a compared model`.

### Task 7.3 `[cli]` Consider flipping the library default for semantic `blocking`

**Files:** `config/models.py:177`, `docs/configuration.md:304-318`, CHANGELOG.

- [ ] **Step 1:** Decide whether `SemanticEvaluatorConfig.blocking` should default to `False` to match init. This is a behaviour change for hand-written configs; if done, note it under a versioned `version: 1` semantics change in CHANGELOG, or defer to a `version: 2` schema.
- [ ] **Step 2:** If flipped: update defaults table, the FAQ entry from Task 1.2, and tests in `tests/unit/test_config_models.py`.

---

## Not planned (review points judged invalid)

- Treating `prompts` as a leftover or removing `python_string` detection (#9). Both are required by the recommended flow. Only the docs residue (Tasks 0.2, 0.6, 1.2) is actioned.
- Adding a LlamaIndex adapter (#2). It is listed as a non-goal in the CLI README and appears on no roadmap. Provider wrappers (Phase 5) are the higher-leverage substitute the reviewer suggested.
- Changing the default `min_equivalence_rate` (#5). Init already writes 0.75 and ships semantic/judge advisory.

## Progress log

| Date | Task | Note |
|---|---|---|
| 2026-09-08 | — | Plan written from verified findings. |
| 2026-09-08 | 0.1–0.6 | Phase 0 complete. cli: a2c8f24, cef4c85, 1861439, 54ff55f. sdk: 647e09f, 7853167. Found and logged Task 0.7 (validate lacks --suite-name). |
| 2026-09-09 | 6.1–6.2 | Phase 6 done out of order (before 2–4, at the maintainer's request). cli: 3dbf245, 1e1de65, 5da36c1, d9c4eba. sdk: ea54c0a. Version bump deferred to the 0.14.0 release commit. |
| 2026-09-09 | 3.0–3.4 | Phase 3 complete, run as three parallel Opus agents (cli; sdk core; sdk helpers in a worktree) then one for LangChain. cli: 6b8f210, 76a8c43, 3222b6a. sdk: aad7a21, ba8bffa, ff7a272, d69d397, 19964ee, 836ea92. Task 3.0 added: the CLI's `extra="forbid"` trace model must accept the field before the SDK writes it. End-to-end verified with the dev SDK: 2.1.0 envelopes with requested calls promote with `promotion_source: requested`. Open: whole-capture fallback when only some model calls carry the field (per-round hybrid considered, not done); capture-first example still records executed-only until SDK 0.4.0 ships the kwarg. |
