# AIMigrate v0.2 — Tool-Call Evaluation — TODO

> Living checklist for the v0.2 release.
> Tick `[ ]` → `[x]` the moment a piece is done. One checkbox = one focused commit (or a tight series).
> Source spec: `aimigrate-v0.2-tool-eval-prd.md.pdf`.
> Builds on: v0.1 MVP (shipped on `main`, 371 tests, full pipeline live).

## North star

A v0.2 release that detects **agent migrations**, not just text migrations:

> A team migrates their customer-routing agent from Claude 4.5 to Claude 5.
> Claude 5 silently stops calling `notify_security_team` on sensitive
> requests. Text-only eval reports green. v0.2 marks it CRITICAL and blocks
> the migration.

Three new evaluators ship on top of the v0.1 pipeline: `tool_selection`,
`tool_arguments`, `tool_trace_structure`. A provider-agnostic `ToolTrace`
data model normalises Anthropic / OpenAI / Gemini responses behind the
same interface.

## Working agreement (apply to every task)

- [x] One checkbox = one focused commit (or a tight series). Mark `[x]` the moment it ships.
- [x] Tests land in the same commit as the code they cover. New module ⇒ new test file.
- [x] CI stays **fully mocked** (no API keys, no cost). `scripts/smoke_live_tools.py` is a manual fixture-recording tool, not a CI test.
- [x] `ruff check`, `ruff format --check`, `mypy --strict src/aimigrate`, `pytest` all green before marking a checkbox done.
- [x] Public functions / classes get Google-style docstrings.
- [x] Update `CHANGELOG.md` under `## [Unreleased]` for any user-visible change.
- [x] **v0.1 backward compatibility is non-negotiable**: every existing v0.1 example must keep working without changes.

User-confirmed scope (this session): **strict PRD scope** — multi-turn,
real tool execution, LangChain auto-detect, smart diffs, streaming are
hard OUT (PRD §3.3 / §13). **Code / tests / docs / examples only** —
no PyPI publish, no customer-discovery outreach, no launch posts in
this plan.

---

## Phase 0 — Branch + scaffolding

Goal: a clean v0.2 work branch with the living checklist in place.

- [x] Create branch `feature/v0.2-tool-eval` off `main`.
- [x] Add `V02_TODO.md` (this file).
- [x] No new prod deps required — LiteLLM, pydantic, scipy, jinja2 already cover everything.
- [x] Verify v0.1 baseline on the new branch: `pytest`, `ruff check`, `ruff format --check`, `mypy --strict src/aimigrate` all green.

---

## Phase 1 — Tool data models + provider parsers + client extension (PRD Week 1)

Goal: `aimigrate test-call --model X --tools tools.yaml --prompt "..."` returns a normalised `ToolTrace` for any of the three providers.

### 1.1 Tool data models (`src/aimigrate/evaluators/tool_models.py`, new)
- [x] `ToolSpec` (name, description, input_schema), `ToolCall` (tool_name, arguments, call_id, parent_call_id, sequence_index), `ToolTrace` (calls, final_text, raised_refusal, refusal_text + computed `call_count` / `tool_names` / `tool_name_set` / `has_parallel_calls()` / `calls_by_tool()`).
- [x] All models inherit from the `_StrictModel` pattern already used in `config/models.py:27` and `runner/models.py:32` (`extra='forbid'`, `validate_assignment=True`).
- [x] `ToolSpec.to_anthropic()` / `.to_openai()` adapters and `ToolSpec.from_dict()` accepting either shape.
- [x] **Tests** (`tests/unit/test_tool_models.py`): `extra='forbid'` round-trips, computed-field correctness, `has_parallel_calls()` truth table, adapter round-trip Anthropic→`from_dict`→`to_anthropic`.

### 1.2 Provider response parser (`src/aimigrate/evaluators/tool_parser.py`, new)
- [x] `ToolParseError(provider, reason, raw)` exception (mirrors existing `ConfigError` shape).
- [x] `detect_provider(model_id) -> str` mapping LiteLLM canonical ids to `anthropic` / `openai` / `gemini`. Reuse the prefix logic from `aimigrate.models.registry._infer_provider_and_canonical`.
- [x] `parse_response_to_trace(response, provider, model_id) -> ToolTrace` dispatcher.
- [x] `_parse_anthropic` (handles native + LiteLLM-normalised-to-OpenAI shape); `_parse_openai` + shared `_parse_openai_shape` (json-decodes string arguments defensively, marks `_parse_error` on malformed); `_parse_gemini` (delegates to `_parse_openai` since LiteLLM normalises).
- [x] **Fixtures** (`tests/unit/fixtures/tool_responses/{anthropic,openai,gemini}/*.json`): start with synthetic hand-crafted JSON. Real responses populate later via `scripts/smoke_live_tools.py`.
- [x] **Tests** (`tests/unit/test_tool_parser.py`, ≥30 cases): per-provider single-call, parallel-calls, text-only, mixed text+tool, refusal, malformed-arguments-JSON marks parse_error, unexpected shape raises clear `ToolParseError`, `detect_provider` covers every branch including unknown.

### 1.3 Tool-aware model client (`src/aimigrate/models/client.py`, extend)
- [x] `ToolCompletionResult` pydantic model: `trace: ToolTrace`, plus the same `model_id` / `input_tokens` / `output_tokens` / `cost_usd` / `latency_ms` fields as `CompletionResult`, plus `raw_provider_response: dict[str, Any]` (debugging only).
- [x] New async `complete_with_tools(model, prompt, tools: list[ToolSpec], *, temperature, max_tokens, extra)` — does NOT replace existing `complete()`. Reuses existing retry policy, cost calc, error-mapping helpers.
- [x] Tools serialised per-provider: `to_anthropic()` for Anthropic models, `to_openai()` for everything else (Gemini accepts the OpenAI shape via LiteLLM's normaliser).
- [x] **Tests** (`tests/unit/test_model_client.py`, extend): mocked `litellm.acompletion` returning the Phase 1.2 fixtures; `ToolCompletionResult` has the right `trace`; existing retry/error-mapping tests parameterised to also cover the tools path.

### 1.4 `aimigrate test-call --tools` extension (`src/aimigrate/cli/commands/test_call.py`, extend)
- [x] Add `--tools <path>` Typer option. When set, load `ToolSpec` list from yaml/json and call `complete_with_tools` instead of `complete`.
- [x] Render the `ToolTrace` as a small Rich panel: tool names list, parallel/sequential indicator, final-text snippet, refusal flag.
- [x] **Tests**: `CliRunner` invocation with a tools fixture confirms `complete_with_tools` is the dispatch target and the panel renders the tool names.

### 1.5 Live fixture-recording script (`scripts/smoke_live_tools.py`, new — manual only)
- [x] Records real responses for `(model, prompt_name)` combos to `tests/unit/fixtures/tool_responses/`. Run by hand with API keys; **not** part of CI.
- [x] Diffs against existing fixtures and prints "drifted" for any that changed shape — early-warning for LiteLLM upstream changes.

### Phase 1 verification
- [x] `aimigrate test-call --model gemini-2.5-flash --tools examples/agent/tools.yaml --prompt "find ACME's recent orders"` returns a Rich panel with the tool names list, no traceback.
- [x] All Phase 1 tests pass; coverage stays ≥ 94%.

---

## Phase 2 — Suite/config schema + `ToolSelectionEvaluator` + pipeline integration (PRD Week 2)

Goal: end-to-end pipeline runs an agent prompt through both models, scores with `tool_selection`, and renders existing report sections.

### 2.1 Suite extension (`src/aimigrate/suite/models.py`, extend)
- [ ] `ExpectedToolCall` model: `tool_name`, optional `arguments`, `match_strategy: Literal["exact", "subset", "contains_per_field"] = "subset"`. `extra='forbid'`.
- [ ] Extend `SuiteExample` with optional fields: `expected_tools: list[ExpectedToolCall] | None`, `expected_tool_count: int | None`, `expected_no_tools: bool = False`, `expected_parallel: bool | None`.
- [ ] **Backward compatibility test**: existing v0.1 suite JSONL (no tool fields) still loads via `aimigrate.suite.loader.load_jsonl`.
- [ ] **Tests**: round-trip new fields; mutual-exclusion check (`expected_no_tools=True` + non-empty `expected_tools` → validation error); load v0.1 suite still works.

### 2.2 Config schema extension (`src/aimigrate/config/models.py`, extend)
- [ ] Add `tools_path: str | None = None` to `PromptDefinition`. When set, the prompt is treated agent-style.
- [ ] Add `ToolSelectionEvaluatorConfig` (mode: exact/set/first/expected, applies_to, severity_floor), `ToolArgumentsEvaluatorConfig` (strategies dict, numeric_tolerance, use_llm_judge_fallback), `ToolTraceStructureEvaluatorConfig` (check_call_count/parallelism/refusals/expected_count, call_count_tolerance).
- [ ] Extend `EvaluatorsConfig` with optional `tool_selection`, `tool_arguments`, `tool_trace_structure` lists/objects.
- [ ] **Tests**: existing v0.1 configs validate unchanged; new fields validate; `tools_path` present without any tool evaluator configured doesn't fail validation (handled via Phase 3 doctor warning instead).

### 2.3 Tools loader (`src/aimigrate/evaluators/tool_loader.py`, new)
- [ ] `load_tools(path: Path) -> list[ToolSpec]` — accept yaml or json; resolve relative paths against the config dir.
- [ ] Friendly errors: missing file, bad shape, empty list (mirrors `ConfigError` / `SuiteError` rendering — plain + rich).
- [ ] **Tests**: yaml + json happy paths, missing file, malformed.

### 2.4 `Call` model + persistence extension (`src/aimigrate/runner/models.py`, extend)
- [ ] Add `trace: ToolTrace | None = None` to `Call`.
- [ ] Wire serialisation in `aimigrate.runner.checkpoint.append_call` and `iter_calls` — `ToolTrace` round-trips losslessly via pydantic so this is mostly free.
- [ ] **Backward-compat test**: existing `raw.jsonl` (no `trace` field) loads cleanly; Calls with `trace=None` serialise without producing the key.

### 2.5 Orchestrator dispatch (`src/aimigrate/runner/orchestrator.py`, extend)
- [ ] If a `PromptDefinition.tools_path` is set, load `ToolSpec`s and call `client.complete_with_tools(...)`; otherwise existing `client.complete(...)` path. Both produce a `Call` row, only one populates `.trace`.
- [ ] Cost gate / progress bar / checkpointing all unchanged.
- [ ] **Tests** (`tests/unit/test_orchestrator.py`, extend): mixed run (one agent prompt + one plain prompt) writes the right `trace` field on the right rows; a tool-call exception still records a `Call(error=...)` and doesn't crash.

### 2.6 `ToolSelectionEvaluator` (`src/aimigrate/evaluators/tool_selection.py`, new)
- [ ] Implements the `Evaluator` Protocol. Four modes: `exact` (sequence equality), `set` (Jaccard), `first` (only first call), `expected` (default; matches against `example.expected_tools` order-preserving).
- [ ] `expected_no_tools=True` short-circuits to 1.0 if target made 0 calls else 0.0.
- [ ] If `mode="expected"` and example has no `expected_tools`, returns a neutral `EvalRecord` with `metadata={"skipped": "no expected_tools"}` — doesn't pollute the analysis.
- [ ] Source-side score follows the same logic against the source's own trace (so a regression vs. expected is visible as a paired delta).
- [ ] **Tests** (`tests/unit/test_tool_selection.py`, ~25 cases): every mode × every interesting suite shape from PRD §9.2.

### 2.7 Evaluator runner integration (`src/aimigrate/cli/commands/evaluate.py`, extend)
- [ ] Build the tool evaluator(s) when `cfg.evaluators.tool_selection` is configured; skip per (prompt, example) when `call.trace is None`.
- [ ] When source or target trace is missing (one side errored), record a neutral `EvalRecord` with `error="upstream call failed"` (existing pattern).
- [ ] **Tests** (`tests/unit/test_evaluate_command.py`, extend): mixed text + agent prompts; agent-only prompts with no tool evaluator config skip cleanly.

### 2.8 Fixture agent project (`examples/agent/`, new)
- [ ] `aimigrate.yaml` (1 agent prompt, 3 tools, length structural + tool_selection; Gemini defaults like the v0.1 init).
- [ ] `prompts.py` with a single `AGENT_SYSTEM_PROMPT` literal.
- [ ] `tools.yaml` defining 3 tools (`search_orders`, `send_email`, `notify_security_team`).
- [ ] `golden.jsonl` with ~20 examples mixing positive (`expected_tools`), negative (`expected_no_tools`), and tag-tagged for slicing.
- [ ] `README.md` walkthrough.

### Phase 2 verification
- [ ] `aimigrate run --from gemini-2.5-flash --to gemini-2.5-pro` against `examples/agent/` (mocked LLM in tests, real Gemini for manual smoke) writes `Call` rows with `.trace` populated; `aimigrate evaluate` produces `tool_selection` records; existing report still renders without crashing on the new evaluator name.

---

## Phase 3 — `ToolArgumentsEvaluator` + `ToolTraceStructureEvaluator` + doctor checks (PRD Week 3)

Goal: every PRD §2.1 failure mode (1–7) maps to a tool evaluator that catches it, with statistical analysis already wired (no Phase 6 changes needed).

### 3.1 `ToolArgumentsEvaluator` (`src/aimigrate/evaluators/tool_arguments.py`, new)
- [ ] Greedy match by `(tool_name, nearest sequence_index)`; `_match_calls` helper documented as a v0.2 simplification (Hungarian opt-in deferred to v0.3 per PRD risk #3).
- [ ] Per-field strategies: `exact`, `subset` (recursive dict/list), `numeric` (relative-error decay, `numeric_tolerance` clamps), `semantic` (cosine similarity via injected `embeddings_fn` — reuse the embedding helper from `aimigrate.evaluators.semantic.CosineSimilarityEvaluator`).
- [ ] Defensive: missing field on either side → 0.0 with metadata; malformed arguments dict (already flagged by parser as `_parse_error`) → 0.0 with `_parse_error` propagated to metadata.
- [ ] Source-side score = 1.0 (source matched itself by definition); target-side score = per-call mean of per-field scores.
- [ ] **Tests** (`tests/unit/test_tool_arguments.py`, ~30 cases): every strategy × edge case from PRD §9.2; nested dict / list subset; greedy match when same tool called twice; malformed args; embeddings-fn fallback when `None`.

### 3.2 `ToolTraceStructureEvaluator` (`src/aimigrate/evaluators/tool_trace_structure.py`, new)
- [ ] Sub-scores: `call_count` (linear decay past `call_count_tolerance`), `parallelism` (boolean match), `refusal_alignment` (boolean match — refusal regression severity-floored to "high" via metadata), `expected_count_alignment` (when `example.expected_tool_count` set).
- [ ] Combined score = mean of enabled sub-scores. Sub-scores recorded in `metadata` for the report.
- [ ] **Tests** (`tests/unit/test_tool_trace_structure.py`, ~20 cases): every sub-score branch, combined-score arithmetic, all-disabled returns 1.0.

### 3.3 Wire both new evaluators into `aimigrate evaluate` (`src/aimigrate/cli/commands/evaluate.py`, extend)
- [ ] Build them from config in the same `_build_evaluators` function used in v0.1.
- [ ] **Integration tests** (`tests/integration/test_tool_pipeline.py`, new) — at least the 8 scenarios from PRD §9.3:
  - `test_full_pipeline_agent_project_no_regression`
  - `test_full_pipeline_detects_missing_tool_call` (tool_selection CRITICAL)
  - `test_full_pipeline_detects_argument_drift` (tool_arguments MEDIUM)
  - `test_full_pipeline_detects_call_count_explosion` (tool_trace_structure)
  - `test_full_pipeline_text_prompts_unaffected` (mixed run)
  - `test_full_pipeline_refusal_regression` (refusal_alignment forces high)
  - `test_full_pipeline_resume_with_tool_calls` (resume across crash)
  - `test_full_pipeline_cache_hits_for_tool_calls` (second run is free)
- [ ] Each scenario uses the agent fixture project + mocked LiteLLM responses; no real API calls.

### 3.4 `aimigrate doctor` warnings (`src/aimigrate/cli/commands/doctor.py`, extend)
- [ ] Warn (yellow ✗) if any prompt has `tools_path` but no tool evaluator is configured.
- [ ] Warn if any suite example has `expected_tools` but no prompt has `tools_path`.
- [ ] **Tests**: monkey-patched config + suite triggers each warning; happy path stays green.

### 3.5 Statistical-pipeline check (no code changes expected)
- [ ] One test that locks in: bimodal score data (typical of tool evaluators) routes through `aimigrate.analysis.statistics.analyze` → Wilcoxon kicks in (Shapiro-Wilk rejects normality), severity classification still works, BH correction still applies.

### Phase 3 verification
- [ ] All 7 PRD §2.1 failure modes map to a passing integration test.
- [ ] `pytest --cov=aimigrate.evaluators.tool_*` ≥ 90%; overall coverage ≥ 94%.

---

## Phase 4 — Report extension + docs + final verification (PRD Week 4 — re-scoped: no PyPI / launch)

Goal: HTML report renders side-by-side trace diffs; new docs cover the agent workflow; v0.2 ready for tagging.

### 4.1 Report data extension (`src/aimigrate/reports/json.py`, extend)
- [ ] `TopRegression` (or sibling) carries the source/target `ToolTrace` when the regression is on a tool evaluator. `build_report_payload` populates from `Call.trace`.
- [ ] **Tests**: payload round-trip for an agent run preserves trace data.

### 4.2 HTML template extension (`src/aimigrate/reports/templates/report.html.j2` + `report.css`)
- [ ] New "Tool Trace Comparison" subsection per agent prompt, rendered only when at least one row has trace data.
- [ ] Side-by-side diff: source list left, target list right, colour-coded (green = exact match, yellow = different args, red = missing/extra, grey = unrelated).
- [ ] When a top-5 regression is on a tool evaluator, render the trace diff inline in place of the existing source/target text panes.
- [ ] CSS additions stay inlined per v0.1's "single-file HTML, no external assets" rule.
- [ ] **Tests** (`tests/unit/test_reports.py`, extend): rendered HTML for an agent run contains "Tool Trace Comparison" section, has the colour classes, no `<script>` tags appeared.

### 4.3 Docs (`docs/agents.md`, new + updates)
- [ ] `docs/agents.md` — full agent eval workflow walkthrough using `examples/agent/`.
- [ ] Update `docs/configuration.md` with `tools_path` + the three new evaluator blocks.
- [ ] Update `docs/evaluators.md` with a "Tool-call evaluators" section (selection / arguments / trace structure) and the seven failure modes from PRD §2.1.
- [ ] Update `docs/faq.md`: add answer to "what about LangChain?" (we read tool defs from yaml/json; LangChain auto-detect is v0.3) and "what about multi-turn?" (single-turn for v0.2).
- [ ] Update `mkdocs.yml` nav with the new page.

### 4.4 README + CHANGELOG + version
- [ ] README quick-start gains an agent example block.
- [ ] CHANGELOG cuts a `## [0.2.0]` section listing every Phase 1–4 piece (mirrors the v0.1 cut).
- [ ] Bump `pyproject.toml` and `src/aimigrate/__init__.py` to `0.2.0`.

### 4.5 Final verification
- [ ] Fresh-machine smoke: `uv venv && uv pip install -e .[dev] && pytest` green from a cold checkout.
- [ ] Backward compat: every v0.1 example (`examples/simple/`, all `tests/integration/fixtures/`) still runs.
- [ ] All seven PRD §2.1 failure modes covered by an integration test that goes red without the v0.2 code and green with it.
- [ ] All checkboxes above marked `[x]`.

---

## Out of scope (deferred to v0.3+)

Per PRD §3.3 / §13: multi-turn agent evaluation, real tool execution
(sandboxed or otherwise), production trace replay, LangChain
`AgentExecutor` auto-detection, "smart" tool-call diffs (LLM-as-judge per
pair), Hungarian-algorithm tool-call matching, statistical correction
for class imbalance, Anthropic Claude Computer Use evaluation,
streaming-aware evaluation.

If you find yourself coding any of the above, **stop**. Open a GitHub
issue tagged `v0.3` and leave the v0.2 plan alone.

---

## Critical files (quick index)

| Area               | New                                                                                     | Extended (existing v0.1)                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Tool data models   | `src/aimigrate/evaluators/tool_models.py`                                               | —                                                                                                                                       |
| Provider parsing   | `src/aimigrate/evaluators/tool_parser.py`                                               | —                                                                                                                                       |
| Tools loader       | `src/aimigrate/evaluators/tool_loader.py`                                               | —                                                                                                                                       |
| New evaluators     | `tool_selection.py`, `tool_arguments.py`, `tool_trace_structure.py`                     | —                                                                                                                                       |
| Model client       | —                                                                                       | `src/aimigrate/models/client.py` (+ `complete_with_tools`)                                                                              |
| Suite              | —                                                                                       | `src/aimigrate/suite/models.py` (`SuiteExample` + `ExpectedToolCall`)                                                                   |
| Config             | —                                                                                       | `src/aimigrate/config/models.py` (`PromptDefinition.tools_path`, three new evaluator configs, `EvaluatorsConfig`)                       |
| Runner             | —                                                                                       | `src/aimigrate/runner/models.py` (`Call.trace`), `orchestrator.py` (dispatch), `checkpoint.py` (free, via pydantic round-trip)          |
| CLI                | `scripts/smoke_live_tools.py` (manual)                                                  | `cli/commands/{test_call,evaluate,doctor}.py`                                                                                           |
| Reports            | —                                                                                       | `reports/json.py`, `reports/templates/report.html.j2`, `reports/templates/report.css`                                                   |
| Examples           | `examples/agent/{aimigrate.yaml,prompts.py,tools.yaml,golden.jsonl,README.md}`          | —                                                                                                                                       |
| Docs               | `docs/agents.md`                                                                        | `docs/{configuration,evaluators,faq}.md`, `mkdocs.yml`                                                                                  |
| Tests (new)        | `tests/unit/test_{tool_models,tool_parser,tool_selection,tool_arguments,tool_trace_structure}.py`, `tests/integration/test_tool_pipeline.py` | extensions to existing `test_{model_client,evaluate_command,orchestrator,doctor,reports}.py`                                            |
| Fixtures (new)     | `tests/unit/fixtures/tool_responses/{anthropic,openai,gemini}/*.json`, `tests/integration/fixtures/agent_project/`                            | —                                                                                                                                       |

## Reuse from v0.1 (do NOT recreate)

- `_StrictModel` pattern (currently parallel in `config/models.py:27` and `runner/models.py:32`) — reuse, extract to `aimigrate.utils.pydantic` only if a third copy would land.
- `aimigrate.config.loader.ConfigError` / `SuiteError` shape — clone the kind/summary/details + plain/rich rendering pattern for `ToolParseError` and the tool-loader error.
- `aimigrate.models.registry.resolve_model` + `_infer_provider_and_canonical` — reuse for provider detection in `tool_parser.py`.
- `aimigrate.models.client.ModelClient` retry policy + cost helpers — reuse inside `complete_with_tools`.
- `aimigrate.evaluators.base.Evaluator` Protocol + `EvalRecord` + `PairedScore` — every new evaluator implements this.
- `aimigrate.evaluators.semantic.CosineSimilarityEvaluator` embedding helper — reuse for the `semantic` strategy in `ToolArgumentsEvaluator`.
- `aimigrate.runner.checkpoint.{append_call,iter_calls}` — already round-trips arbitrary pydantic; `Call.trace` Just Works.
- `aimigrate.analysis.statistics.analyze` — handles bimodal distributions via the existing Wilcoxon fallback; no changes needed.
- `aimigrate.reports.json.build_report_payload` + jinja template structure — extend, don't rewrite.

## Verification (end-to-end)

1. From a fresh checkout: `uv venv --python 3.14 && uv pip install -e .[dev] && pytest` green.
2. `aimigrate init` (existing template) still produces a v0.1-compatible project that runs end-to-end.
3. `examples/agent/` runs via `aimigrate run` (mocked in tests; real Gemini for the manual smoke) with `tool_selection` / `tool_arguments` / `tool_trace_structure` scores in `scores.jsonl`.
4. `aimigrate analyze` produces severity badges on tool evaluators; `aimigrate report --open` shows the new "Tool Trace Comparison" section.
5. All 7 PRD §2.1 failure modes (wrong tool, dropped tool, arg drift, sequence reorder, parallel↔serial, loop divergence, refusal injection) covered by an integration test that fails without v0.2 and passes with it.
6. `pytest --cov=aimigrate` ≥ 94% (target: don't regress); new modules ≥ 90%.
7. `mypy --strict src/aimigrate` clean; `ruff check` + `ruff format --check` clean.
