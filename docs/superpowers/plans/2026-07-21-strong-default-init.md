# Strong-Default `evalshift init` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the config `evalshift init` generates give honest verdicts with zero edits: advisory vs blocking evaluators, CI-aware small-n budgets, judge fixes, provider-aware scaffold, capture dedup.

**Architecture:** Additive `blocking` flag threaded config → evaluator → `EvalRecord` → policy engine; Wilson-interval budget checks in `analysis/policy.py`; failure paths raise instead of neutral-scoring; `init` gains a provider prompt that parameterises the scaffold template; `capture sync` dedupes on `input_hash`.

**Tech Stack:** Python 3.14, pydantic v2 (`extra="forbid"`), typer, litellm, pytest.

**Spec:** `docs/superpowers/specs/2026-07-21-strong-default-init-design.md`

## Global Constraints

- Python 3.14+, `mypy --strict` clean, `ruff check` + `ruff format` clean.
- All modules `from __future__ import annotations`; Google-style docstrings.
- Config models stay `extra="forbid"`.
- New config field ⇒ doc entry in `docs/configuration.md`; user-visible change ⇒ `CHANGELOG.md` under `## [Unreleased]`.
- **Deviation:** NO git commits — the working tree holds the user's uncommitted in-flight work in overlapping files. Commit steps are replaced by full-suite verify checkpoints; user reviews the diff.

---

### Task 1: `blocking` flag on evaluator configs

**Files:**
- Modify: `src/evalshift/config/models.py` (7 evaluator config classes)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `cfg.evaluators.semantic.blocking: bool` (etc. on all 7 evaluator config models), default `True`.

- [ ] **Step 1: Write failing tests** — `blocking` parses per model, defaults `True`, typo still rejected:

```python
def test_evaluator_blocking_flag_parses_and_defaults() -> None:
    cfg = load_config(write_yaml(..."""
evaluators:
  semantic:
    embedding_model: gemini/gemini-embedding-001
    blocking: false
  llm_judge:
    - criterion_name: eq
      criterion_prompt: which is better
      blocking: false
  structural:
    - type: length
      min_chars: 1
"""))
    assert cfg.evaluators.semantic.blocking is False
    assert cfg.evaluators.llm_judge[0].blocking is False
    assert cfg.evaluators.structural[0].blocking is True
```

- [ ] **Step 2: Run, verify FAIL** (`extra_forbidden` for `blocking`).
- [ ] **Step 3: Implement** — add to each of the 7 evaluator config models:

```python
    blocking: bool = True
```

with docstring line: `blocking: Whether regressions from this evaluator can fail the migration verdict. Advisory (false) evaluators still score and appear in reports.`

- [ ] **Step 4: Run, verify PASS.**

### Task 2: `EvalRecord.blocking` + stamping at evaluate time

**Files:**
- Modify: `src/evalshift/evaluators/base.py` (EvalRecord)
- Modify: `src/evalshift/cli/commands/evaluate.py` (`_build_evaluators`, record construction sites)
- Test: `tests/unit/test_evaluate_command.py` (or existing evaluate test file)

**Interfaces:**
- Produces: `EvalRecord.blocking: bool = True`; every record written by `run_evaluate` carries the config value.
- Mechanism: `_build_evaluators` sets `evaluator.blocking = <config value>` attribute on each instance; all `EvalRecord(...)` constructions in `evaluate.py` pass `blocking=getattr(evaluator, "blocking", True)`.

- [ ] **Step 1: Failing test** — config with `llm_judge.blocking: false` + fake evaluator run ⇒ judge records have `blocking is False`, others `True`; old scores.jsonl row without the key loads as `True`.
- [ ] **Step 2: Verify FAIL.**
- [ ] **Step 3: Implement** (field + stamping at all 5 `EvalRecord(` sites in evaluate.py).
- [ ] **Step 4: Verify PASS.**

### Task 3: Failure paths raise; `drop_params`

**Files:**
- Modify: `src/evalshift/evaluators/llm_judge.py` (remove neutral-tie fallback)
- Modify: `src/evalshift/evaluators/semantic.py` (remove 0/0 fallback)
- Modify: `src/evalshift/models/client.py` (add `"drop_params": True` to completion kwargs in `complete_messages` and `complete_messages_with_tools`)
- Modify: `src/evalshift/evaluators/base.py` (protocol docstring: evaluators may raise; harness records the error)
- Test: `tests/unit/test_llm_judge.py`, `tests/unit/test_semantic.py`, `tests/unit/test_client.py`

**Interfaces:**
- Produces: judge/semantic `.score()` raises `EvaluatorError` on API/parse failure; `_score_one` (already) converts raises to `error=`-stamped records excluded by `_metrics`.

- [ ] **Step 1: Failing tests** — judge with client that raises ⇒ `pytest.raises(EvaluatorError)`; same for semantic embedding failure; client kwargs include `drop_params: True`.
- [ ] **Step 2: Verify FAIL.**
- [ ] **Step 3: Implement:**

```python
        except Exception as exc:
            raise EvaluatorError(
                f"judge call failed for {prompt_id}/{example_id}: {exc}"
            ) from exc
```

(semantic analogous: `embedding failed for ...`). Client: add `"drop_params": True` beside `"temperature"` in both kwargs dicts.

- [ ] **Step 4: Verify PASS; check `_score_one` integration test writes `error=` record.**

### Task 4: Policy engine — advisory partition + Wilson CI budgets

**Files:**
- Modify: `src/evalshift/analysis/policy.py`
- Test: `tests/unit/test_policy.py`

**Interfaces:**
- Produces: `MigrationDecision.advisory: PolicyMetricSummary | None = None`, `MigrationDecision.advisory_regressions: list[BlockingRegression]` (default empty), `BudgetResult.ci_low/ci_high: float | None = None`, `BudgetResult.conclusive: bool = True`; `_wilson_interval(count: int, n: int, z: float = 1.96) -> tuple[float, float]`.
- Verdict semantics: observed within budget → pass; breach with CI excluding budget → fail; breach with CI straddling → `inconclusive` + `reason`.

- [ ] **Step 1: Failing tests:**

```python
def test_advisory_records_do_not_gate(): ...   # 4 advisory regressions, 0 blocking → verdict pass, advisory.regression_rate == 1.0
def test_small_n_breach_is_inconclusive(): ... # n=8 blocking, 3 regressions (0.375 > 0.30), Wilson low < 0.30 → "inconclusive", reason mentions n
def test_large_n_breach_fails(): ...           # n=400, 160 regressions (0.40), Wilson low > 0.30 → "fail"
def test_clean_small_n_passes(): ...           # n=8, 0 regressions → "pass" despite wide CI
def test_advisory_comparisons_not_blocking(): ... # critical semantic comparison from advisory evaluator → advisory_regressions, verdict pass
```

- [ ] **Step 2: Verify FAIL.**
- [ ] **Step 3: Implement** — partition records on `r.blocking`; derive `advisory_evaluators = {r.evaluator_name for r in records if not r.blocking}`; filter comparisons for `_verdict_for`/`_blocking_regressions`; Wilson:

```python
def _wilson_interval(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))
```

Rate budgets get `ci_low/ci_high/conclusive`; `_verdict_for` gains the three-way logic; `reason` composed for inconclusive. `inconclusive_decision` unchanged. Slice decisions use the same logic.

- [ ] **Step 4: Verify PASS; run full `pytest tests/unit/test_policy.py`.**

### Task 5: `capture sync` dedup

**Files:**
- Modify: `src/evalshift/cli/commands/capture.py` (`capture_sync`)
- Test: `tests/unit/test_capture_sync.py` (or existing capture test file)

**Interfaces:**
- Produces: duplicate `(suite, input_hash)` captures skipped (first wins) with summary count; `--keep-duplicates` flag preserves old behavior.

- [ ] **Step 1: Failing test** — two captures, same suite + input_hash ⇒ 1 promoted, summary contains "duplicate"; with `--keep-duplicates` ⇒ 2 promoted.
- [ ] **Step 2: Verify FAIL.**
- [ ] **Step 3: Implement** in the `records_by_suite` build loop:

```python
    seen_inputs: set[tuple[str, str]] = set()
    skipped_duplicates = 0
    ...
        key = (envelope.suite, envelope.input_hash)
        if not keep_duplicates and envelope.input_hash and key in seen_inputs:
            skipped_duplicates += 1
            continue
        seen_inputs.add(key)
```

plus summary segment `f", skipped {skipped_duplicates} duplicate capture(s) (same input)"` and the typer option.

- [ ] **Step 4: Verify PASS.**

### Task 6: Provider-aware, judge-fixed init scaffold

**Files:**
- Modify: `src/evalshift/cli/commands/init.py` (template → function of provider; `--provider` option; TTY prompt)
- Test: `tests/unit/test_init.py`

**Interfaces:**
- Produces: `render_minimal_config(*, profile: str, provider: str = "gemini") -> str`; `--provider [gemini|openai|anthropic]`; non-TTY default `gemini`.
- Scaffold content changes: symmetric criterion prompt (spec §3), `blocking: false` on semantic + llm_judge with comment, provider model table (spec §5), judge-family comment.

- [ ] **Step 1: Failing tests** — `--provider openai` config contains `gpt-5.4-mini`/`gpt-5.6-luna`/`text-embedding-3-small` and NOT `gemini-`; anthropic variant comments out semantic; default remains gemini; criterion has no "TARGET"/"SOURCE" tokens; `blocking: false` present twice; generated YAML round-trips `load_config`.
- [ ] **Step 2: Verify FAIL.**
- [ ] **Step 3: Implement** — `_PROVIDER_MODELS: dict[str, ...]` table; template with `{source_model}`/`{judge_model}`/`{semantic_block}` slots; prompt via `typer.prompt` guarded by `sys.stdin.isatty()`.
- [ ] **Step 4: Verify PASS; run existing `tests/unit/test_init.py` fully (template assertions there will need updating to the new text).**

### Task 7: Docs + changelog + full verify

**Files:**
- Modify: `docs/configuration.md` (blocking field entry; provider-aware init note)
- Modify: `CHANGELOG.md` (`## [Unreleased]`)

- [ ] **Step 1: Write docs entry** for `evaluators.*.blocking` (semantics, default, scaffold defaults) and `capture sync --keep-duplicates`.
- [ ] **Step 2: CHANGELOG entries** (Added: blocking flag, provider prompt, dedup; Changed: judge/semantic failures excluded not neutral; Fixed: reasoning-model judges, scaffold criterion).
- [ ] **Step 3: Full verify:** `ruff check . && ruff format --check . && mypy --strict src/evalshift && pytest -m "not integration"` — all green.

## Self-Review

- Spec coverage: §1→T1+T2, §2→T4, §3→T3+T6(criterion), §4→T1(scaffold flag)+T4, §5→T6, §6→T5, testing→each task, docs/compat→T7. No gaps.
- Placeholders: none (each step has code or exact assertions).
- Type consistency: `blocking: bool` end-to-end; `_wilson_interval` returns `tuple[float, float]` used only in T4.
