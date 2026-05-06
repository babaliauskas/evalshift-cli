# AIMigrate MVP — TODO

> Living checklist for the AIMigrate CLI MVP.
> Tick `[ ]` → `[x]` the moment a piece is done. One checkbox = one focused commit (or a tight series).
> Source spec: `aimigrate-cli-mvp-build-plan.md.pdf`.

## North star

A `pip install aimigrate` package that proves the wedge:
> Given my prompts and inputs, run them on two LLM models and tell me — with statistical confidence — what regressed.

## Working agreement

- [x] Python `>=3.14`, fully type-hinted, `mypy --strict` clean.
- [x] Tests land in the same commit as the code they cover. New module ⇒ new test file.
- [x] CI runs **fully mocked** integration tests (hermetic, no API keys, no cost). A separate `scripts/smoke_live.py` (run manually with real keys before tagging a release) catches LiteLLM upstream breakage.
- [x] `ruff check`, `ruff format --check`, `mypy --strict src/aimigrate`, and `pytest` must all pass before marking a checkbox done.
- [x] Public functions/classes get Google-style docstrings. Modules get a one-line module docstring.
- [x] Update `CHANGELOG.md` (Keep a Changelog format) under `## [Unreleased]` for any user-visible change.
- [x] If a task uncovers scope drift, stop and update this file rather than silently expanding.

---

## Phase 0 — Repo bootstrap

Goal: a clean, lintable, testable skeleton. No product code yet.

- [x] `pyproject.toml` with PEP 621 metadata: name `aimigrate`, version `0.0.1`, MIT, Python `>=3.14`.
- [x] Production deps (typer, pydantic, httpx, litellm, rich, jinja2, numpy, scipy, sqlalchemy, aiosqlite, pyyaml, tiktoken, jsonschema, pygments).
- [x] Dev deps (pytest, pytest-asyncio, pytest-cov, ruff, mypy, pre-commit, mkdocs-material, respx, type stubs).
- [x] `uv venv --python 3.14` + bootstrap docs in `CONTRIBUTING.md`.
- [x] Directory tree: `src/aimigrate/{cli,config,parsers,models,suite,runner,evaluators,analysis,reports,cache,utils}` + `tests/{unit,integration}` + `docs/` + `examples/`. Module docstring on every package.
- [x] `.gitignore`, `LICENSE` (MIT), expanded `README.md` with badges + status, `CHANGELOG.md`, `CONTRIBUTING.md`.
- [x] `ruff.toml` (line 100, target py314, rules: E/F/I/N/UP/B/A/C4/SIM/RUF).
- [x] `mypy.ini` (`strict = true` on `src/aimigrate`, pydantic plugin).
- [x] `pytest` config in `pyproject.toml` (asyncio mode auto, coverage on).
- [x] `.pre-commit-config.yaml` (ruff + mypy + std hooks).
- [x] `.github/workflows/ci.yml` (lint + format-check + mypy + pytest on Python 3.14).
- [x] `py.typed` marker.
- [x] Local verification: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` all green.
- [ ] Push to `origin/main`; verify GitHub Actions CI run is green.

---

## Phase 1 — CLI skeleton, config models, `init` & `doctor`

Goal: installable CLI; `aimigrate --version`, `aimigrate doctor`, `aimigrate init` all work.

### 1.1 Pydantic config models (`src/aimigrate/config/models.py`)
- [x] `PromptDefinition`, `Defaults`, `SliceConfig`, `StructuralEvaluatorConfig`, `LLMJudgeConfig`, plus `SemanticEvaluatorConfig` + typed `EvaluatorsConfig` wrapper (deviation from PDF's loose `dict` for `mypy --strict`).
- [x] Top-level `AIMigrateConfig` with `version: Literal[1]`.
- [x] Field validators: `manual` ⇒ `content` required, `path`/`variable` forbidden; `python_string` ⇒ `path`+`variable` required, `content` forbidden. Structural-evaluator type-specific field requirements. Duplicate prompt-id detection.
- [x] **Tests** (`tests/unit/test_config_models.py`, 38 cases): valid round-trip via `model_dump`; every invalid path raises `ValidationError` with a matched message. 91% coverage on `src/aimigrate`.

### 1.2 Config loader (`src/aimigrate/config/loader.py`)
- [x] `load_config(path) -> AIMigrateConfig` — yaml load + pydantic validate.
- [x] `ConfigError` exception with `kind` (missing/not_a_file/yaml_parse/not_a_mapping/schema), `summary`, `details: list[ConfigErrorDetail]`. Both `format_plain()` and `format_rich()` (Panel) renderers.
- [x] **Tests** (`tests/unit/test_config_loader.py`, 21 cases): happy path round-trip, missing/dir/empty/list paths, malformed YAML reports line, schema errors carry field paths, multiple errors collected at once. 93% overall coverage.

### 1.3 CLI entrypoint (`src/aimigrate/cli/main.py`)
- [x] Typer `app` with `--version` flag.
- [x] Register `init`, `run`, `report`, `doctor` (stubs exit 2 until implemented).
- [x] Wire `[project.scripts] aimigrate = "aimigrate.cli.main:app"`.
- [x] **Tests**: `CliRunner` for `--version`, `--help`, stub exit codes.

### 1.4 `aimigrate doctor` (`src/aimigrate/cli/commands/doctor.py`)
- [x] Pure `run_checks(cwd, env) -> list[CheckResult]` core (Python version, three provider env keys, `aimigrate.yaml` presence + validity).
- [x] Rich table with ✓/✗ glyphs (green ✓ for ok, yellow ✗ for warnings, red ✗ for hard failures).
- [x] Exit 0 unless a hard failure exists (currently: invalid config); missing keys / no config = warn but exit 0.
- [x] **Tests** (`tests/unit/test_doctor.py`, 13 cases): pure-function tests for every status × every check, plus CLI-level tests via `CliRunner` covering exit codes 0 and 1. 95% coverage.

### 1.5 `aimigrate init` (`src/aimigrate/cli/commands/init.py`)
- [x] Heavily commented starter `aimigrate.yaml` covering every common section (prompts, defaults, evaluators, slices).
- [x] Example `prompts.py` (with `GREET_PROMPT` referenced by the yaml) and a 3-row `golden.jsonl`.
- [x] `--force` / `-f` to overwrite; default refuses and lists every conflicting file.
- [x] `--directory` / `-d` to scaffold into a target dir; auto-creates missing parents.
- [x] **Tests** (`tests/unit/test_init.py`, 11 cases): files written + parse via `load_config`; `prompts.py` is valid Python and defines every variable the yaml references; suite is valid JSONL; conflict detection + `--force` overwrite + `--directory` flag.

### 1.6 Phase docs
- [x] `docs/getting-started.md` covers install + API keys + `init` + `doctor`.
- [x] README quick-start updated to reflect what's actually shipped vs. in-progress.

---

## Phase 2 — Prompt + suite loading

### 2.1 Suite models (`src/aimigrate/suite/models.py`)
- [ ] `SuiteExample` (id, inputs, tags, optional expected).
- [ ] `Suite` with `by_tag`, `validate_inputs_for_template`.
- [ ] **Tests**.

### 2.2 Suite loader (`src/aimigrate/suite/loader.py`)
- [ ] `load_jsonl(path) -> Suite` with line-by-line errors; reject duplicate ids.
- [ ] **Tests**: well-formed, malformed line, empty, duplicates.

### 2.3 Prompt parsers (`src/aimigrate/parsers/`)
- [ ] `base.py` Protocol returning `PromptTemplate`.
- [ ] `manual.py` reads inline `content` from config.
- [ ] `python_string.py` AST-walks `.py` for module-level Assign matching `variable`; rejects non-literal strings (no eval).
- [ ] **Tests** for each parser, including AST safety.

### 2.4 Template substitution (`src/aimigrate/utils/templating.py`)
- [ ] `string.Formatter` to extract template vars; `render(template, inputs)` raises on missing.
- [ ] `validate_suite_against_prompts(suite, prompts)` aborts with full missing-var list.
- [ ] **Tests**: happy path, missing/extra vars, escaped braces.

### 2.5 Temporary `validate` command
- [ ] Hidden dev command confirming prompts + suite compatible (removed in Phase 8).
- [ ] **Tests** with `tests/integration/fixtures/`.

---

## Phase 3 — Model client, registry, cache

### 3.1 Model registry (`src/aimigrate/models/registry.py`)
- [ ] Hard-coded metadata for MVP set: `claude-4.5-sonnet`, `claude-5-sonnet`, `claude-5-opus`, `gpt-5`, `gpt-5-mini`, `gemini-2.5-pro`, `gemini-2.5-flash`.
- [ ] `get_model(id)` / `list_supported()`; unknown raises `UnknownModelError`.
- [ ] **Tests**.

### 3.2 Cache schema (`src/aimigrate/cache/schema.py`)
- [ ] SQLAlchemy 2.0 `Base` + `CachedCall` per PDF §5.3.
- [ ] Async engine factory at `~/.aimigrate/cache.db`.
- [ ] **Tests** with in-memory sqlite.

### 3.3 Cache store (`src/aimigrate/cache/store.py`)
- [ ] `cache_key(...)` sha256 over canonical JSON.
- [ ] async `get/put`, default 7-day TTL.
- [ ] `aimigrate cache clear` CLI.
- [ ] **Tests**: hit/miss/expiry/key stability.

### 3.4 Model client (`src/aimigrate/models/client.py`)
- [ ] Thin async `litellm.acompletion` wrapper.
- [ ] Map errors to `RateLimitError`, `AuthError`, `ModelError`.
- [ ] Exponential-backoff retry (3 attempts, jitter).
- [ ] Per-call cost via `litellm.completion_cost`; tokens via tiktoken / native.
- [ ] **Tests** with `respx` / monkeypatch.

### 3.5 Cost estimator (`src/aimigrate/utils/cost.py`)
- [ ] Sample-based: first 5 examples → avg tokens × N×M×2.
- [ ] **Tests**.

### 3.6 `aimigrate test-call` smoke command
- [ ] `--model X --prompt "say hello"` proves connectivity.
- [ ] **Tests** with mocked client.

---

## Phase 4 — Run orchestrator

### 4.1 Run/Evaluation models (`src/aimigrate/runner/models.py`)
- [ ] `Run` and `Evaluation` pydantic models.
- [ ] **Tests** (model_dump_json round-trip).

### 4.2 Checkpoint persistence (`src/aimigrate/runner/checkpoint.py`)
- [ ] Atomic `state.json` (write-temp + rename) per PDF §5.4.
- [ ] Append-only `raw.jsonl`.
- [ ] `resume_run(run_dir)`; abort on `config_hash` change.
- [ ] **Tests**: simulated crash + resume.

### 4.3 Orchestrator (`src/aimigrate/runner/orchestrator.py`)
- [ ] Async loop with `asyncio.Semaphore(concurrency)`.
- [ ] Cache → live call → record per `(prompt, example, model)`.
- [ ] Rich progress bar with cost + ETA.
- [ ] Pre-flight cost confirmation (auto-yes if `--yes` or under `defaults.max_cost_usd`).
- [ ] Checkpoint every 50 calls.
- [ ] **Tests** with mocked model client.

### 4.4 `aimigrate run` command
- [ ] CLI args `--from`, `--to`, `--prompt`, `--suite`, `--config`, `--resume`, `--yes`.
- [ ] Output dir `.aimigrate/runs/<run_id>/`.
- [ ] **Tests** end-to-end with fixture suite + mocked LLM.

### 4.5 Validation milestone
- [ ] Dogfood: 1 prompt × 50 examples × 2 models → `raw.jsonl` with 100 lines.

---

## Phase 5 — Evaluators

### 5.1 Evaluator protocol (`src/aimigrate/evaluators/base.py`)
- [ ] `EvalResult` pydantic model (PDF §5.2).
- [ ] `Evaluator` Protocol with async `evaluate(...)`.
- [ ] **Tests**.

### 5.2 Structural (`src/aimigrate/evaluators/structural.py`)
- [ ] `JsonSchemaEvaluator`, `RegexEvaluator`, `LengthEvaluator`.
- [ ] **Tests** for each.

### 5.3 Semantic (`src/aimigrate/evaluators/semantic.py`)
- [ ] `CosineSimilarityEvaluator` with `text-embedding-3-small`.
- [ ] Embeddings cached.
- [ ] **Tests** with mocked embeddings.

### 5.4 LLM-as-judge (`src/aimigrate/evaluators/llm_judge.py`)
- [ ] `PairwiseJudgeEvaluator` with order-randomization, defensive JSON parse.
- [ ] **Tests** for each verdict + malformed output.

### 5.5 `aimigrate evaluate <run-id>` command
- [ ] Loads `raw.jsonl` → runs evaluators → writes `scores.jsonl`.
- [ ] **Tests** end-to-end.

---

## Phase 6 — Statistics & slicing

### 6.1 Slicing (`src/aimigrate/analysis/slicing.py`)
- [ ] Tag grouping, multi-membership, implicit `all` slice.
- [ ] Per-slice aggregates (n, mean, std, min, max, deltas).
- [ ] **Tests**: overlapping tags, empty slices.

### 6.2 Statistics (`src/aimigrate/analysis/statistics.py`)
- [ ] Pair scores; require n≥5; flag n<20.
- [ ] Shapiro-Wilk → paired t-test or Wilcoxon.
- [ ] Cohen's d (paired).
- [ ] 95% CI on effect size (analytical or bootstrap).
- [ ] Benjamini-Hochberg correction across all (prompt × evaluator × slice).
- [ ] Severity classification (critical/high/medium/low/improved/none) per PDF §5.5.
- [ ] **Tests** with synthetic distributions; cross-check BH against `statsmodels`.

### 6.3 `aimigrate analyze <run-id>` command
- [ ] Reads `scores.jsonl` → writes `analysis.json`.
- [ ] **Tests**: golden snapshot.

### 6.4 Statistical-rigor review
- [ ] Self-review: every claim backed by p, corrected p, effect size, CI, n.
- [ ] `docs/methodology.md` documents the math.

---

## Phase 7 — HTML report

### 7.1 Report data (`src/aimigrate/reports/json.py`)
- [ ] `build_report_payload(run_id) -> ReportData`; persist `report.json`.
- [ ] **Tests** (snapshot).

### 7.2 HTML template (`src/aimigrate/reports/templates/report.html.j2` + `report.css`)
- [ ] Self-contained: inline CSS, no external assets.
- [ ] Sections: header, executive summary, per-prompt deep dive (aggregate + slice tables, top-5 worst regressions side-by-side), methodology appendix.
- [ ] Severity color coding consistent across the file.
- [ ] Static (no JS) for MVP.

### 7.3 Renderer (`src/aimigrate/reports/html.py`)
- [ ] `render_html(report_data) -> str`; CLI writes `report.html`.
- [ ] Pygments JSON highlighting.
- [ ] **Tests**: parseable HTML, expected anchors, severity classes.

### 7.4 `aimigrate report <run-id>` command
- [ ] Regenerate from stored data; `--open` via `webbrowser`.
- [ ] **Tests**.

### 7.5 Manual review
- [ ] Eyeball on dogfood run; iterate on copy + layout.

---

## Phase 8 — Polish, docs, OSS readiness

### 8.1 Documentation site (MkDocs Material)
- [ ] `mkdocs.yml` nav: Home, Getting Started, Configuration, Evaluators, Methodology, FAQ, Changelog.
- [ ] `docs/configuration.md` covers every `aimigrate.yaml` field.
- [ ] `docs/evaluators.md` covers each evaluator + when to use it.
- [ ] `docs/faq.md` includes "does this send my prompts to your servers?" → no, fully local.
- [ ] GitHub Pages deploy workflow.

### 8.2 Examples (`examples/`)
- [ ] `examples/simple/` — single prompt, 20 examples, all evaluators.
- [ ] `examples/advanced/` — multi-prompt, slices, JSON schema, judge criterion.
- [ ] Both runnable in CI with mocked models.

### 8.3 README polish
- [ ] 30-second pitch, install, 60-second example, screenshot, badges.
- [ ] "Status: alpha" banner; clear non-goals.

### 8.4 OSS hygiene
- [ ] `CONTRIBUTING.md` covers tests/style/commits.
- [ ] `CODE_OF_CONDUCT.md` (Contributor Covenant).
- [ ] Issue + PR templates under `.github/`.
- [ ] `SECURITY.md` with disclosure email.
- [ ] Remove dev-only `validate`/`test-call` commands or move under hidden `--debug`.
- [ ] Bump version to `0.1.0`; cut changelog section.

### 8.5 Final verification
- [ ] Fresh-machine simulation under 5 minutes.
- [ ] `pytest --cov=aimigrate` ≥ 85% line coverage.
- [ ] `mypy --strict src/aimigrate` clean.
- [ ] `ruff check` clean.
- [ ] All checkboxes above marked `[x]`.

---

## Out of scope (deferred to v0.2+)

Per PDF §1.2: hosted backend / web UI / accounts / billing / TS SDK / LangChain auto-detect / OpenAI message auto-detect / YAML+Jinja templates / CSV suites / Langfuse-LangSmith import / CI integrations / multi-criterion judge / migration suggestions / auto-clustering / custom Python evaluator plugins / non-SQLite caching / >2-model comparison / hosted auth / cost forecasting beyond pre-flight estimate / domain & landing page / launch posts.

If you find yourself coding any of the above, **stop**. Open an issue, leave the MVP alone.
