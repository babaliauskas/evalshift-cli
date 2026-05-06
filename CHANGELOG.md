# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository scaffolding: `pyproject.toml`, lint/type/test config, CI, MIT license.
- Empty `src/aimigrate` package skeleton matching the planned module layout.
- `aimigrate.config.models`: pydantic v2 schema for `aimigrate.yaml` (prompts, evaluators, slices, defaults) with strict validation, `extra='forbid'`, and detection-mode field invariants.
- `aimigrate.config.loader`: `load_config()` plus a structured `ConfigError` carrying file path, error kind, and per-field details with both plain-text and Rich panel rendering.
- `aimigrate doctor` command: reports Python version, presence of provider API keys, and `aimigrate.yaml` validity. Exits 1 only on hard failures (invalid config); missing keys are warnings.
- `aimigrate init` command: scaffolds `aimigrate.yaml` + `prompts.py` + `golden.jsonl` starter files. Refuses to overwrite by default; `--force` enables overwrite, `--directory` targets a non-cwd path.
- `aimigrate.suite` package: pydantic models (`SuiteExample`, `Suite`) plus a JSONL loader (`load_jsonl`) with line-numbered error reporting, blank-line tolerance, multi-error collection, and `extra='forbid'` rejection of unknown row keys.
- `aimigrate.parsers` package: `PromptParser` protocol with `ManualParser` for inline prompts and `PythonStringParser` that safely AST-extracts string-literal prompts from `.py` files. Non-literal value forms (f-strings, concatenation, function calls, attribute access, name references) are explicitly rejected; the parser never runs user code.
- `aimigrate.utils.templating` module: `extract_variables`, `render` (strict, missing-var aware), and a bulk `validate_suite_against_prompts` pre-flight check that collects every (prompt, example) compatibility issue at once.
- `aimigrate validate` command (hidden dev command): loads config + suite + prompts and verifies they are mutually compatible. Will be removed or relocated under a hidden `--debug` group at the v0.1.0 cut.
- `aimigrate.models.registry`: hard-coded metadata for the supported model set (Anthropic Claude Sonnet 4.5 / Opus 4.5, OpenAI GPT-4o / mini, Gemini 2.5 Pro / Flash) plus PDF-style aliases (`claude-4.5-sonnet`, `gemini-2.5-flash`, etc.) that resolve to LiteLLM canonical IDs.
- `aimigrate.cache`: SQLAlchemy 2.0 schema + async `CacheStore` (`get`/`put`/`clear`/`count`) with sha256 cache keys and a 7-day default TTL. Backing file at `~/.aimigrate/cache.db`. Added `greenlet` runtime dep (required by SQLAlchemy async).
- `aimigrate cache clear` CLI command (under a new `cache` sub-app) wipes the local cache.
- `aimigrate.models.client`: async `ModelClient` wrapping `litellm.acompletion` with uniform error mapping (`RateLimitError` / `AuthError` / `ModelError`), full-jitter exponential backoff retries (`AuthError` short-circuits), and per-call cost + token bookkeeping in a `CompletionResult` dataclass.
- `aimigrate.utils.cost`: `estimate_run_cost(...)` for pre-flight cost estimation with a sample-based prompt-token average and defensive fallbacks when LiteLLM lacks pricing or token-counter data.
- `aimigrate test-call` command (hidden): one-shot smoke test that sends a single prompt to a single model. Renders the response, token counts, cost, and latency in a Rich panel; exits 1 on auth/rate-limit/unknown-model errors.
