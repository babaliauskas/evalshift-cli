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
