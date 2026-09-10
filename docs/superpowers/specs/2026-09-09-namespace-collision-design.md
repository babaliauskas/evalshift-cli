# `evalshift` namespace collision — packaging design

Date: 2026-09-09
Status: approved; implemented in the same change
Plan: `docs/superpowers/plans/2026-09-08-external-review-response.md`, Phase 6 (review point #1)

## Problem

Two distributions ship a top-level `evalshift/` import package:

| Distribution | Repo | Import package | Owns |
| --- | --- | --- | --- |
| `evalshift-sdk` | evalshift-sdk | `evalshift` | capture decorator, trace models, sinks. `evalshift/__init__.py` re-exports the public API. |
| `evalshift` | evalshift-cli | `evalshift` | Typer app, runner, evaluators, reports. |

pip installs both into the same `site-packages/evalshift/` directory. The second
install overwrites the first one's `__init__.py`, the loser's public surface
disappears, and `pip uninstall` of either removes files the other still needs.
Every install page therefore repeats the same rule — separate virtual
environments — in `README.md`, `docs/getting-started.md`, `docs/sdk.md`,
`DOCS.md`, `AGENTS.md`, `llms-full.txt`, `examples/capture-first/`, and on the
SDK side in `README.md`, `DOCS.md`, and `examples/support_agent/README.md`.

The external review (2026-09-08) named this the biggest setup friction. The SDK's
`docs/DECISIONS.md` had it tracked as **D1-followup** with two candidate fixes:
"CLI depends on SDK, or a `[cli]` extra".

## Decision

Option A from the plan, in its second form: **the CLI moves to the import
package `evalshift_cli` and declares `evalshift-sdk` as a runtime dependency.**
The SDK changes no code.

Concretely:

- `src/evalshift/` → `src/evalshift_cli/` in evalshift-cli; every internal import
  `evalshift.<module>` → `evalshift_cli.<module>`.
- Distribution name stays `evalshift`. Console script stays `evalshift`, now
  pointing at `evalshift_cli.cli.main:app`. `python -m evalshift` becomes
  `python -m evalshift_cli`.
- `dependencies` gains `evalshift-sdk>=0.3.0`.
- `import evalshift` is the SDK in every environment that has the CLI.
  `pip install evalshift` is enough for a project that both instruments an agent
  and runs evaluations.

### Why this form of A

The plan's first form — CLI code under `evalshift.cli` inside the SDK's import
root — does not work across two distributions. The SDK's `evalshift/__init__.py`
re-exports the public API (`capture`, `record_model_call`, `configure`, the sinks,
`load_capture`, `SCHEMA_VERSION`, `__version__`). A subpackage contributed by a
second distribution requires `evalshift` to be a PEP 420 namespace package, which
forbids that `__init__.py`; the SDK would lose `import evalshift; evalshift.capture`
and its version attribute. Editable installs of the two repos side by side need
the namespace form as well. A separate top-level package has none of these
constraints and keeps each repo's import root, licence, and CI independent.

### Why not B or C

- **B — rename the SDK to `evalshift_sdk` with a deprecation shim.** Breaks
  `from evalshift import capture` in every instrumented production agent once the
  shim goes, to fix a problem those agents do not have. The SDK's import path is
  its public API; the CLI's is not (see *Compatibility*).
- **C — one distribution with a `[cli]` extra.** A repo merge plus reconciling
  AGPL-3.0-or-later (CLI) with MIT (SDK) per subpackage. Cost out of proportion.

### The dependency

`evalshift-sdk>=0.3.0`, no upper bound.

- 0.3.0 is the first SDK that writes trace schema `2.0.0`, the schema
  `captures/models.py` reads.
- No upper bound because the SDK is stdlib-only (the dependency adds nothing to
  the CLI's environment) and because the CLI reads captures through its own
  pydantic models, guarded by the SDK's vendored-mirror parity test. An SDK
  release cannot break the CLI through the import.
- The dependency is a packaging guarantee, not a code path. The CLI still does not
  import the SDK; files under `.evalshift/` remain the only interface, so the docs
  keep "never call each other" and drop "separate environments". Phases 2 and 3
  may choose to import SDK readers later; nothing here requires it.

## What changes for users

| | Before | After |
| --- | --- | --- |
| Install the CLI | `pip install evalshift` | Same. Pulls `evalshift-sdk` with it. |
| Install the SDK only (production agents) | `pip install evalshift-sdk` | Same. |
| Both in one environment | Not supported. | Supported. |
| `evalshift` command | | Unchanged. |
| `evalshift.yaml`, `golden.jsonl`, `.evalshift/` | | Unchanged. |
| `import evalshift` | Whichever package was installed last. | Always the SDK. |
| `python -m evalshift` | Runs the CLI. | `python -m evalshift_cli`. |
| Scripts importing CLI internals | `from evalshift.models.client import …` | `from evalshift_cli.models.client import …` |
| GitHub Action | Installs `evalshift==<pin>`, calls the console script. | Unaffected. |

## Compatibility and deprecation window

There is no shim. The CLI cannot ship any file under `evalshift/` without
recreating the collision, so the old import path cannot be kept alive even for one
release. The CLI's Python import path has never been documented as public: the
README defines the surface as the command line and the file formats, and the only
internal-path mentions outside this repo's own code are `scripts/`, `docs/faq.md`,
and `llms-full.txt`. The window is therefore the CHANGELOG entry. The change ships
as **0.14.0** (minor, pre-1.0), marked *Breaking*, naming the two renames a user
could notice: `python -m evalshift` and the internal import path. 0.13.x stays on
PyPI, and the GitHub Action pins by version.

Upgrade paths:

1. CLI-only environment: `pip install -U evalshift`. pip removes 0.13's `evalshift/`
   files by RECORD, then installs `evalshift_cli/` and the SDK.
2. Agent environment that already has the SDK: `pip install evalshift` now works.
3. Existing two-environment setups keep working. Nothing forces consolidation.
4. Development checkouts: `uv pip install -e ".[dev]"` again, so the editable
   `.pth` moves from `evalshift` to `evalshift_cli`.

## `evalshift doctor` check

New row `evalshift-sdk`, second in the table after the Python row. Advisory only:
the CLI does not need the SDK to run, so the row never fails the command.

| Condition | Status | Detail |
| --- | --- | --- |
| `evalshift-sdk` distribution installed and `import evalshift` exposes `capture` and `SCHEMA_VERSION` | ok | `<version> (import name evalshift)` |
| Distribution not installed (`--no-deps`, hand-built env) | warn | not installed; `from evalshift import capture` fails here; `pip install evalshift-sdk` |
| Module imports but is not the SDK | warn | `import evalshift` resolves to `<dir>`, not the SDK. An older evalshift CLI's leftover files or a local `evalshift/` directory shadow it. |
| Import raises | warn | `import evalshift` failed: `<exception>` |

Detection imports the module (cheap, stdlib-only, inert without
`EVALSHIFT_CAPTURE=1`) and checks for the two attributes that neither a pre-0.14
CLI package nor a stray directory has. The version comes from
`importlib.metadata`, falling back to `evalshift.__version__` for editable or
otherwise metadata-less installs. The importer and the version lookup are
injectable so tests exercise every row without touching the interpreter; one test
runs against the real environment and so also pins the dependency declaration —
CI's venv only gets the SDK through it.

## Files

evalshift-cli:

- `src/evalshift/` → `src/evalshift_cli/` (git mv), imports rewritten in `src/`,
  `tests/`, `scripts/`.
- `pyproject.toml`: scripts entry, wheel/sdist package paths, coverage source,
  new dependency.
- `mypy.ini`, `ruff.toml` (`known-first-party`), `Makefile`,
  `.github/workflows/ci.yml`, `.pre-commit-config.yaml`: path `src/evalshift_cli`.
- `cli/commands/all.py`: own-record detection compares the logger root to the new
  package name.
- `cli/commands/doctor.py` + `tests/unit/test_doctor.py`: the row above.
- Docs: `README.md`, `docs/getting-started.md`, `docs/sdk.md`, `docs/faq.md`,
  `DOCS.md`, `AGENTS.md`, `llms-full.txt`, `CLAUDE.md`, `CONTRIBUTING.md`,
  `examples/capture-first/` (README and agent docstring), `CHANGELOG.md`.

evalshift-sdk (docs only):

- `docs/DECISIONS.md`: D-pkg and D1-followup marked resolved, pointing here.
- `README.md`, `DOCS.md`, `examples/support_agent/README.md`: drop the two-venv
  rule; say the CLI installs the SDK.
- `tests/conformance/cli_models_vendored.py`: source path in the header.
- `CHANGELOG.md`.

## Out of scope

- The CLI importing SDK code. Phases 2 and 3 decide that on their own merits.
- A repo merge.
- evalshift-client's mirrored `public/cli-llms-full.txt`; it is synced from this
  repo's `llms-full.txt` by that repo's process.
- A `__main__` in the SDK to redirect `python -m evalshift`. The SDK is a library;
  Python's own message ("'evalshift' is a package and cannot be directly
  executed") plus the CHANGELOG is enough.

## Verification

- `make ci` in evalshift-cli with `evalshift-sdk` installed from PyPI, not the
  sibling checkout — the environment a user gets.
- `python -c "import evalshift, evalshift_cli; print(evalshift.__file__)"` prints
  the SDK's path.
- `evalshift doctor` shows the `evalshift-sdk` row as ok.
- `uv build`; the wheel's file list contains no `evalshift/` entries.
- `examples/capture-first`: the agent and `evalshift capture sync` run from one
  environment.
