# Contributing to EvalShift

Thanks for your interest! EvalShift is open source under the
[AGPL-3.0-or-later](LICENSE) license. By submitting a contribution you agree
that it is provided under the same license.

## Development setup

We use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Clone and enter the repo
git clone https://github.com/babaliauskas/EvalShift.git
cd evalshift

# Create a virtualenv with Python 3.11 and install dev deps
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks (wires both the commit and the pre-push stage)
pre-commit install
```

The pre-push stage runs the same four commands as CI — `ruff check .`,
`ruff format --check .`, `mypy --strict src/evalshift`, `pytest` — over the
whole tree, so a push that would turn CI red is refused locally. Run them by
hand any time with `make ci`.

## Common commands

```bash
pytest                                  # run the test suite
pytest -m "not integration"             # unit tests only
ruff check .                            # lint
ruff format .                           # auto-format
mypy --strict src/evalshift             # type-check
pre-commit run --all-files              # everything pre-commit runs
```

## Style

- Python 3.11+, fully type-hinted, `mypy --strict` clean.
- Public functions and classes have Google-style docstrings.
- Tests live alongside the module they cover, mirrored under `tests/unit` or `tests/integration`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

## Pull requests

1. Create a feature branch.
2. Add tests for new behavior.
3. Update `CHANGELOG.md` under `## [Unreleased]`.
4. Ensure CI is green (lint + type-check + tests).
5. Request review.

## Releasing (maintainers)

A release is a commit, a tag, and nothing else — CI does the publishing.

1. On `main`, bump `version` in `pyproject.toml` and rename the CHANGELOG's
   `## [Unreleased]` section to `## [X.Y.Z] - YYYY-MM-DD` (leave a fresh empty
   `[Unreleased]` above it), in a single commit titled `chore(release): X.Y.Z`.
2. Tag that commit and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z — <one-line headline from the CHANGELOG>"
   git push origin vX.Y.Z
   ```

3. The tag push triggers [release.yml](.github/workflows/release.yml), which
   asserts the tag matches `pyproject.toml`, builds with `uv build`, publishes
   to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
   (no token secret), and sends a `repository_dispatch` to
   `babaliauskas/evalshift-action` so its `bump-cli-pin` workflow opens the
   pin-bump PR immediately.

One-time setup, held outside the repo:

- **PyPI trusted publisher** on the `evalshift` project: repository
  `babaliauskas/evalshift-cli`, workflow `release.yml`, environment `pypi`.
- **`EVALSHIFT_ACTION_DISPATCH_TOKEN`** (optional repo secret): fine-grained
  PAT with contents: write on `babaliauskas/evalshift-action`. Without it the
  dispatch is skipped and the action repo's daily PyPI poll picks the release
  up within a day.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
