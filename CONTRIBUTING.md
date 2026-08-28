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

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
