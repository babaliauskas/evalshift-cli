"""Entry point for ``python -m evalshift``.

Delegates to the Typer app defined in :mod:`evalshift.cli.main`.
"""

from __future__ import annotations

from evalshift.cli.main import app


def main() -> None:
    """Invoke the EvalShift CLI."""
    app()


if __name__ == "__main__":
    main()
