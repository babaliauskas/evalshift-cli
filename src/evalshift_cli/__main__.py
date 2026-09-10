"""Entry point for ``python -m evalshift_cli``.

Delegates to the Typer app defined in :mod:`evalshift_cli.cli.main`.
"""

from __future__ import annotations

from evalshift_cli.cli.main import app


def main() -> None:
    """Invoke the EvalShift CLI."""
    app()


if __name__ == "__main__":
    main()
