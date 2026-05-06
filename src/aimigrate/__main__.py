"""Entry point for ``python -m aimigrate``.

Delegates to the Typer app defined in :mod:`aimigrate.cli.main`.
"""

from __future__ import annotations

from aimigrate.cli.main import app


def main() -> None:
    """Invoke the AIMigrate CLI."""
    app()


if __name__ == "__main__":
    main()
