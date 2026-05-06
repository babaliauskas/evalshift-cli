"""Top-level Typer application.

This module is intentionally thin: it constructs the ``app`` object used as the
console-script entry point and registers each subcommand. Subcommand logic
lives in :mod:`aimigrate.cli.commands`.
"""

from __future__ import annotations

from typing import Annotated

import typer

from aimigrate import __version__
from aimigrate.cli.commands.analyze import analyze_command as _analyze
from aimigrate.cli.commands.cache import cache_app
from aimigrate.cli.commands.doctor import doctor as _doctor
from aimigrate.cli.commands.evaluate import evaluate as _evaluate
from aimigrate.cli.commands.init import init as _init
from aimigrate.cli.commands.report import report as _report
from aimigrate.cli.commands.run import run as _run
from aimigrate.cli.commands.test_call import test_call as _test_call
from aimigrate.cli.commands.validate import validate as _validate

app = typer.Typer(
    name="aimigrate",
    help="Run your prompts on two LLMs and find out what regressed.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"aimigrate {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the AIMigrate version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """AIMigrate: safer LLM model migrations via paired evaluation."""


app.command(name="doctor")(_doctor)
app.command(name="init")(_init)
app.command(name="run")(_run)
app.command(name="evaluate")(_evaluate)
app.command(name="analyze")(_analyze)
app.command(name="report")(_report)
app.add_typer(cache_app, name="cache")
# `validate` and `test-call` are dev/debug commands hidden from the
# top-level help. They'll be removed (or relocated under a hidden
# --debug group) in Phase 8.4.
app.command(name="validate", hidden=True)(_validate)
app.command(name="test-call", hidden=True)(_test_call)


if __name__ == "__main__":
    app()
