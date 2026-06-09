"""Top-level Typer application.

This module is intentionally thin: it constructs the ``app`` object used as the
console-script entry point and registers each subcommand. Subcommand logic
lives in :mod:`evalshift.cli.commands`.
"""

from __future__ import annotations

from typing import Annotated

import typer

from evalshift import __version__
from evalshift.cli.commands.all import all_command as _all
from evalshift.cli.commands.analyze import analyze_command as _analyze
from evalshift.cli.commands.bundle import bundle as _bundle
from evalshift.cli.commands.cache import cache_app
from evalshift.cli.commands.diff import diff_app
from evalshift.cli.commands.doctor import doctor as _doctor
from evalshift.cli.commands.evaluate import evaluate as _evaluate
from evalshift.cli.commands.init import init as _init
from evalshift.cli.commands.inspect import inspect as _inspect
from evalshift.cli.commands.login import login as _login
from evalshift.cli.commands.push import push as _push
from evalshift.cli.commands.replay import replay_app
from evalshift.cli.commands.report import report as _report
from evalshift.cli.commands.run import run as _run
from evalshift.cli.commands.test_call import test_call as _test_call
from evalshift.cli.commands.validate import validate as _validate
from evalshift.cli.commands.whoami import whoami as _whoami

app = typer.Typer(
    name="evalshift",
    help="Run your prompts on two LLMs and find out what regressed.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"evalshift {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the EvalShift version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """EvalShift: safer LLM model migrations via paired evaluation."""


app.command(name="doctor")(_doctor)
app.command(name="init")(_init)
app.command(name="run")(_run)
app.command(name="evaluate")(_evaluate)
app.command(name="analyze")(_analyze)
app.command(name="report")(_report)
app.command(name="login")(_login)
app.command(name="whoami")(_whoami)
app.command(name="bundle")(_bundle)
app.command(name="push")(_push)
app.command(name="all")(_all)
app.add_typer(cache_app, name="cache")
app.command(name="inspect")(_inspect)
app.add_typer(diff_app, name="diff")
app.add_typer(replay_app, name="replay")
# `validate` and `test-call` are dev/debug commands hidden from the
# top-level help. They'll be removed (or relocated under a hidden
# --debug group) in Phase 8.4.
app.command(name="validate", hidden=True)(_validate)
app.command(name="test-call", hidden=True)(_test_call)


if __name__ == "__main__":
    app()
