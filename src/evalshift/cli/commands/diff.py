"""Implementation of ``evalshift diff`` debug commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands.debug_artifacts import calls_for_example, run_dir

diff_app = typer.Typer(help="Diff recorded source/target artifacts.")


@diff_app.command(name="case")
def diff_case(
    run_id: Annotated[str, typer.Argument(help="Run id.")],
    example_id: Annotated[str, typer.Argument(help="Example id.")],
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state.", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Print a compact source/target output diff for one example."""
    console = Console()
    calls = calls_for_example(run_dir(run_id, runs_base), example_id)
    source = calls.get("source")
    target = calls.get("target")
    console.print(f"[bold]diff case[/bold] {example_id}")
    console.print("\n[bold]source[/bold]")
    console.print(source.text if source else "(missing)")
    console.print("\n[bold]target[/bold]")
    console.print(target.text if target else "(missing)")


__all__ = ["diff_app"]
