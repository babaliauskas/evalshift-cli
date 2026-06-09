"""Implementation of ``evalshift replay`` debug commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from evalshift.cli.commands.debug_artifacts import calls_for_example, run_dir

replay_app = typer.Typer(help="Replay recorded EvalShift artifacts without live model calls.")


@replay_app.command(name="case")
def replay_case(
    run_id: Annotated[str, typer.Argument(help="Run id.")],
    example_id: Annotated[str, typer.Argument(help="Example id.")],
    model: Annotated[
        Literal["source", "target"],
        typer.Option("--model", help="Recorded side to replay."),
    ] = "target",
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state.", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Print the recorded source or target call for one example."""
    console = Console()
    call = calls_for_example(run_dir(run_id, runs_base), example_id).get(model)
    if call is None:
        console.print(f"[red]✗[/red] no recorded {model} call for {example_id}")
        raise typer.Exit(code=1)
    console.print(call.text)


__all__ = ["replay_app"]
