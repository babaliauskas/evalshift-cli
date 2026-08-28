"""Implementation of ``evalshift diff`` debug commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands.debug_artifacts import calls_for_example, run_dir, traces_for_example
from evalshift.traces.diff import diff_traces

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
    rd = run_dir(run_id, runs_base)
    traces = traces_for_example(rd, example_id)
    if "source" in traces and "target" in traces:
        console.print(f"[bold]agent trace diff[/bold] {example_id}")
        diff = diff_traces(traces["source"], traces["target"])
        for item in diff.items:
            source_name = item.source_name or "-"
            target_name = item.target_name or "-"
            suffix = (
                f" {item.field}: {item.source_value!r} -> {item.target_value!r}"
                if item.field
                else ""
            )
            console.print(
                f"{item.kind:14} {source_name:24} {target_name:24} {item.category}{suffix}",
            )
        if not diff.items:
            console.print("no trace differences")
        return

    calls = calls_for_example(rd, example_id)
    source = calls.get("source")
    target = calls.get("target")
    console.print(f"[bold]diff case[/bold] {example_id}")
    console.print("\n[bold]source[/bold]")
    console.print(source.text if source else "(missing)")
    console.print("\n[bold]target[/bold]")
    console.print(target.text if target else "(missing)")


__all__ = ["diff_app"]
