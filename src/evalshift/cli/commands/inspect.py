"""Implementation of ``evalshift inspect`` debug command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import typer
from rich.console import Console
from rich.table import Table

from evalshift.cli.commands.debug_artifacts import (
    calls_for_example,
    load_scores,
    run_dir,
    traces_for_example,
)


def inspect(
    subject: Annotated[
        str,
        typer.Argument(help="Run id, or 'case' for example-level inspection."),
    ],
    run_id: Annotated[str | None, typer.Argument(help="Run id for 'case'.")] = None,
    example_id: Annotated[str | None, typer.Argument(help="Example id for 'case'.")] = None,
    failed: Annotated[
        bool,
        typer.Option("--failed", help="Only show examples with negative score deltas."),
    ] = False,
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state.", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Inspect recorded run artifacts."""
    if subject == "case":
        if run_id is None or example_id is None:
            raise click.UsageError("usage: evalshift inspect case <run-id> <example-id>")
        _inspect_case(run_id=run_id, example_id=example_id, runs_base=runs_base)
        return
    _inspect_run(run_id=subject, failed=failed, runs_base=runs_base)


def _inspect_run(*, run_id: str, failed: bool, runs_base: Path) -> None:
    console = Console()
    records = load_scores(run_dir(run_id, runs_base))
    if failed:
        records = [r for r in records if r.delta < 0 or r.error is not None]
    table = Table(title=f"inspect · {run_id}")
    table.add_column("example")
    table.add_column("prompt")
    table.add_column("evaluator")
    table.add_column("delta", justify="right")
    table.add_column("categories")
    for record in records:
        cats = record.metadata.get("failure_categories", [])
        cat_text = ", ".join(cats) if isinstance(cats, list) else ""
        table.add_row(
            record.example_id,
            record.prompt_id,
            record.evaluator_name,
            f"{record.delta:+.3f}",
            cat_text,
        )
    console.print(table)


def _inspect_case(*, run_id: str, example_id: str, runs_base: Path) -> None:
    console = Console()
    rd = run_dir(run_id, runs_base)
    calls = calls_for_example(rd, example_id)
    console.print(f"[bold]case[/bold] {example_id}")
    for role in ("source", "target"):
        call = calls.get(role)
        if call is None:
            continue
        console.print(f"\n[bold]{role}[/bold] · {call.model_id}")
        console.print(call.text or "(empty)")
        console.print(f"[dim]cost ${call.cost_usd:.4f} · latency {call.latency_ms} ms[/dim]")

    records = [r for r in load_scores(rd) if r.example_id == example_id]
    if records:
        table = Table(title="scores")
        table.add_column("evaluator")
        table.add_column("delta", justify="right")
        table.add_column("explanation")
        for record in records:
            table.add_row(record.evaluator_name, f"{record.delta:+.3f}", record.explanation)
        console.print(table)

    traces = traces_for_example(rd, example_id)
    if traces:
        trace_table = Table(title="agent trace")
        trace_table.add_column("role")
        trace_table.add_column("tools")
        for role in ("source", "target"):
            trace = traces.get(role)
            if trace is None:
                continue
            trace_table.add_row(
                role, " -> ".join(call.name for call in trace.tool_calls) or "(none)"
            )
        console.print(trace_table)


__all__ = ["inspect"]
