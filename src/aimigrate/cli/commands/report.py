"""Implementation of ``aimigrate report <run-id>``.

Builds the report payload from the run directory's artefacts, writes
``report.html`` (and ``report.json``), and optionally opens the HTML
file in the user's default browser via ``--open``.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from aimigrate.cli.commands.doctor import CONFIG_FILENAME
from aimigrate.reports.html import write_html
from aimigrate.reports.json import build_report_payload, write_report_json
from aimigrate.runner.checkpoint import (
    CheckpointError,
    read_state,
    run_dir_for,
)


def report(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to report on."),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help=f"Path to aimigrate.yaml (default: ./{CONFIG_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(CONFIG_FILENAME),
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open",
            help="Open the rendered HTML report in your default browser.",
        ),
    ] = False,
    runs_base: Annotated[
        Path,
        typer.Option(
            "--runs-base",
            help="Base directory for run state (advanced).",
            hidden=True,
            file_okay=False,
        ),
    ] = Path(".aimigrate") / "runs",
) -> None:
    """Render a single-file HTML report from a fully-analysed run."""
    console = Console()

    run_dir = run_dir_for(run_id, runs_base)
    try:
        read_state(run_dir)  # validates state.json exists + parses
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        payload = build_report_payload(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        console.print(
            "Hint: run [cyan]aimigrate evaluate[/cyan] then "
            "[cyan]aimigrate analyze[/cyan] before [cyan]report[/cyan].",
        )
        raise typer.Exit(code=1) from exc

    json_path = write_report_json(payload, run_dir)
    html_path = write_html(payload, run_dir)

    console.print(f"[green]✓[/green] {html_path}")
    console.print(f"[green]✓[/green] {json_path}")

    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())

    _ = config_path  # reserved for future config-driven rendering knobs.


__all__ = ["report"]
