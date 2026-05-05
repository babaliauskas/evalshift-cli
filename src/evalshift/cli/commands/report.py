"""Implementation of ``evalshift report <run-id>``.

Builds the report payload from the run directory's artefacts, writes
``report.html`` (and ``report.json``), and optionally opens the HTML
file in the user's default browser via ``--open``.
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.config.loader import ConfigError, load_config
from evalshift.reports.html import write_html
from evalshift.reports.json import build_report_payload, write_report_json
from evalshift.runner.checkpoint import (
    CheckpointError,
    read_state,
    run_dir_for,
)


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Outcome of rendering an HTML + JSON report."""

    run_id: str
    html_path: Path
    json_path: Path


def run_report(
    *,
    run_id: str,
    config_path: Path,
    runs_base: Path,
) -> ReportResult:
    """Render the HTML + JSON report for a fully-analysed run."""
    run_dir = run_dir_for(run_id, runs_base)
    read_state(run_dir)  # validates state.json exists + parses

    # Best-effort config load: drives the per-example "tool match"
    # column. If the user moved the config or the report runs against
    # a foreign run dir, fall back to no tool detection.
    tool_evaluator_names: frozenset[str] = frozenset()
    try:
        cfg = load_config(config_path)
    except ConfigError:
        cfg = None
    if cfg is not None:
        tool_evaluator_names = frozenset(
            e.name
            for family in (
                cfg.evaluators.tool_selection,
                cfg.evaluators.tool_arguments,
                cfg.evaluators.tool_trace_structure,
            )
            for e in family
        )

    payload = build_report_payload(
        run_dir,
        tool_evaluator_names=tool_evaluator_names,
    )

    json_path = write_report_json(payload, run_dir)
    html_path = write_html(payload, run_dir)

    return ReportResult(run_id=run_id, html_path=html_path, json_path=json_path)


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
            help=f"Path to evalshift.yaml (default: ./{CONFIG_FILENAME}).",
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
    ] = Path(".evalshift") / "runs",
) -> None:
    """Render a single-file HTML report from a fully-analysed run."""
    console = Console()

    try:
        result = run_report(
            run_id=run_id,
            config_path=config_path,
            runs_base=runs_base,
        )
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        console.print(
            "Hint: run [cyan]evalshift evaluate[/cyan] then "
            "[cyan]evalshift analyze[/cyan] before [cyan]report[/cyan].",
        )
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] {result.html_path}")
    console.print(f"[green]✓[/green] {result.json_path}")

    if open_browser:
        webbrowser.open(result.html_path.resolve().as_uri())


__all__ = ["ReportResult", "report", "run_report"]
