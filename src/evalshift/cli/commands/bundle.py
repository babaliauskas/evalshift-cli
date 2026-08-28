"""Implementation of ``evalshift bundle``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands._suites import (
    SUITE_FILENAME,
    UnknownSuiteNameError,
    resolve_suite_override,
)
from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.config.loader import ConfigError
from evalshift.hosted.bundle import BundleError, build_bundle


def bundle(
    run_id: Annotated[str, typer.Argument(help="Local run id to package.")],
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
    suite_path: Annotated[
        Path | None,
        typer.Option(
            "--suite",
            "-s",
            help=(
                "Path to the JSONL suite. Defaults to the suite recorded in the "
                f"run's state (typically ./{SUITE_FILENAME})."
            ),
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    suite_name: Annotated[
        str | None,
        typer.Option(
            "--suite-name",
            help="Named suite from evalshift.yaml suites: (e.g. a promoted capture).",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for run_bundle.json.gz."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", help="Hosted project slug in org/project form."),
    ] = None,
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state (advanced).", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Build a hosted run bundle from local artifacts."""
    console = Console()
    try:
        resolved_suite_path = resolve_suite_override(
            suite_path=suite_path,
            suite_name=suite_name,
            config_path=config_path,
        )
    except (ConfigError, UnknownSuiteNameError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        result = build_bundle(
            run_id,
            config_path=config_path,
            suite_path=resolved_suite_path,
            suite_name=suite_name,
            runs_base=runs_base,
            output=output,
            project=project,
        )
    except BundleError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]✓[/green] {result.path}")


__all__ = ["bundle"]
