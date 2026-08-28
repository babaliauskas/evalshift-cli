"""Implementation of ``evalshift push``."""

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
from evalshift.hosted.push import PushError, push_bundle, push_local_run


def push(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Local run id to push. Omit when using --bundle."),
    ] = None,
    bundle_path: Annotated[
        Path | None,
        typer.Option("--bundle", help="Path to a prebuilt run_bundle.json.gz."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", help="Hosted project slug in org/project form."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Hosted API base URL."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Hosted API token."),
    ] = None,
    create_project: Annotated[
        bool,
        typer.Option(
            "--create-project/--no-create-project",
            help="Auto-create the project when the org is visible and permissions allow it.",
        ),
    ] = True,
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
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state (advanced).", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Push a local run bundle to hosted EvalShift."""
    console = Console()
    if bundle_path is None and run_id is None:
        console.print("[red]✗[/red] pass a run id or --bundle PATH")
        raise typer.Exit(code=1)
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
        if bundle_path is not None:
            result = push_bundle(
                bundle_path,
                config_path=config_path,
                project=project,
                host=host,
                token=token,
                create_project=create_project,
                console=console,
                runs_base=runs_base,
            )
        else:
            assert run_id is not None
            result = push_local_run(
                run_id=run_id,
                config_path=config_path,
                suite_path=resolved_suite_path,
                suite_name=suite_name,
                runs_base=runs_base,
                project=project,
                host=host,
                token=token,
                create_project=create_project,
                console=console,
            )
    except PushError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    # soft_wrap: this line is the command's machine-readable output — CI pipes
    # `push` and reads the URL back off stdout. Rich folds at 80 columns when it
    # cannot measure a terminal, which is exactly the piped case, and a hosted run
    # URL runs past that, so an unguarded print emits half a URL that still looks
    # like one.
    console.print(result.view_url, soft_wrap=True)


__all__ = ["push"]
