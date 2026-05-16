"""Implementation of ``evalshift push``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.cli.commands.init import SUITE_FILENAME
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
        Path,
        typer.Option(
            "--suite",
            "-s",
            help=f"Path to the JSONL suite (default: ./{SUITE_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(SUITE_FILENAME),
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
        if bundle_path is not None:
            result = push_bundle(
                bundle_path,
                config_path=config_path,
                project=project,
                host=host,
                token=token,
                create_project=create_project,
                console=console,
            )
        else:
            assert run_id is not None
            result = push_local_run(
                run_id=run_id,
                config_path=config_path,
                suite_path=suite_path,
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
    console.print(result.view_url)


__all__ = ["push"]
