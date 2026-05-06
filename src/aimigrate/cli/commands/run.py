"""Implementation of ``aimigrate run`` — the headline command.

Loads config + suite, dispatches to the async orchestrator, and prints
a human-friendly summary on completion. Friendly errors via Rich for
every failure mode the orchestrator can raise.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from aimigrate.cli.commands.doctor import CONFIG_FILENAME
from aimigrate.cli.commands.init import SUITE_FILENAME
from aimigrate.config.loader import ConfigError, load_config
from aimigrate.models.registry import UnknownModelError
from aimigrate.parsers.base import PromptParseError
from aimigrate.runner.orchestrator import (
    RunAborted,
    RunResult,
    run_orchestrator,
)
from aimigrate.suite.loader import SuiteError, load_jsonl
from aimigrate.utils.templating import SuiteCompatibilityError


def run(
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            "-f",
            help="Source model id or alias. Overrides config defaults.source_model.",
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--to",
            "-t",
            help="Target model id or alias. Overrides config defaults.target_model.",
        ),
    ] = None,
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
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Continue the most recent in-progress run.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Run paired evaluation on two models against your golden suite."""
    console = Console()

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    source_model = source or cfg.defaults.source_model
    target_model = target or cfg.defaults.target_model
    if not source_model or not target_model:
        console.print(
            "[red]✗[/red] missing model selection: pass [bold]--from[/bold] / "
            "[bold]--to[/bold] or set [bold]defaults.source_model[/bold] / "
            "[bold]defaults.target_model[/bold] in aimigrate.yaml.",
        )
        raise typer.Exit(code=1)

    try:
        suite = load_jsonl(suite_path)
    except SuiteError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    try:
        result = asyncio.run(
            run_orchestrator(
                config=cfg,
                config_path=config_path,
                suite=suite,
                suite_path=suite_path,
                source_model=source_model,
                target_model=target_model,
                resume=resume,
                yes=yes,
                console=console,
            ),
        )
    except PromptParseError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except SuiteCompatibilityError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except RunAborted as exc:
        console.print(f"[yellow]⚠[/yellow] aborted: {exc}")
        raise typer.Exit(code=1) from exc
    except UnknownModelError as exc:
        # Should be unreachable now that resolve_model is permissive, but
        # keep this as a defensive net so any future strict-path leak
        # surfaces as a friendly error instead of a traceback.
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_summary(console, result)


def _print_summary(console: Console, result: RunResult) -> None:
    body = (
        f"[bold]{result.run_id}[/bold]\n\n"
        f"[dim]calls:[/dim]   "
        f"[green]{result.completed_calls}[/green]/{result.total_calls} completed  "
        f"({result.cached_calls} cached, "
        f"{result.live_calls} live, "
        f"[red]{result.failed_calls} failed[/red])\n"
        f"[dim]cost:[/dim]    ${result.total_cost_usd:.4f}\n"
        f"[dim]outputs:[/dim] {result.run_dir / 'raw.jsonl'}\n\n"
        f"[bold]Next:[/bold] [cyan]aimigrate evaluate {result.run_id}[/cyan]"
    )
    console.print(
        Panel(
            body,
            border_style="green",
            title="aimigrate run",
            title_align="left",
        ),
    )


__all__ = ["run"]
