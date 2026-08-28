"""``evalshift runs`` — inspect and prune the local ``.evalshift/runs/`` history.

Every ``run`` / ``all`` invocation writes a fresh ``r_<date>_<suite>_<hex>`` directory, so run
history grows without bound unless pruned. ``runs clean`` applies the same retention rules the
orchestrator runs automatically after each completed run (:func:`evalshift.runner.checkpoint.prune_runs`),
but on demand and with explicit overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.config.loader import ConfigError, load_config
from evalshift.config.models import Retention
from evalshift.runner.checkpoint import prune_runs, resolve_max_runs

CONFIG_FILENAME = "evalshift.yaml"

runs_app = typer.Typer(help="Inspect and prune the local run history under .evalshift/runs/.")

_RunsBaseOption = Annotated[
    Path,
    typer.Option(
        "--runs-base",
        help="Base directory for run state (advanced).",
        hidden=True,
        file_okay=False,
    ),
]


def _load_retention(config_path: Path) -> Retention:
    """Return the configured retention policy, or the built-in defaults if config is absent/invalid.

    ``runs clean`` must work even without a valid ``evalshift.yaml`` (a user cleaning up disk
    shouldn't be blocked by an unrelated config error), so a load failure falls back to defaults.
    """
    try:
        return load_config(config_path).retention
    except (ConfigError, FileNotFoundError, OSError):
        return Retention()


@runs_app.command(name="clean")
def runs_clean(
    keep: Annotated[
        int | None,
        typer.Option(
            "--keep",
            help="Keep this many newest runs per suite (default: from config, else 20). "
            "0 disables count-based pruning.",
        ),
    ] = None,
    older_than: Annotated[
        int | None,
        typer.Option(
            "--older-than",
            help="Also delete runs older than this many days.",
        ),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="Only prune runs for this suite slug."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be deleted without deleting anything."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to evalshift.yaml.", dir_okay=False),
    ] = Path(CONFIG_FILENAME),
    runs_base: _RunsBaseOption = Path(".evalshift") / "runs",
) -> None:
    """Delete old run directories. Never touches an in-progress run."""
    console = Console()
    retention = _load_retention(config_path)

    # Precedence: explicit --keep flag > EVALSHIFT_MAX_RUNS env > config default.
    max_runs = max(keep, 0) if keep is not None else resolve_max_runs(retention.max_runs_per_suite)
    run_ttl_days = older_than if older_than is not None else retention.run_ttl_days

    if max_runs <= 0 and run_ttl_days is None:
        console.print(
            "[dim]nothing to do — pass [cyan]--keep N[/cyan] or "
            "[cyan]--older-than DAYS[/cyan] (count pruning is disabled).[/dim]"
        )
        return

    # Preview the candidates first (dry-run semantics), then confirm before deleting.
    candidates = prune_runs(
        runs_base,
        max_runs_per_suite=max_runs,
        run_ttl_days=run_ttl_days,
        suite=suite,
        dry_run=True,
    )

    if not candidates:
        console.print("[dim]nothing to clean — run history is already within limits.[/dim]")
        return

    if dry_run:
        console.print(f"[bold]Would delete {len(candidates)} run(s):[/bold]")
        for path in candidates:
            console.print(f"  [dim]{path.name}[/dim]")
        return

    if not yes:
        typer.confirm(f"Delete {len(candidates)} run director(ies)?", abort=True)

    removed = prune_runs(
        runs_base,
        max_runs_per_suite=max_runs,
        run_ttl_days=run_ttl_days,
        suite=suite,
    )
    console.print(f"[green]✓[/green] removed {len(removed)} run director(ies).")


__all__ = ["runs_app"]
