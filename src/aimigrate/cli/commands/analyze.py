"""Implementation of ``aimigrate analyze <run-id>``.

Loads ``scores.jsonl`` from a completed evaluation, groups records into
slices, runs the statistics pipeline, and writes ``analysis.json`` to
the run directory. The Phase 7 report renderer reads that file.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from aimigrate.analysis.slicing import (
    SliceAggregate,
    aggregates,
    build_slices,
)
from aimigrate.analysis.statistics import ComparisonResult, analyze
from aimigrate.cli.commands.doctor import CONFIG_FILENAME
from aimigrate.cli.commands.evaluate import SCORES_FILENAME
from aimigrate.config.loader import ConfigError, load_config
from aimigrate.evaluators.base import EvalRecord
from aimigrate.runner.checkpoint import (
    CheckpointError,
    read_state,
    run_dir_for,
)
from aimigrate.suite.loader import SuiteError, load_jsonl

ANALYSIS_FILENAME: str = "analysis.json"


def analyze_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to analyze."),
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
    """Run paired statistical tests over scored evaluations."""
    console = Console()

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    run_dir = run_dir_for(run_id, runs_base)
    try:
        state = read_state(run_dir)
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    scores_path = run_dir / SCORES_FILENAME
    if not scores_path.exists():
        console.print(
            f"[red]✗[/red] no {SCORES_FILENAME} in {run_dir}; "
            f"run [cyan]aimigrate evaluate {run_id}[/cyan] first.",
        )
        raise typer.Exit(code=1)

    records = _load_score_records(scores_path)
    if not records:
        console.print(f"[red]✗[/red] {scores_path} is empty")
        raise typer.Exit(code=1)

    # Load suite for tag lookups when slicing.
    try:
        suite = load_jsonl(state.suite_path)
    except SuiteError:
        # Fall back to no-suite slicing (only the implicit "all" slice).
        from aimigrate.suite.models import Suite

        suite = Suite()

    sliced = build_slices(records=records, suite=suite)
    comparisons = analyze(sliced_by_slice=sliced)

    # Build the JSON artefact.
    aggregates_by_slice = {
        name: asdict(_aggregate_for(name, sliced.get(name, []))) for name in sliced
    }
    payload: dict[str, Any] = {
        "run_id": run_id,
        "models": {
            "source": state.models.source,
            "target": state.models.target,
        },
        "n_examples": len(suite),
        "n_records": len(records),
        "slices": list(sliced.keys()),
        "aggregates": aggregates_by_slice,
        "comparisons": [_comparison_to_dict(c) for c in comparisons],
    }
    out_path = run_dir / ANALYSIS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(console, comparisons, out_path, run_id)
    _ = cfg  # kept for future config-driven slicing options.


def _load_score_records(path: Path) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            out.append(EvalRecord.model_validate_json(text))
    return out


def _aggregate_for(name: str, sliced: list[Any]) -> SliceAggregate:
    return aggregates(sliced, name)


def _comparison_to_dict(c: ComparisonResult) -> dict[str, Any]:
    return {
        "prompt_id": c.prompt_id,
        "evaluator_name": c.evaluator_name,
        "slice_name": c.slice_name,
        "n": c.n,
        "test": c.test,
        "statistic": c.statistic,
        "p_value": c.p_value,
        "p_value_corrected": c.p_value_corrected,
        "effect_size": c.effect_size,
        "effect_size_ci_low": c.effect_size_ci_low,
        "effect_size_ci_high": c.effect_size_ci_high,
        "delta_mean": c.delta_mean,
        "severity": c.severity,
        "notes": c.notes,
    }


_SEVERITY_GLYPHS: dict[str, tuple[str, str]] = {
    "critical": ("✗", "red"),
    "high": ("✗", "red"),
    "medium": ("⚠", "yellow"),
    "low": ("⚠", "yellow"),
    "improved": ("↑", "green"),
    "none": ("✓", "green"),
    "insufficient": ("?", "dim"),
}


def _print_summary(
    console: Console,
    comparisons: list[ComparisonResult],
    out_path: Path,
    run_id: str,
) -> None:
    table = Table(title=f"analysis · {run_id}", show_lines=False)
    table.add_column("severity", no_wrap=True)
    table.add_column("prompt")
    table.add_column("evaluator")
    table.add_column("slice")
    table.add_column("n", justify="right")
    table.add_column("Δmean", justify="right")
    table.add_column("|d|", justify="right")
    table.add_column("p_corr", justify="right")
    for c in comparisons:
        # Skip the implicit all-slice if there's a more specific row;
        # but for the MVP just show every row to keep things simple.
        glyph, style = _SEVERITY_GLYPHS.get(c.severity, ("?", "dim"))
        table.add_row(
            f"[{style}]{glyph} {c.severity}[/{style}]",
            c.prompt_id,
            c.evaluator_name,
            c.slice_name,
            str(c.n),
            f"{c.delta_mean:+.3f}",
            f"{abs(c.effect_size):.2f}",
            f"{c.p_value_corrected:.3f}",
        )
    console.print(table)
    console.print(
        f"[green]✓[/green] wrote analysis to [bold]{out_path}[/bold]",
    )
    console.print(
        f"[bold]Next:[/bold] [cyan]aimigrate report {run_id}[/cyan]   [dim](Phase 7)[/dim]",
    )


__all__ = ["ANALYSIS_FILENAME", "analyze_command"]
