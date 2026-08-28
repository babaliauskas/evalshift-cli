"""Commands for importing externally recorded agent traces."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.runner.checkpoint import CheckpointError, iter_calls, read_state, run_dir_for
from evalshift.traces.loader import (
    TRACES_FILENAME,
    TraceLoadError,
    index_traces,
    load_traces_jsonl,
    pairs_for_prompt_examples,
    write_traces_jsonl,
)
from evalshift.traces.models import AgentTrace, TraceRole

traces_app = typer.Typer(help="Import bring-your-own-agent traces.")


@traces_app.command(name="import")
def import_traces(
    run_id: Annotated[str, typer.Argument(help="Run id to attach traces to.")],
    source: Annotated[
        Path,
        typer.Option("--source", help="JSONL file with source-side traces."),
    ],
    target: Annotated[
        Path,
        typer.Option("--target", help="JSONL file with target-side traces."),
    ],
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Fail when any completed run pair lacks a trace pair."),
    ] = False,
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state.", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Import source and target trace JSONL files into a completed run."""
    console = Console()
    try:
        run_dir = run_dir_for(run_id, runs_base)
        read_state(run_dir)
        source_traces = load_traces_jsonl(source)
        target_traces = load_traces_jsonl(target)
        _validate_imported_traces(
            run_id=run_id,
            source_traces=source_traces,
            target_traces=target_traces,
            prompt_examples=_known_prompt_examples(run_dir),
            strict=strict,
        )
    except (CheckpointError, TraceLoadError, ValueError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    traces = [*source_traces, *target_traces]
    output_path = run_dir / TRACES_FILENAME
    write_traces_jsonl(output_path, traces)
    pairs = pairs_for_prompt_examples(traces, prompt_examples=_known_prompt_examples(run_dir))
    console.print(f"[green]✓[/green] Imported traces for run {run_id}")
    console.print(f"source: {len(source_traces)} traces")
    console.print(f"target: {len(target_traces)} traces")
    console.print(f"missing pairs: {len(_missing_pairs(traces, _known_prompt_examples(run_dir)))}")
    console.print(f"artifact: {output_path}")
    console.print(f"paired traces: {len(pairs)}")


def _known_prompt_examples(run_dir: Path) -> list[tuple[str, str]]:
    """Return completed source/target prompt/example pairs from raw.jsonl."""
    by_key: dict[tuple[str, str], set[str]] = {}
    for call in iter_calls(run_dir):
        by_key.setdefault((call.prompt_id, call.example_id), set()).add(call.role)
    return sorted(key for key, roles in by_key.items() if {"source", "target"} <= roles)


def _validate_imported_traces(
    *,
    run_id: str,
    source_traces: list[AgentTrace],
    target_traces: list[AgentTrace],
    prompt_examples: list[tuple[str, str]],
    strict: bool,
) -> None:
    _validate_side(run_id=run_id, traces=source_traces, expected_role="source")
    _validate_side(run_id=run_id, traces=target_traces, expected_role="target")
    traces = [*source_traces, *target_traces]
    index_traces(traces)
    known = set(prompt_examples)
    for trace in traces:
        key = (trace.prompt_id, trace.example_id)
        if key not in known:
            raise ValueError(
                f"unknown prompt/example in trace: prompt={trace.prompt_id!r} "
                f"example={trace.example_id!r}",
            )
    missing = _missing_pairs(traces, prompt_examples)
    if strict and missing:
        formatted = ", ".join(f"{prompt}/{example}" for prompt, example in missing)
        raise ValueError(f"missing trace pairs: {formatted}")


def _validate_side(*, run_id: str, traces: list[AgentTrace], expected_role: TraceRole) -> None:
    for trace in traces:
        if trace.run_id != run_id:
            raise ValueError(f"trace run_id {trace.run_id!r} does not match {run_id!r}")
        if trace.role != expected_role:
            raise ValueError(
                f"{expected_role} file contains {trace.role!r} trace for {trace.example_id!r}",
            )


def _missing_pairs(
    traces: list[AgentTrace],
    prompt_examples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    indexed = index_traces(traces)
    out: list[tuple[str, str]] = []
    for prompt_id, example_id in prompt_examples:
        if (prompt_id, example_id, "source") not in indexed or (
            prompt_id,
            example_id,
            "target",
        ) not in indexed:
            out.append((prompt_id, example_id))
    return out


__all__ = ["traces_app"]
