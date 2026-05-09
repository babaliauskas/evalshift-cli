"""Implementation of ``evalshift evaluate <run-id>``.

Loads ``raw.jsonl`` from a completed run, pairs source and target calls
by ``(prompt_id, example_id)``, runs every configured evaluator, and
appends one :class:`EvalRecord` per (pair x evaluator) to
``scores.jsonl``.

Failed source or target calls (those with ``error != None`` in
``raw.jsonl``) are recorded with ``error="upstream call failed"`` and a
neutral 0.5/0.5 score so the analysis layer can decide how to handle
them rather than silently skipping them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.config.loader import ConfigError, load_config
from evalshift.config.models import EvalShiftConfig
from evalshift.evaluators.base import EvalRecord, Evaluator, EvaluatorError, PairedScore
from evalshift.evaluators.llm_judge import PairwiseJudgeEvaluator
from evalshift.evaluators.semantic import CosineSimilarityEvaluator
from evalshift.evaluators.structural import (
    JsonSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from evalshift.evaluators.tool_arguments import ToolArgumentsEvaluator
from evalshift.evaluators.tool_selection import ToolSelectionEvaluator
from evalshift.evaluators.tool_trace_structure import ToolTraceStructureEvaluator
from evalshift.runner.checkpoint import (
    CheckpointError,
    iter_calls,
    read_state,
    run_dir_for,
)
from evalshift.runner.models import Call
from evalshift.suite.loader import SuiteError, load_jsonl
from evalshift.suite.models import Suite, SuiteExample

SCORES_FILENAME: str = "scores.jsonl"


@dataclass(frozen=True, slots=True)
class _PairedCalls:
    """Source + target calls for one (prompt, example) pair."""

    prompt_id: str
    example_id: str
    source: Call
    target: Call


def evaluate(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to evaluate (e.g. r_20260601_abc123)."),
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
    """Score every (source, target) pair from a completed run."""
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

    if state.status != "completed":
        console.print(
            f"[yellow]⚠[/yellow] run is {state.status!r}; results may be incomplete.",
        )

    project_root = config_path.resolve().parent
    try:
        evaluators = _build_evaluators(cfg, project_root)
    except EvaluatorError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not evaluators:
        console.print(
            "[red]✗[/red] no evaluators configured. Add at least one entry under "
            "[bold]evaluators:[/bold] in evalshift.yaml.",
        )
        raise typer.Exit(code=1)

    pairs = _pair_calls(run_dir)
    if not pairs:
        console.print("[red]✗[/red] no (source, target) pairs found in raw.jsonl")
        raise typer.Exit(code=1)

    # Load the suite so tool evaluators (which consume ``SuiteExample``)
    # can find the row matching each pair. v0.1 evaluators don't need it
    # but loading it is cheap and makes the dispatch uniform.
    examples_by_id = _load_examples_by_id(state.suite_path)

    records = asyncio.run(
        _score_all(console, evaluators, pairs, run_id, examples_by_id),
    )

    output_path = run_dir / SCORES_FILENAME
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")

    console.print(
        f"[green]✓[/green] wrote {len(records)} eval records to {output_path}",
    )
    console.print(
        f"[bold]Next:[/bold] [cyan]evalshift analyze {run_id}[/cyan]",
    )


def _build_evaluators(cfg: EvalShiftConfig, project_root: Path) -> list[Evaluator]:
    # Tool evaluators (v0.2) implement ``score_pair`` rather than the
    # text-evaluator ``score`` method, so they don't satisfy the
    # :class:`Evaluator` Protocol structurally. We treat them as
    # duck-typed ``Evaluator``-likes — the dispatch in ``_score_one``
    # checks for ``score_pair`` before calling.
    out: list[Evaluator] = []
    for s in cfg.evaluators.structural:
        if s.type == "json_schema":
            assert s.schema_path is not None  # pydantic invariant
            schema_path = Path(s.schema_path)
            if not schema_path.is_absolute():
                schema_path = project_root / schema_path
            out.append(JsonSchemaEvaluator(schema_path=schema_path))
        elif s.type == "regex":
            assert s.pattern is not None
            out.append(RegexEvaluator(pattern=s.pattern))
        else:  # length
            out.append(
                LengthEvaluator(min_chars=s.min_chars, max_chars=s.max_chars),
            )

    if cfg.evaluators.semantic is not None:
        out.append(
            CosineSimilarityEvaluator(
                embedding_model=cfg.evaluators.semantic.embedding_model,
            ),
        )

    for j in cfg.evaluators.llm_judge:
        out.append(
            PairwiseJudgeEvaluator(
                criterion_name=j.criterion_name,
                criterion_prompt=j.criterion_prompt,
                judge_model=j.judge_model,
            ),
        )

    # v0.2 — tool-call evaluators. Each implements ``score_pair`` rather
    # than ``score`` because they consume the (source, target) ToolTrace
    # pair directly.
    for ts in cfg.evaluators.tool_selection:
        out.append(ToolSelectionEvaluator(ts))  # type: ignore[arg-type]
    for ta in cfg.evaluators.tool_arguments:
        out.append(ToolArgumentsEvaluator(ta))  # type: ignore[arg-type]
    for tts in cfg.evaluators.tool_trace_structure:
        out.append(ToolTraceStructureEvaluator(tts))  # type: ignore[arg-type]

    return out


def _is_tool_evaluator(evaluator: Evaluator) -> bool:
    """True iff ``evaluator`` consumes :class:`ToolTrace` pairs (v0.2)."""
    return hasattr(evaluator, "score_pair")


def _pair_calls(run_dir: Path) -> list[_PairedCalls]:
    by_key: dict[tuple[str, str], dict[str, Call]] = {}
    for call in iter_calls(run_dir):
        key = (call.prompt_id, call.example_id)
        by_key.setdefault(key, {})[call.role] = call
    pairs: list[_PairedCalls] = []
    for (prompt_id, example_id), sides in sorted(by_key.items()):
        if "source" in sides and "target" in sides:
            pairs.append(
                _PairedCalls(
                    prompt_id=prompt_id,
                    example_id=example_id,
                    source=sides["source"],
                    target=sides["target"],
                ),
            )
    return pairs


def _load_examples_by_id(suite_path: str) -> dict[str, SuiteExample]:
    """Load the suite and index examples by id; tolerate failures."""
    try:
        suite: Suite = load_jsonl(suite_path)
    except SuiteError, FileNotFoundError, OSError:
        return {}
    return {ex.id: ex for ex in suite.examples}


async def _score_all(
    console: Console,
    evaluators: list[Evaluator],
    pairs: list[_PairedCalls],
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    total = len(pairs) * len(evaluators)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task(
        description=f"[cyan]evaluate {run_id}[/cyan]",
        total=total,
    )

    with progress:
        for pair in pairs:
            for evaluator in evaluators:
                record = await _score_one(evaluator, pair, run_id, examples_by_id)
                records.append(record)
                progress.advance(task_id)
    return records


async def _score_one(
    evaluator: Evaluator,
    pair: _PairedCalls,
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
) -> EvalRecord:
    upstream_failed = pair.source.error is not None or pair.target.error is not None
    if upstream_failed:
        upstream = "; ".join(x for x in (pair.source.error, pair.target.error) if x is not None)
        return EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            source_score=0.5,
            target_score=0.5,
            delta=0.0,
            error=f"upstream call failed: {upstream}",
        )

    # v0.2 — tool evaluators consume the (source, target) ToolTrace pair.
    # Plain prompts have ``call.trace is None`` and we skip cleanly with
    # a neutral record so the analysis layer doesn't see noise.
    if _is_tool_evaluator(evaluator):
        return await _score_one_tool(evaluator, pair, run_id, examples_by_id)

    try:
        score: PairedScore = await evaluator.score(
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            input_vars={},
            source_output=pair.source.text,
            target_output=pair.target.text,
        )
    except Exception as exc:
        return EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            source_score=0.5,
            target_score=0.5,
            delta=0.0,
            error=f"evaluator error: {exc}",
        )

    return EvalRecord(
        run_id=run_id,
        prompt_id=pair.prompt_id,
        example_id=pair.example_id,
        evaluator_name=evaluator.name,
        source_score=score.source_score,
        target_score=score.target_score,
        delta=score.delta,
        explanation=score.explanation,
        metadata=score.metadata,
    )


async def _score_one_tool(
    evaluator: Evaluator,
    pair: _PairedCalls,
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
) -> EvalRecord:
    """Dispatch a v0.2 tool evaluator. Skips plain (text) calls cleanly."""
    if pair.source.trace is None or pair.target.trace is None:
        # No tool trace on either side → nothing for this evaluator to do.
        return EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            source_score=1.0,
            target_score=1.0,
            delta=0.0,
            metadata={"skipped": "no tool trace on this pair"},
        )
    example = examples_by_id.get(pair.example_id) or SuiteExample(id=pair.example_id)
    try:
        record = await evaluator.score_pair(  # type: ignore[attr-defined]
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example=example,
            source_trace=pair.source.trace,
            target_trace=pair.target.trace,
        )
    except Exception as exc:
        return EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            source_score=0.5,
            target_score=0.5,
            delta=0.0,
            error=f"evaluator error: {exc}",
        )
    return record  # type: ignore[no-any-return]


__all__ = ["SCORES_FILENAME", "evaluate"]
