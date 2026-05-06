"""The async orchestrator that drives ``aimigrate run``.

This is the place where every Phase 0-3 piece comes together:

1. Load + validate config (Phase 1.2) and suite (Phase 2.2).
2. Pick the right parser per prompt (Phase 2.3) and produce
   :class:`PromptTemplate` objects.
3. Pre-flight check that every example is compatible with every prompt
   (Phase 2.4) so we don't burn money on a misconfigured run.
4. Estimate total cost (Phase 3.5); confirm with the user if it's over
   the threshold (skip with ``--yes``).
5. Open the local cache (Phase 3.3) and the model client (Phase 3.4).
6. Build the work list (one :class:`WorkItem` per
   ``(prompt x example x {source, target})``).
7. Skip work items already recorded in ``raw.jsonl`` (resume support).
8. Process the rest under a concurrency semaphore, writing each
   completed :class:`Call` to disk and checkpointing the run state
   every ``CHECKPOINT_EVERY`` completions.
9. On clean exit mark the state ``"completed"``.

Live UI is via ``rich.progress``. The ``run_id`` and run directory
location are returned to the caller (the CLI) so it can print a clear
"Next: aimigrate evaluate <run-id>" footer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from aimigrate.cache.store import CacheStore, cache_key
from aimigrate.config.models import AIMigrateConfig
from aimigrate.evaluators.tool_loader import load_tools
from aimigrate.evaluators.tool_models import ToolSpec
from aimigrate.models.client import ModelClient, ModelClientError
from aimigrate.models.registry import resolve_model
from aimigrate.parsers.base import PromptParseError, PromptTemplate
from aimigrate.parsers.manual import ManualParser
from aimigrate.parsers.python_string import PythonStringParser
from aimigrate.runner.checkpoint import (
    append_call,
    completed_call_keys,
    compute_config_hash,
    find_latest_in_progress,
    generate_run_id,
    run_dir_for,
    touch_checkpoint,
    validate_resume,
    write_state,
)
from aimigrate.runner.models import Call, CallRole, RunModels, RunState
from aimigrate.suite.models import Suite, SuiteExample
from aimigrate.utils.cost import CostEstimate, estimate_run_cost
from aimigrate.utils.templating import (
    SuiteCompatibilityError,
    render,
    validate_suite_against_prompts,
)

CHECKPOINT_EVERY: int = 50
COST_CONFIRM_THRESHOLD_USD: float = 10.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One unit of work: a single LLM call to make.

    When ``tools`` is non-empty, the orchestrator dispatches to
    ``ModelClient.complete_with_tools`` and the resulting :class:`Call`
    carries a populated ``trace``. Otherwise the standard text-only
    ``ModelClient.complete`` path runs.
    """

    prompt: PromptTemplate
    example: SuiteExample
    role: CallRole
    model_id: str  # canonical id, post-alias resolution
    tools: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary returned from :func:`run_orchestrator`."""

    run_id: str
    run_dir: Path
    total_calls: int
    completed_calls: int
    cached_calls: int
    live_calls: int
    failed_calls: int
    total_cost_usd: float


class RunAborted(Exception):  # noqa: N818 — keeps the existing public name.
    """Raised when the user declines the cost prompt or a precondition fails."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_orchestrator(
    *,
    config: AIMigrateConfig,
    config_path: Path,
    suite: Suite,
    suite_path: Path,
    source_model: str,
    target_model: str,
    runs_base: Path | None = None,
    resume: bool = False,
    yes: bool = False,
    console: Console | None = None,
    client: ModelClient | None = None,
    cache: CacheStore | None = None,
) -> RunResult:
    """Execute (or resume) a run end-to-end.

    Args:
        config: Loaded and validated config.
        config_path: Path the config came from (used for prompt-parser
            relative-path resolution).
        suite: Loaded suite.
        suite_path: Path the suite came from. Stored verbatim in
            ``state.json`` so reports can quote the source.
        source_model: Canonical id or alias for the "before" side.
        target_model: Canonical id or alias for the "after" side.
        runs_base: Override for ``.aimigrate/runs/``. Tests pass a
            tmp path; production leaves it default.
        resume: If True, look for the latest in-progress run and pick
            up where it left off. The config + suite must match.
        yes: Skip the cost-confirmation prompt regardless of estimate.
        console: Rich console (created if not supplied).
        client: :class:`ModelClient` to use; mostly here for tests.
        cache: :class:`CacheStore` to use; mostly here for tests.
    """
    cons = console or Console()

    project_root = config_path.resolve().parent
    templates = _parse_prompts(config, project_root)

    # Pre-flight compatibility — abort early if any example is missing a
    # template variable. This raises the friendly Phase 2 error.
    validate_suite_against_prompts(suite, templates)

    # v0.2 — load any per-prompt tool specs so agent prompts dispatch
    # to ``complete_with_tools``. Plain prompts get an empty tuple.
    tools_by_prompt = _load_tools_per_prompt(config, project_root)

    canonical_source = resolve_model(source_model).id
    canonical_target = resolve_model(target_model).id

    # Run setup: either fresh or resume.
    config_hash = compute_config_hash(config, str(suite_path))
    run_dir, state, completed_keys = _setup_run(
        config=config,
        config_hash=config_hash,
        suite=suite,
        suite_path=suite_path,
        templates=templates,
        canonical_source=canonical_source,
        canonical_target=canonical_target,
        runs_base=runs_base,
        resume=resume,
    )

    # Build the full work list (every required call) and filter out
    # whatever the resume scan already saw.
    work = _build_work_list(
        templates=templates,
        suite=suite,
        canonical_source=canonical_source,
        canonical_target=canonical_target,
        tools_by_prompt=tools_by_prompt,
    )
    pending = [w for w in work if (w.prompt.id, w.example.id, w.role) not in completed_keys]

    # Cost estimate + confirmation. We only show the prompt for *new*
    # runs above the threshold; a resume is implicitly already approved.
    if not resume:
        estimate = _estimate(
            templates=templates,
            suite=suite,
            canonical_source=canonical_source,
            canonical_target=canonical_target,
        )
        if (
            not yes
            and estimate.estimated_usd > COST_CONFIRM_THRESHOLD_USD
            and not _confirm_cost(cons, estimate, len(work))
        ):
            state = state.model_copy(update={"status": "failed"})
            write_state(run_dir, state)
            raise RunAborted("user declined the cost prompt")

    # Real work.
    cache_owned = cache is None
    cache_inst = cache or await CacheStore.open()
    client_inst = client or ModelClient()

    try:
        result = await _process_work(
            cons=cons,
            run_dir=run_dir,
            state=state,
            client=client_inst,
            cache=cache_inst,
            templates=templates,
            pending=pending,
            already_done=len(work) - len(pending),
            total=len(work),
            concurrency=config.defaults.concurrency,
            cache_enabled=config.defaults.cache,
        )
    finally:
        if cache_owned:
            await cache_inst.close()

    return result


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _parse_prompts(config: AIMigrateConfig, project_root: Path) -> list[PromptTemplate]:
    """Pick the right parser per prompt; raise PromptParseError on first failure."""
    out: list[PromptTemplate] = []
    for prompt in config.prompts:
        parser = ManualParser() if prompt.detection == "manual" else PythonStringParser()
        out.append(parser.parse(prompt, project_root))
    return out


def _load_tools_per_prompt(
    config: AIMigrateConfig,
    project_root: Path,
) -> dict[str, tuple[ToolSpec, ...]]:
    """Return ``{prompt_id: tuple[ToolSpec, ...]}`` for every agent-style prompt.

    Prompts without ``tools_path`` map to an empty tuple. Errors raised
    by the loader are surfaced unchanged so the CLI can render them.
    """
    out: dict[str, tuple[ToolSpec, ...]] = {}
    for prompt in config.prompts:
        if not prompt.tools_path:
            out[prompt.id] = ()
            continue
        path = Path(prompt.tools_path)
        if not path.is_absolute():
            path = project_root / path
        out[prompt.id] = tuple(load_tools(path))
    return out


def _setup_run(
    *,
    config: AIMigrateConfig,
    config_hash: str,
    suite: Suite,
    suite_path: Path,
    templates: list[PromptTemplate],
    canonical_source: str,
    canonical_target: str,
    runs_base: Path | None,
    resume: bool,
) -> tuple[Path, RunState, set[tuple[str, str, str]]]:
    """Create a fresh run directory or resume the latest in-progress one."""
    if resume:
        existing = find_latest_in_progress(runs_base)
        if existing is None:
            raise RunAborted(
                "no in-progress run to resume; run without --resume to start fresh",
            )
        state = validate_resume(existing, expected_hash=config_hash)
        return existing, state, completed_call_keys(existing)

    run_id = generate_run_id()
    run_dir = run_dir_for(run_id, runs_base)
    total = len(templates) * len(suite) * 2  # source + target
    state = RunState(
        run_id=run_id,
        config_hash=config_hash,
        started_at=datetime.now(UTC),
        models=RunModels(source=canonical_source, target=canonical_target),
        prompt_ids=[t.id for t in templates],
        suite_path=str(suite_path),
        total_evaluations=total,
    )
    write_state(run_dir, state)
    return run_dir, state, set()


def _build_work_list(
    *,
    templates: list[PromptTemplate],
    suite: Suite,
    canonical_source: str,
    canonical_target: str,
    tools_by_prompt: dict[str, tuple[ToolSpec, ...]] | None = None,
) -> list[WorkItem]:
    tools_by_prompt = tools_by_prompt or {}
    work: list[WorkItem] = []
    for tmpl in templates:
        tools = tools_by_prompt.get(tmpl.id, ())
        for example in suite.examples:
            work.append(
                WorkItem(
                    prompt=tmpl,
                    example=example,
                    role="source",
                    model_id=canonical_source,
                    tools=tools,
                ),
            )
            work.append(
                WorkItem(
                    prompt=tmpl,
                    example=example,
                    role="target",
                    model_id=canonical_target,
                    tools=tools,
                ),
            )
    return work


def _estimate(
    *,
    templates: list[PromptTemplate],
    suite: Suite,
    canonical_source: str,
    canonical_target: str,
) -> CostEstimate:
    # Pick the longest-looking template as the representative for the
    # estimate so we err on the side of an over-estimate.
    representative = max((t.content for t in templates), key=len, default="")
    return estimate_run_cost(
        template=representative,
        examples=[e.inputs for e in suite.examples],
        n_prompts=len(templates),
        models=[canonical_source, canonical_target],
    )


def _confirm_cost(
    console: Console,
    estimate: CostEstimate,
    total_calls: int,
) -> bool:
    console.print(
        f"This run will make [bold]{total_calls}[/bold] LLM calls "
        f"(estimated cost [bold]${estimate.estimated_usd:.2f}[/bold]).",
    )
    console.print(
        "Continue? [Y/n] ",
        end="",
    )
    answer = (input().strip() or "y").lower()
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# The async work loop
# ---------------------------------------------------------------------------


async def _process_work(
    *,
    cons: Console,
    run_dir: Path,
    state: RunState,
    client: ModelClient,
    cache: CacheStore,
    templates: list[PromptTemplate],
    pending: list[WorkItem],
    already_done: int,
    total: int,
    concurrency: int,
    cache_enabled: bool,
) -> RunResult:
    """Process every pending work item under the concurrency semaphore."""
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = already_done
    cached_count = 0
    live_count = 0
    failed_count = 0
    total_cost = 0.0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("cost ${task.fields[cost]:.2f}"),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=cons,
        transient=False,
    )
    task_id = progress.add_task(
        description=f"[cyan]{state.run_id}[/cyan]",
        total=total,
        completed=already_done,
        cost=0.0,
    )

    template_by_id: dict[str, PromptTemplate] = {t.id: t for t in templates}

    async def _do_one(item: WorkItem) -> None:
        nonlocal completed, cached_count, live_count, failed_count, total_cost
        async with sem:
            tmpl = template_by_id[item.prompt.id]
            prompt_text = render(tmpl.content, item.example.inputs)
            call = await _execute(
                client=client,
                cache=cache,
                run_id=state.run_id,
                item=item,
                prompt_text=prompt_text,
                cache_enabled=cache_enabled,
            )

            async with write_lock:
                append_call(run_dir, call)
                completed += 1
                if call.cached:
                    cached_count += 1
                elif call.error is None:
                    live_count += 1
                if call.error is not None:
                    failed_count += 1
                total_cost += call.cost_usd
                progress.update(task_id, advance=1, cost=total_cost)
                if completed % CHECKPOINT_EVERY == 0:
                    write_state(run_dir, touch_checkpoint(state, completed))

    with progress:
        await asyncio.gather(*(_do_one(item) for item in pending))

    # Final checkpoint + status flip.
    final_state = touch_checkpoint(state, completed).model_copy(
        update={"status": "completed"},
    )
    write_state(run_dir, final_state)

    return RunResult(
        run_id=state.run_id,
        run_dir=run_dir,
        total_calls=total,
        completed_calls=completed,
        cached_calls=cached_count,
        live_calls=live_count,
        failed_calls=failed_count,
        total_cost_usd=total_cost,
    )


async def _execute(
    *,
    client: ModelClient,
    cache: CacheStore,
    run_id: str,
    item: WorkItem,
    prompt_text: str,
    cache_enabled: bool,
) -> Call:
    """Cache-check → live call → record. Returns the constructed Call.

    For agent-style work items (``item.tools`` non-empty), dispatches to
    :meth:`ModelClient.complete_with_tools` and stores the parsed
    :class:`ToolTrace` on the resulting :class:`Call`. The local SQLite
    cache is intentionally bypassed for tool calls in v0.2 — caching
    serialised traces is a v0.3 polish.
    """
    meta = resolve_model(item.model_id)

    if item.tools:
        return await _execute_with_tools(
            client=client,
            run_id=run_id,
            item=item,
            prompt_text=prompt_text,
            canonical_id=meta.id,
        )

    key = cache_key(
        model_id=meta.id,
        prompt_text=prompt_text,
        inputs=item.example.inputs,
        temperature=meta.default_temperature,
        max_tokens=meta.default_max_tokens,
    )

    if cache_enabled:
        hit = await cache.get(key)
        if hit is not None:
            return Call(
                run_id=run_id,
                prompt_id=item.prompt.id,
                example_id=item.example.id,
                model_id=meta.id,
                role=item.role,
                text=hit.response_text,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                cost_usd=hit.cost_usd,
                latency_ms=hit.latency_ms,
                cached=True,
            )

    try:
        result = await client.complete(model=meta.id, prompt=prompt_text)
    except ModelClientError as exc:
        return Call(
            run_id=run_id,
            prompt_id=item.prompt.id,
            example_id=item.example.id,
            model_id=meta.id,
            role=item.role,
            error=str(exc),
        )

    call = Call(
        run_id=run_id,
        prompt_id=item.prompt.id,
        example_id=item.example.id,
        model_id=meta.id,
        role=item.role,
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )

    if cache_enabled:
        await cache.put(
            key,
            model_id=meta.id,
            prompt_text=prompt_text,
            inputs=item.example.inputs,
            response_text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )

    return call


async def _execute_with_tools(
    *,
    client: ModelClient,
    run_id: str,
    item: WorkItem,
    prompt_text: str,
    canonical_id: str,
) -> Call:
    """Tool-aware call path: dispatch + record the trace on the Call."""
    try:
        result = await client.complete_with_tools(
            model=canonical_id,
            prompt=prompt_text,
            tools=list(item.tools),
        )
    except ModelClientError as exc:
        return Call(
            run_id=run_id,
            prompt_id=item.prompt.id,
            example_id=item.example.id,
            model_id=canonical_id,
            role=item.role,
            error=str(exc),
        )
    return Call(
        run_id=run_id,
        prompt_id=item.prompt.id,
        example_id=item.example.id,
        model_id=canonical_id,
        role=item.role,
        text=result.trace.final_text or "",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        trace=result.trace,
    )


# Re-exported so the CLI can catch them with one import.
__all__ = [
    "CHECKPOINT_EVERY",
    "COST_CONFIRM_THRESHOLD_USD",
    "PromptParseError",
    "RunAborted",
    "RunResult",
    "SuiteCompatibilityError",
    "WorkItem",
    "run_orchestrator",
]
