"""The async orchestrator that drives ``evalshift run``.

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
"Next: evalshift evaluate <run-id>" footer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, MutableMapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from evalshift.cache.store import CacheStore, cache_key
from evalshift.captures.reader import CaptureError, capture_base, load_toolset
from evalshift.captures.toolset import fingerprint_tools
from evalshift.config.models import EvalShiftConfig
from evalshift.evaluators.tool_models import ToolSpec
from evalshift.models.capabilities import honors_temperature
from evalshift.models.client import ModelClient, ModelClientError
from evalshift.models.registry import resolve_model
from evalshift.parsers.base import PromptParseError, PromptTemplate
from evalshift.parsers.manual import ManualParser
from evalshift.parsers.python_string import PythonStringParser
from evalshift.runner.checkpoint import (
    append_call,
    completed_call_keys,
    compute_config_hash,
    find_latest_in_progress,
    generate_run_id,
    prune_runs,
    resolve_max_runs,
    run_dir_for,
    touch_checkpoint,
    validate_resume,
    write_state,
)
from evalshift.runner.generation import translate_generation_config
from evalshift.runner.models import Call, CallRole, RunModels, RunState
from evalshift.suite.models import ChatMessage, Suite, SuiteExample
from evalshift.utils.cost import CostEstimate, estimate_run_cost
from evalshift.utils.templating import (
    SuiteCompatibilityError,
    render,
    validate_suite_against_prompts,
)

log = logging.getLogger(__name__)

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
    ``ModelClient.complete`` path runs. ``tools`` is sourced from the
    dispatched ``example``'s own toolset (:func:`resolve_example_tools`) —
    two ``WorkItem``s built from the same prompt can carry different
    toolsets, or none, depending on what each example asserts.
    """

    prompt: PromptTemplate
    example: SuiteExample
    role: CallRole
    model_id: str  # canonical id, post-alias resolution
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int | None = None  # effective cap; None → registry default


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


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Snapshot of in-flight run state, emitted to ``on_progress``.

    Sent once per completed call so the caller can render its own bar.
    """

    completed: int
    total: int
    cached: int
    live: int
    failed: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class CostPlan:
    """Cost estimate + total call count, used by ``preflight_cost``."""

    estimated_usd: float
    total_calls: int


class RunAborted(Exception):  # noqa: N818 — keeps the existing public name.
    """Raised when the user declines the cost prompt or a precondition fails."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_orchestrator(
    *,
    config: EvalShiftConfig,
    config_path: Path,
    suite: Suite,
    suite_path: Path,
    source_model: str,
    target_model: str,
    runs_base: Path | None = None,
    resume: bool = False,
    yes: bool = False,
    run_slug: str | None = None,
    suite_name: str | None = None,
    console: Console | None = None,
    client: ModelClient | None = None,
    cache: CacheStore | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
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
        runs_base: Override for ``.evalshift/runs/``. Tests pass a
            tmp path; production leaves it default.
        resume: If True, look for the latest in-progress run and pick
            up where it left off. The config + suite must match.
        yes: Skip the cost-confirmation prompt regardless of estimate.
        run_slug: Suite slug baked into the run id.
        suite_name: The ``suites:`` key the run was launched with, recorded in
            ``state.json`` so later stages can resolve this suite's evaluator
            set. ``None`` for a raw ``--suite <path>`` run.
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

    # Per-example toolsets (v0.3): each example carries its own toolset
    # (inline ``tools`` or a ``toolset_ref`` sidecar) rather than inheriting
    # one shared toolset from its prompt's config — see resolve_example_tools.
    # A ``toolset_ref`` is resolved against whichever of several candidate
    # base directories actually holds its sidecar; never assumed to be
    # ``.evalshift`` (see toolset_base_candidates).
    toolset_bases = toolset_base_candidates(suite_path=suite_path)

    # Effective completion cap per prompt: the prompt's own override, else
    # the run-wide default. Fed into both the client call and the cache key.
    max_tokens_by_prompt = {
        p.id: (p.max_tokens if p.max_tokens is not None else config.defaults.max_tokens)
        for p in config.prompts
    }

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
        run_slug=run_slug,
        suite_name=suite_name,
    )

    # Build the full work list (every required call) and filter out
    # whatever the resume scan already saw.
    work = _build_work_list(
        templates=templates,
        suite=suite,
        canonical_source=canonical_source,
        canonical_target=canonical_target,
        toolset_bases=toolset_bases,
        max_tokens_by_prompt=max_tokens_by_prompt,
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
            on_progress=on_progress,
        )
    finally:
        if cache_owned:
            await cache_inst.close()

    _prune_old_runs(config, runs_base, keep_run_id=result.run_id, console=cons)
    return result


def _prune_old_runs(
    config: EvalShiftConfig,
    runs_base: Path | None,
    *,
    keep_run_id: str,
    console: Console,
) -> None:
    """Best-effort retention sweep after a run completes; never fails the run."""
    try:
        removed = prune_runs(
            runs_base,
            max_runs_per_suite=resolve_max_runs(config.retention.max_runs_per_suite),
            run_ttl_days=config.retention.run_ttl_days,
            keep_run_id=keep_run_id,
        )
    except Exception:  # retention must never break a completed run
        return
    if removed:
        console.print(f"[dim]retention: pruned {len(removed)} old run(s)[/dim]")


def preflight_cost(
    *,
    config: EvalShiftConfig,
    config_path: Path,
    suite: Suite,
    source_model: str,
    target_model: str,
) -> CostPlan:
    """Estimate cost + call count without dispatching any work.

    Used by ``evalshift all`` to render the "estimated cost" row before
    starting the run, and to decide whether the cost-confirmation prompt
    will fire. Mirrors what :func:`run_orchestrator` computes internally
    so the two stay in lockstep.
    """
    project_root = config_path.resolve().parent
    templates = _parse_prompts(config, project_root)
    canonical_source = resolve_model(source_model).id
    canonical_target = resolve_model(target_model).id
    estimate = _estimate(
        templates=templates,
        suite=suite,
        canonical_source=canonical_source,
        canonical_target=canonical_target,
    )
    total_calls = len(templates) * len(suite) * 2
    return CostPlan(
        estimated_usd=estimate.estimated_usd,
        total_calls=total_calls,
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _parse_prompts(config: EvalShiftConfig, project_root: Path) -> list[PromptTemplate]:
    """Pick the right parser per prompt; raise PromptParseError on first failure."""
    out: list[PromptTemplate] = []
    for prompt in config.prompts:
        parser = ManualParser() if prompt.detection == "manual" else PythonStringParser()
        out.append(parser.parse(prompt, project_root))
    return out


def toolset_base_candidates(*, suite_path: Path) -> tuple[Path, ...]:
    """Ordered, de-duplicated candidate directories for resolving a ``toolset_ref`` sidecar.

    A ``toolset_ref`` value is content-addressed and location-agnostic — it says
    nothing about where its sidecar lives — so the caller must supply candidate
    base directories for :func:`~evalshift.captures.reader.toolset_path` /
    :func:`~evalshift.captures.reader.load_toolset`. Computed once per run and
    tried in order by :func:`resolve_example_tools` (first hit wins):

    1. :func:`~evalshift.captures.reader.capture_base` — ``$EVALSHIFT_DIR`` or
       ``.evalshift`` (resolved relative to the current working directory). The
       standard location: what ``capture sync``/``capture promote`` write, and
       where a real captured-then-promoted suite's sidecars live. Tried first
       (the common case) — never *assumed* when it 404s, only preferred.
    2. ``suite_path``'s own resolved directory — covers a checked-in,
       hand-authored suite that ships its sidecar alongside the golden file
       itself (e.g. ``examples/agent/golden.jsonl`` next to
       ``examples/agent/toolsets/``, colocating its own copy of the toolset
       rather than pointing at a shared directory). ``.evalshift/`` is
       gitignored and cannot hold committed content, so a repo-shipped
       example cannot rely on tier 1.

    Returns:
        Candidate directories, de-duplicated and order-preserving. Always
        non-empty (tier 1 alone guarantees that).
    """
    candidates: list[Path] = [capture_base(), suite_path.resolve().parent]
    # dict.fromkeys dedupes while preserving first-seen order; Path is hashable.
    return tuple(dict.fromkeys(candidates))


def _load_toolset_from_candidates(
    ref: str,
    bases: Sequence[Path],
    cache: MutableMapping[str, list[ToolSpec]],
) -> list[ToolSpec]:
    """Resolve ``ref`` against each of ``bases`` in order; the first hit wins.

    A ``"missing"`` :class:`~evalshift.captures.reader.CaptureError` at one base
    means "not here, try the next candidate" and is swallowed until every base
    is exhausted. Any other error kind (corrupt JSON, wrong schema, ...) means a
    sidecar *was* found but is broken, and must surface immediately rather than
    being masked by silently moving on to the next candidate.

    Raises:
        CaptureError: ``kind="missing"``, naming every location tried, if no
            candidate resolves ``ref``.
    """
    if ref in cache:
        return cache[ref]
    tried: list[Path] = []
    for base in bases:
        try:
            return load_toolset(ref, base=base, cache=cache)
        except CaptureError as exc:
            if exc.kind != "missing":
                raise
            tried.append(exc.path)
    locations = ", ".join(str(p) for p in tried) or "(no candidate base directories)"
    raise CaptureError(
        tried[0] if tried else Path(ref),
        "missing",
        f"toolset sidecar for {ref!r} not found in any candidate location: {locations}",
    )


def resolve_example_tools(
    example: SuiteExample,
    *,
    toolset_bases: Sequence[Path],
    toolset_cache: MutableMapping[str, list[ToolSpec]],
) -> tuple[ToolSpec, ...]:
    """Resolve one example's toolset — inline, or a ``toolset_ref`` sidecar.

    ``SuiteExample`` guarantees exactly one of ``tools`` / ``toolset_ref`` is set
    (its ``_check_exactly_one_toolset_field`` validator), so this never falls
    back to an empty-by-default toolset: an example naming a ref resolves it or
    the run fails loudly, and an example that says ``tools: []`` dispatches with
    none because it truthfully has none — not because resolution silently gave
    up.

    Args:
        example: The suite example being dispatched.
        toolset_bases: Candidate directories to resolve a ``toolset_ref``
            sidecar against, tried in order — see
            :func:`toolset_base_candidates`. Unused (no I/O) when ``example``
            carries inline ``tools``.
        toolset_cache: Run-scoped cache keyed by ``toolset_ref``, shared across
            every example, so N examples referencing the same toolset read and
            parse its sidecar at most once.

    Returns:
        The example's toolset, possibly empty — a real, asserted value, never
        a silent substitute for one that couldn't be found.

    Raises:
        CaptureError: ``example.toolset_ref`` is set but no candidate base
            resolves it to a valid sidecar.
    """
    if example.tools is not None:
        return tuple(example.tools)
    assert example.toolset_ref is not None  # guaranteed by SuiteExample's validator
    return tuple(_load_toolset_from_candidates(example.toolset_ref, toolset_bases, toolset_cache))


def _fingerprint_toolset(tools: Sequence[ToolSpec]) -> str:
    """Content-address a resolved toolset the same way regardless of its source.

    An inline ``tools:`` list and a ``toolset_ref`` sidecar both resolve to the
    same ``list[ToolSpec]`` shape by the time dispatch sees them. Fingerprinting
    that resolved list — via Task 2's
    :func:`~evalshift.captures.toolset.fingerprint_tools` — rather than trusting
    a ``toolset_ref`` string verbatim guarantees the two spellings of the same
    toolset produce the same fingerprint, and therefore the same cache key.
    """
    return fingerprint_tools([t.to_anthropic() for t in tools])


def _setup_run(
    *,
    config: EvalShiftConfig,
    config_hash: str,
    suite: Suite,
    suite_path: Path,
    templates: list[PromptTemplate],
    canonical_source: str,
    canonical_target: str,
    runs_base: Path | None,
    resume: bool,
    run_slug: str | None = None,
    suite_name: str | None = None,
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

    run_id = generate_run_id(suite_slug=run_slug)
    run_dir = run_dir_for(run_id, runs_base)
    total = len(templates) * len(suite) * 2  # source + target
    state = RunState(
        run_id=run_id,
        config_hash=config_hash,
        started_at=datetime.now(UTC),
        models=RunModels(source=canonical_source, target=canonical_target),
        prompt_ids=[t.id for t in templates],
        suite_path=str(suite_path),
        suite_name=suite_name,
        total_evaluations=total,
        non_deterministic_models=detect_non_deterministic_models(
            source=canonical_source,
            target=canonical_target,
        ),
    )
    write_state(run_dir, state)
    return run_dir, state, set()


def detect_non_deterministic_models(*, source: str, target: str) -> list[str]:
    """Return the run's model ids that do not honour ``temperature``.

    Both arms are checked because either can be the affected one, and an A/A
    run may name the same model twice — the result is deduplicated so such a
    model is reported once. Order follows source-then-target so reports read
    predictably.

    Args:
        source: Canonical id of the source model.
        target: Canonical id of the target model.

    Returns:
        Canonical ids whose sampling is non-deterministic, or an empty list
        when both arms sample deterministically (the case for every provider
        at the time of writing).
    """
    affected: list[str] = []
    for model_id in (source, target):
        if model_id not in affected and not honors_temperature(model_id):
            affected.append(model_id)
    return affected


def _build_work_list(
    *,
    templates: list[PromptTemplate],
    suite: Suite,
    canonical_source: str,
    canonical_target: str,
    # No default (M3 of the final review): the one caller (run_orchestrator)
    # always computes real candidates via toolset_base_candidates first,
    # which is never empty -- an omitted argument here would silently run
    # with zero candidate bases rather than fail loudly, so "never default"
    # is enforced by the signature, not just the docstring.
    toolset_bases: Sequence[Path],
    max_tokens_by_prompt: dict[str, int] | None = None,
) -> list[WorkItem]:
    max_tokens_by_prompt = max_tokens_by_prompt or {}

    # Resolve every example's own toolset exactly once, regardless of how many
    # templates it gets dispatched under (a suite runs against every prompt).
    # toolset_cache is shared across the whole pass so N examples that
    # reference the same toolset_ref read and parse its sidecar once.
    toolset_cache: dict[str, list[ToolSpec]] = {}
    tools_by_example: dict[str, tuple[ToolSpec, ...]] = {
        example.id: resolve_example_tools(
            example,
            toolset_bases=toolset_bases,
            toolset_cache=toolset_cache,
        )
        for example in suite.examples
    }

    work: list[WorkItem] = []
    for tmpl in templates:
        max_tokens = max_tokens_by_prompt.get(tmpl.id)
        for example in suite.examples:
            tools = tools_by_example[example.id]
            work.append(
                WorkItem(
                    prompt=tmpl,
                    example=example,
                    role="source",
                    model_id=canonical_source,
                    tools=tools,
                    max_tokens=max_tokens,
                ),
            )
            work.append(
                WorkItem(
                    prompt=tmpl,
                    example=example,
                    role="target",
                    model_id=canonical_target,
                    tools=tools,
                    max_tokens=max_tokens,
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
    # Multi-turn examples carry a recorded history prefix that gets sent
    # on every call; count its characters as extra input so the estimate
    # doesn't silently ignore what can be a large prefix.
    per_example_extra_chars = [
        sum(len(m.content) for m in (e.history or [])) for e in suite.examples
    ]
    return estimate_run_cost(
        template=representative,
        examples=[e.inputs for e in suite.examples],
        n_prompts=len(templates),
        models=[canonical_source, canonical_target],
        per_example_extra_chars=per_example_extra_chars,
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


def build_messages(example: SuiteExample, prompt_text: str) -> list[dict[str, Any]] | None:
    """Build the messages list for a dispatch, or ``None`` for single-turn.

    Args:
        example: The suite example being dispatched. Its ``history``
            (``None`` for single-turn, else a recorded conversation
            prefix) determines whether message-mode dispatch is used.
        prompt_text: The current turn's fully-rendered prompt text.

    Returns:
        ``None`` when ``example.history is None`` — callers should use the
        plain-prompt dispatch path. Otherwise the recorded history followed
        by the current turn as a ``user`` message. Tool turns are emitted in
        the OpenAI wire shape (``assistant.tool_calls[].function.arguments``
        as a JSON string, ``tool`` messages keyed by ``tool_call_id``);
        LiteLLM translates that to each provider's own form, so recorded
        agent loops replay against any backend. An empty-list ``history``
        still returns a (single-element) message list — it marks the example
        as message-mode even though there's no prefix to replay.
    """
    if example.history is None:
        return None
    return [_dispatch_message(m) for m in example.history] + [
        {"role": "user", "content": prompt_text},
    ]


def history_for_cache_key(example: SuiteExample) -> list[dict[str, Any]] | None:
    """The example's history as plain dicts, or ``None`` for single-turn.

    Unset tool fields are omitted so a text-only prefix serialises exactly as
    it did before ``tool_calls``/``tool_call_id`` existed — adding the fields
    must not invalidate every cached multi-turn response. Recorded tool calls
    *are* part of the key: two examples differing only in tool arguments are
    different prompts and must not collide.
    """
    if example.history is None:
        return None
    return [m.model_dump(exclude_none=True) for m in example.history]


def _dispatch_message(msg: ChatMessage) -> dict[str, Any]:
    """Render one history entry in the provider-neutral OpenAI wire shape."""
    if msg.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }
    if msg.role == "assistant" and msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for index, call in enumerate(msg.tool_calls)
            ],
        }
    return {"role": msg.role, "content": msg.content}


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
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> RunResult:
    """Process every pending work item under the concurrency semaphore."""
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = already_done
    cached_count = 0
    live_count = 0
    failed_count = 0
    total_cost = 0.0

    use_callback = on_progress is not None
    progress: Progress | None = None
    task_id: TaskID | None = None
    if not use_callback:
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
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1, cost=total_cost)
                if on_progress is not None:
                    on_progress(
                        ProgressEvent(
                            completed=completed,
                            total=total,
                            cached=cached_count,
                            live=live_count,
                            failed=failed_count,
                            cost_usd=total_cost,
                        ),
                    )
                if completed % CHECKPOINT_EVERY == 0:
                    write_state(run_dir, touch_checkpoint(state, completed))

    progress_ctx = progress if progress is not None else nullcontext()
    with progress_ctx:
        await asyncio.gather(*(_do_one(item) for item in pending))

    # Final checkpoint + status flip. Runtime-discovered temperature
    # rejections join the probe-detected list here so the report's
    # non-determinism banner covers both. Probe entries keep their order;
    # runtime additions follow, sorted, deduplicated. (A resumed run
    # re-discovers rejections on its live calls; a fully-cached resume
    # makes no live calls and adds nothing new — the prior final state
    # already carried them.)
    runtime_nondet = sorted(
        set(client.temperature_rejected_models) - set(state.non_deterministic_models)
    )
    final_state = touch_checkpoint(state, completed).model_copy(
        update={
            "status": "completed",
            "non_deterministic_models": [
                *state.non_deterministic_models,
                *runtime_nondet,
            ],
        },
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
    messages = build_messages(item.example, prompt_text)

    if item.tools:
        return await _execute_with_tools(
            client=client,
            run_id=run_id,
            item=item,
            prompt_text=prompt_text,
            canonical_id=meta.id,
            messages=messages,
        )

    # Effective cap: prompt/run config override, else the registry default.
    # The same value is keyed AND sent so a cache hit matches the live call.
    effective_max_tokens = (
        item.max_tokens if item.max_tokens is not None else meta.default_max_tokens
    )

    # Recorded generation config (captured suites): the same values are keyed
    # AND sent, so editing the example's config invalidates its cache entries.
    gen_temperature, gen_extra = translate_generation_config(item.example.generation_config)
    effective_temperature = (
        gen_temperature if gen_temperature is not None else meta.default_temperature
    )

    history_for_key = history_for_cache_key(item.example)
    # item.tools is always empty here — a non-empty toolset routes to
    # _execute_with_tools above and never reaches this line — so this is
    # always None in practice today. Written as a real conditional (not
    # hard-coded) because that empty-ness is a routing fact, not a cache-key
    # rule: this is the one call site that mirrors _execute_with_tools's own
    # `_fingerprint_toolset(item.tools) if item.tools else None` shape, so the
    # two stay in lockstep if either path's routing condition ever changes.
    # None (omit from the payload) matches the history/generation_config
    # precedent below: this call never sends a `tools` parameter to the
    # provider at all, so it keeps its pre-existing cache key rather than
    # forking on a toolset dimension that doesn't apply to it.
    toolset_fingerprint = _fingerprint_toolset(item.tools) if item.tools else None
    key = cache_key(
        model_id=meta.id,
        prompt_text=prompt_text,
        inputs=item.example.inputs,
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
        history=history_for_key,
        generation_config=item.example.generation_config,
        toolset_fingerprint=toolset_fingerprint,
    )

    if cache_enabled:
        hit = await cache.get(key)
        if hit is not None:
            if hit.finish_reason == "length":
                log.warning(
                    "cached response for model %s was truncated (finish_reason=length)",
                    meta.id,
                )
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
                finish_reason=hit.finish_reason,
            )

    try:
        if messages is not None:
            result = await client.complete_messages(
                model=meta.id,
                messages=messages,
                temperature=gen_temperature,
                max_tokens=effective_max_tokens,
                extra=gen_extra,
            )
        else:
            result = await client.complete(
                model=meta.id,
                prompt=prompt_text,
                temperature=gen_temperature,
                max_tokens=effective_max_tokens,
                extra=gen_extra,
            )
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
        finish_reason=result.finish_reason,
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
            finish_reason=result.finish_reason,
        )

    return call


async def _execute_with_tools(
    *,
    client: ModelClient,
    run_id: str,
    item: WorkItem,
    prompt_text: str,
    canonical_id: str,
    messages: list[dict[str, Any]] | None = None,
) -> Call:
    """Tool-aware call path: dispatch + record the trace on the Call.

    ``messages`` is set for multi-turn examples (``example.history`` is
    not ``None``) and dispatches via
    :meth:`ModelClient.complete_messages_with_tools`; ``None`` keeps the
    existing single-prompt :meth:`ModelClient.complete_with_tools` path.
    The local cache is bypassed for both — unchanged from before.
    """
    gen_temperature, gen_extra = translate_generation_config(item.example.generation_config)
    try:
        if messages is not None:
            result = await client.complete_messages_with_tools(
                model=canonical_id,
                messages=messages,
                tools=list(item.tools),
                temperature=gen_temperature,
                max_tokens=item.max_tokens,
                extra=gen_extra,
            )
        else:
            result = await client.complete_with_tools(
                model=canonical_id,
                prompt=prompt_text,
                tools=list(item.tools),
                temperature=gen_temperature,
                max_tokens=item.max_tokens,
                extra=gen_extra,
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
        finish_reason=result.finish_reason,
    )


# Re-exported so the CLI can catch them with one import.
__all__ = [
    "CHECKPOINT_EVERY",
    "COST_CONFIRM_THRESHOLD_USD",
    "CostPlan",
    "ProgressEvent",
    "PromptParseError",
    "RunAborted",
    "RunResult",
    "SuiteCompatibilityError",
    "WorkItem",
    "build_messages",
    "history_for_cache_key",
    "preflight_cost",
    "resolve_example_tools",
    "run_orchestrator",
    "toolset_base_candidates",
]
