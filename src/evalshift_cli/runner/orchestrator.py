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
   ``(prompt x example x {source, target} x sample)`` — one sample unless
   ``defaults.samples_per_example`` asks for repeats).
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
from collections.abc import Callable, Mapping, MutableMapping, Sequence
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

from evalshift_cli.cache.store import CacheStore, cache_key
from evalshift_cli.captures.reader import CaptureError, capture_base, load_toolset
from evalshift_cli.captures.toolset import fingerprint_tools
from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.evaluators.tool_models import ToolCall, ToolSpec, ToolTrace
from evalshift_cli.models.capabilities import (
    TOOL_STRICT_PARAM,
    honors_temperature,
    silently_unsent_params,
    unsupported_params,
)
from evalshift_cli.models.client import ModelClient, ModelClientError
from evalshift_cli.models.registry import resolve_model
from evalshift_cli.parsers.base import PromptParseError, PromptTemplate
from evalshift_cli.parsers.manual import ManualParser
from evalshift_cli.parsers.python_string import PythonStringParser
from evalshift_cli.runner.checkpoint import (
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
from evalshift_cli.runner.generation import translate_generation_config
from evalshift_cli.runner.models import Call, CallRole, RunModels, RunState
from evalshift_cli.suite.models import ChatMessage, Suite, SuiteExample, ToolResultFixture
from evalshift_cli.utils.cost import CostEstimate, estimate_run_cost
from evalshift_cli.utils.templating import (
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
    """One unit of work: one example replayed against one model.

    That is a single LLM call, except for an example carrying
    ``tool_result_fixtures``, which is replayed teacher-forced over
    :meth:`SuiteExample.rounds_to_replay` rounds — one call per round, merged
    into the one :class:`Call` this item produces. Either way a ``WorkItem``
    maps 1:1 to a ``(prompt, example, role, sample)`` row of ``raw.jsonl``,
    which is what resume, evaluate pairing and the progress bar all count.
    ``sample_index`` is ``0`` unless ``defaults.samples_per_example`` repeats
    the example, in which case each repeat is its own item.

    When ``tools`` is non-empty, the orchestrator dispatches to
    ``ModelClient.complete_with_tools`` and the resulting :class:`Call`
    carries a populated ``trace``. Otherwise the standard text-only
    ``ModelClient.complete`` path runs (and never replays rounds — only the
    tool path loops). ``tools`` is sourced from the
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
    sample_index: int = 0


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
    # Resolved once for the whole run, before anything else needs them: the
    # work list dispatches them, and the run-start capability record reads
    # their ``strict`` flags. Resolving twice would re-read sidecars from disk.
    tools_by_example = resolve_suite_tools(suite, toolset_bases=toolset_bases)

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
        tools_by_example=tools_by_example,
    )

    # Build the full work list (every required call) and filter out
    # whatever the resume scan already saw.
    work = _build_work_list(
        templates=templates,
        suite=suite,
        canonical_source=canonical_source,
        canonical_target=canonical_target,
        tools_by_example=tools_by_example,
        max_tokens_by_prompt=max_tokens_by_prompt,
        samples_per_example=config.defaults.samples_per_example,
    )
    pending = [
        w for w in work if (w.prompt.id, w.example.id, w.role, w.sample_index) not in completed_keys
    ]

    # Cost estimate + confirmation. We only show the prompt for *new*
    # runs above the threshold; a resume is implicitly already approved.
    if not resume:
        estimate = _estimate(
            templates=templates,
            suite=suite,
            canonical_source=canonical_source,
            canonical_target=canonical_target,
            samples_per_example=config.defaults.samples_per_example,
        )
        if (
            not yes
            and estimate.estimated_usd > COST_CONFIRM_THRESHOLD_USD
            and not _confirm_cost(cons, estimate, estimate.total_calls)
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
            samples_per_example=config.defaults.samples_per_example,
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
        samples_per_example=config.defaults.samples_per_example,
    )
    # LLM calls, not ``Call`` rows: a teacher-forced example costs one call per
    # replayed round. ``_estimate`` already computed exactly that shape
    # (``n_prompts x sum(rounds_to_replay) x 2 x samples_per_example``), so read
    # it off there rather than recomputing ``len(templates) * len(suite) * 2``
    # and drifting.
    return CostPlan(
        estimated_usd=estimate.estimated_usd,
        total_calls=estimate.total_calls,
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
    base directories for :func:`~evalshift_cli.captures.reader.toolset_path` /
    :func:`~evalshift_cli.captures.reader.load_toolset`. Computed once per run and
    tried in order by :func:`resolve_example_tools` (first hit wins):

    1. :func:`~evalshift_cli.captures.reader.capture_base` — ``$EVALSHIFT_DIR`` or
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

    A ``"missing"`` :class:`~evalshift_cli.captures.reader.CaptureError` at one base
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


def resolve_suite_tools(
    suite: Suite,
    *,
    toolset_bases: Sequence[Path],
) -> dict[str, tuple[ToolSpec, ...]]:
    """Resolve every example's toolset once, for the whole run.

    A suite runs against every prompt, so an example is dispatched N times but
    its toolset must be read from disk at most once; the run-start capability
    record needs the same resolved specs. Both callers share this result rather
    than resolving independently.

    Args:
        suite: The suite about to be replayed.
        toolset_bases: Candidate directories for ``toolset_ref`` sidecars,
            tried in order — see :func:`toolset_base_candidates`.

    Returns:
        Example id → that example's resolved toolset, possibly empty.

    Raises:
        CaptureError: An example names a ``toolset_ref`` no candidate base
            resolves. Raised here, before the run directory exists, because a
            run that cannot offer its tools has nothing to record.
    """
    toolset_cache: dict[str, list[ToolSpec]] = {}
    return {
        example.id: resolve_example_tools(
            example,
            toolset_bases=toolset_bases,
            toolset_cache=toolset_cache,
        )
        for example in suite.examples
    }


def _fingerprint_toolset(tools: Sequence[ToolSpec]) -> str:
    """Content-address a resolved toolset the same way regardless of its source.

    An inline ``tools:`` list and a ``toolset_ref`` sidecar both resolve to the
    same ``list[ToolSpec]`` shape by the time dispatch sees them. Fingerprinting
    that resolved list — via Task 2's
    :func:`~evalshift_cli.captures.toolset.fingerprint_tools` — rather than trusting
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
    tools_by_example: Mapping[str, Sequence[ToolSpec]] | None = None,
) -> tuple[Path, RunState, set[tuple[str, str, str, int]]]:
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
    samples = config.defaults.samples_per_example
    total = len(templates) * len(suite) * 2 * samples  # (source + target) x samples
    state = RunState(
        run_id=run_id,
        config_hash=config_hash,
        started_at=datetime.now(UTC),
        models=RunModels(source=canonical_source, target=canonical_target),
        prompt_ids=[t.id for t in templates],
        suite_path=str(suite_path),
        suite_name=suite_name,
        total_evaluations=total,
        samples_per_example=samples,
        non_deterministic_models=detect_non_deterministic_models(
            source=canonical_source,
            target=canonical_target,
        ),
        dropped_params=detect_dropped_params(
            source=canonical_source,
            target=canonical_target,
            suite=suite,
            tools_by_example=tools_by_example,
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


#: Recorded ``generation_config`` key → the OpenAI-style parameter name
#: ``litellm.get_supported_openai_params`` answers about. Several providers
#: spell one constraint differently (Gemini's ``response_mime_type`` /
#: ``response_schema`` are structured output, its ``tool_config`` is
#: ``tool_choice``, its ``max_output_tokens`` is the completion cap), so the
#: map is many-to-one and the result is deduplicated.
#:
#: Keys absent from this map are not probed. That is deliberate: LiteLLM only
#: answers about OpenAI params, so a foreign key (``seed``,
#: ``candidate_count``) has no answer to give and guessing one would
#: manufacture a banner out of ignorance.
_GENERATION_KEY_TO_OPENAI_PARAM: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "response_format": "response_format",
    "response_mime_type": "response_format",
    "response_schema": "response_format",
    "max_tokens": "max_tokens",
    "max_output_tokens": "max_tokens",
    "tool_choice": "tool_choice",
    "tool_config": "tool_choice",
    "parallel_tool_calls": "parallel_tool_calls",
}

#: Parameters this probe deliberately leaves alone because a dedicated,
#: stronger check already owns them. ``temperature`` is
#: :func:`detect_non_deterministic_models`' subject: that probe runs on every
#: run rather than only when a capture happened to record a temperature, and
#: it earns its own banner. Reporting it here too would put two banners on one
#: run saying the same thing.
_PARAMS_REPORTED_ELSEWHERE = frozenset({"temperature"})


def requested_generation_params(suite: Suite) -> list[str]:
    """Return the OpenAI-param names the suite's recorded configs ask for.

    Args:
        suite: The suite about to be replayed. Examples without a
            ``generation_config`` (hand-written ones, and captures from before
            the SDK recorded it) contribute nothing.

    Returns:
        Sorted, deduplicated parameter names to probe both arms with. Empty
        when no example recorded a constraint worth checking — in which case
        there is nothing LiteLLM could drop and no probe is needed at all.
    """
    names: set[str] = set()
    for example in suite.examples:
        for key in example.generation_config or {}:
            param = _GENERATION_KEY_TO_OPENAI_PARAM.get(key)
            if param is not None and param not in _PARAMS_REPORTED_ELSEWHERE:
                names.add(param)
    return sorted(names)


def detect_dropped_params(
    *,
    source: str,
    target: str,
    suite: Suite,
    tools_by_example: Mapping[str, Sequence[ToolSpec]] | None = None,
) -> dict[str, list[str]]:
    """Return each arm's generation parameters LiteLLM will silently drop.

    ``ModelClient`` sets ``drop_params=True`` so a call carrying a parameter
    the provider never accepted still succeeds. That keeps runs alive and
    costs the constraint: the replay stops reproducing what the capture
    pinned, and the arm measures the model change *plus* a missing constraint.
    Recording it here is what lets the report say so.

    Two sources feed one record. Most parameters are *probed*
    (:func:`~evalshift_cli.models.capabilities.unsupported_params` asks
    LiteLLM). A short enumerated list is not probeable, because LiteLLM
    positively claims support and then discards the value while building the
    provider body — those come from
    :func:`~evalshift_cli.models.capabilities.silently_unsent_params`. Either
    way the constraint is missing from the wire, so the two results merge into
    one sorted list per model.

    Both arms are checked because either can be the affected one — a source
    model is replayed too, not merely recorded — and an A/A run may name the
    same model twice, so a model is checked and listed at most once.

    Args:
        source: Canonical id of the source model.
        target: Canonical id of the target model.
        suite: The suite being replayed; supplies the parameters to ask about.
        tools_by_example: The run's already-resolved toolsets, example id →
            specs (see :func:`resolve_suite_tools`). Passed in rather than
            re-resolved because resolving reads sidecars from disk. Only the
            specs' ``strict`` flags are read: a strict tool anywhere in the
            suite means the replay asks for schema-exact arguments, which is
            recorded under the :data:`~evalshift_cli.models.capabilities.TOOL_STRICT_PARAM`
            pseudo-name. ``None`` (the default) asks about generation
            parameters only.

    Returns:
        Canonical model id → sorted parameter names that model will not
        receive. Models that honour every recorded constraint are absent, so
        an empty dict means nothing is being dropped (and is also what an
        uncertain LiteLLM returns — see
        :func:`~evalshift_cli.models.capabilities.unsupported_params`).
    """
    probed = requested_generation_params(suite)
    # The pseudo-parameter never goes to the probe: LiteLLM has no answer for a
    # name outside its OpenAI vocabulary, and asking would only invite one.
    constraints = set(probed)
    if _suite_asks_for_strict_tools(tools_by_example):
        constraints.add(TOOL_STRICT_PARAM)
    if not constraints:
        return {}
    dropped: dict[str, list[str]] = {}
    for model_id in (source, target):
        if model_id in dropped:
            continue
        found = set(silently_unsent_params(model_id, constraints))
        if probed:
            found |= set(unsupported_params(model_id, probed))
        missing = sorted(found)
        if not missing:
            continue
        dropped[model_id] = missing
        for param in missing:
            # Warning, and once per (model, param) rather than once per
            # dispatched call: a per-call debug line for something that
            # invalidates a whole arm is both unreadable and unread.
            log.warning(
                "%s will not receive %r; it is dropped before the request reaches the "
                "provider, so this run does not replay the recorded constraint on that arm",
                model_id,
                param,
            )
    return dropped


def _suite_asks_for_strict_tools(
    tools_by_example: Mapping[str, Sequence[ToolSpec]] | None,
) -> bool:
    """Report whether any resolved toolset in the run carries ``strict``.

    One strict tool is enough: the flag is per-tool, but the record is per
    model, and a model that cannot express strictness cannot express it for
    any of them.
    """
    if not tools_by_example:
        return False
    return any(tool.strict for tools in tools_by_example.values() for tool in tools)


def _build_work_list(
    *,
    templates: list[PromptTemplate],
    suite: Suite,
    canonical_source: str,
    canonical_target: str,
    # No default (M3 of the final review): the one caller (run_orchestrator)
    # resolves every example's toolset up front via resolve_suite_tools --
    # once for the whole run, however many templates each example is
    # dispatched under -- and the run-start capability record reads the same
    # mapping. An omitted argument here would silently dispatch every example
    # tool-less rather than fail loudly, so "never default" is enforced by the
    # signature, not just the docstring.
    tools_by_example: Mapping[str, tuple[ToolSpec, ...]],
    max_tokens_by_prompt: dict[str, int] | None = None,
    samples_per_example: int = 1,
) -> list[WorkItem]:
    """One item per ``(prompt, example, role, sample)``.

    ``samples_per_example`` (``defaults.samples_per_example``) repeats every
    ``(prompt, example)`` that many times per role, ``sample_index`` running
    ``0..N-1``. The default of ``1`` reproduces the single-sample list
    exactly.
    """
    max_tokens_by_prompt = max_tokens_by_prompt or {}

    work: list[WorkItem] = []
    for tmpl in templates:
        max_tokens = max_tokens_by_prompt.get(tmpl.id)
        for example in suite.examples:
            tools = tools_by_example[example.id]
            for sample_index in range(samples_per_example):
                work.append(
                    WorkItem(
                        prompt=tmpl,
                        example=example,
                        role="source",
                        model_id=canonical_source,
                        tools=tools,
                        max_tokens=max_tokens,
                        sample_index=sample_index,
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
                        sample_index=sample_index,
                    ),
                )
    return work


def _estimate(
    *,
    templates: list[PromptTemplate],
    suite: Suite,
    canonical_source: str,
    canonical_target: str,
    samples_per_example: int = 1,
) -> CostEstimate:
    # Pick the longest-looking template as the representative for the
    # estimate so we err on the side of an over-estimate.
    representative = max((t.content for t in templates), key=len, default="")
    # Multi-turn examples carry a recorded history prefix that gets sent
    # on every call; count its characters as extra input so the estimate
    # doesn't silently ignore what can be a large prefix. A teacher-forced
    # example also grows its own context round by round as recorded results
    # are fed back — count every rendered fixture once, so the growing
    # context shows up in the estimate rather than surprising the user.
    per_example_extra_chars = [
        sum(len(m.content) for m in (e.history or []))
        + sum(
            len(_render_fixture(fixture))
            for round_fixtures in (e.tool_result_fixtures or [])
            for fixture in round_fixtures
        )
        for e in suite.examples
    ]
    return estimate_run_cost(
        template=representative,
        examples=[e.inputs for e in suite.examples],
        n_prompts=len(templates),
        models=[canonical_source, canonical_target],
        per_example_extra_chars=per_example_extra_chars,
        # One call per replayed round, not one per example (see
        # SuiteExample.rounds_to_replay). Every example is 1 unless its
        # fixtures ask for a teacher-forced loop.
        per_example_calls=[e.rounds_to_replay() for e in suite.examples],
        # Every one of those calls repeats per sample (defaults.samples_per_example).
        samples_per_example=samples_per_example,
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


def build_round_messages(
    example: SuiteExample,
    prompt_text: str,
    round_index: int,
) -> list[dict[str, Any]] | None:
    """Build the messages for round ``round_index`` of a teacher-forced replay.

    Round *k* shows the candidate the rendered prompt followed by the
    **recorded** rounds ``0..k-1`` — the ground-truth assistant tool calls and
    the recorded results of those calls — and asks it for round *k*. The
    candidate's own calls are never fed back, which is what makes round *k*
    comparable across source, target and ground truth: all three saw a
    byte-identical context.

    Args:
        example: The suite example being replayed. Its ``history``,
            ``expected_tool_rounds`` and ``tool_result_fixtures`` are the
            whole input.
        prompt_text: The current turn's fully-rendered prompt text.
        round_index: 0-based round to build, below
            :meth:`SuiteExample.rounds_to_replay`.

    Returns:
        For ``round_index == 0``, exactly what :func:`build_messages` returns —
        ``None`` for a single-turn example (dispatch via the plain-prompt path)
        or the history prefix plus the current turn otherwise. Round 0 is
        deliberately identical to a single-shot dispatch: replaying a loop must
        not change what the first round asks for. For ``round_index >= 1``, that
        same prefix followed by one ``assistant`` message per recorded round
        (its ``tool_calls`` carrying positional ids ``call_r{j}_{i}``) and one
        ``tool`` message per recorded result, in the OpenAI wire shape
        :func:`_dispatch_message` already emits.
    """
    if round_index == 0:
        return build_messages(example, prompt_text)

    messages: list[dict[str, Any]] = [
        *(_dispatch_message(m) for m in (example.history or [])),
        {"role": "user", "content": prompt_text},
    ]
    expected_rounds = example.expected_tool_rounds or []
    fixture_rounds = example.tool_result_fixtures or []
    for j in range(min(round_index, len(expected_rounds), len(fixture_rounds))):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_r{j}_{i}",
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.arguments or {}),
                        },
                    }
                    for i, call in enumerate(expected_rounds[j])
                ],
            },
        )
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": f"call_r{j}_{i}",
                "content": _render_fixture(fixture),
            }
            for i, fixture in enumerate(fixture_rounds[j])
        )
    return messages


def _render_fixture(fixture: ToolResultFixture) -> str:
    """Render one recorded tool result as the ``content`` of a ``tool`` message.

    A recorded error is sent as ``{"error": ...}`` — the recorded agent saw the
    failure too, and its next round is the ground truth for what to do about
    it. A ``str`` result is sent verbatim (tools that already return text must
    not be double-encoded); anything else is JSON, with ``default=str`` so a
    stray non-serialisable value degrades to its repr instead of raising
    mid-run.
    """
    if fixture.error is not None:
        return json.dumps({"error": fixture.error})
    if isinstance(fixture.result, str):
        return fixture.result
    return json.dumps(fixture.result, ensure_ascii=False, default=str)


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
    samples_per_example: int = 1,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> RunResult:
    """Process every pending work item under the concurrency semaphore.

    ``samples_per_example`` decides whether the cache key carries the sample
    dimension: on a single-sample run it must not, so every existing cache
    entry stays valid; on a repeated-sampling run it must, or every sample
    after the first would be served from the first's cached response.
    """
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
                cache_sample_index=item.sample_index if samples_per_example > 1 else None,
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
    cache_sample_index: int | None = None,
) -> Call:
    """Cache-check → live call → record. Returns the constructed Call.

    ``cache_sample_index`` is the sample dimension of the cache key: ``None``
    on a single-sample run (keys unchanged), ``item.sample_index`` when
    ``defaults.samples_per_example > 1`` so each sample is its own live call.

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
        # None: this path is single-shot by construction. Only the tool path
        # replays rounds, and it bypasses the cache entirely (unchanged since
        # v0.2), so nothing passes a real round today — the key's round
        # dimension exists so tool-call caching can land without a migration.
        round_index=None,
        sample_index=cache_sample_index,
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
                sample_index=item.sample_index,
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
            sample_index=item.sample_index,
            error=str(exc),
        )

    call = Call(
        run_id=run_id,
        prompt_id=item.prompt.id,
        example_id=item.example.id,
        model_id=meta.id,
        role=item.role,
        sample_index=item.sample_index,
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
    """Tool-aware call path: teacher-forced replay loop + one Call per example.

    The example is replayed for :meth:`SuiteExample.rounds_to_replay` rounds —
    one for a single-shot example, else one per round its
    ``tool_result_fixtures`` cover plus the answer round after the last covered
    one. Round *k* is dispatched with the recorded rounds ``0..k-1`` fed back as
    assistant/tool turns (:func:`build_round_messages`); the candidate's own
    calls are never fed back. Every round's response is merged into ONE
    :class:`Call` (tokens/cost/latency summed, calls tagged with their
    ``round_index``), because one row per ``(prompt, example, role)`` is what
    resume, evaluate pairing, reports and the bundle all key on.

    ``messages`` carries round 0's dispatch as :func:`build_messages` computed
    it: set for multi-turn examples (``example.history`` is not ``None``), in
    which case round 0 goes through
    :meth:`ModelClient.complete_messages_with_tools`, and ``None`` for
    single-turn examples, which keeps the existing single-prompt
    :meth:`ModelClient.complete_with_tools` path so a single-shot example makes
    a byte-identical client call to before this loop existed. Rounds ``k >= 1``
    are always message-mode. The local cache is bypassed throughout — unchanged
    from before.
    """
    gen_temperature, gen_extra = translate_generation_config(item.example.generation_config)
    rounds = item.example.rounds_to_replay()

    merged_calls: list[ToolCall] = []
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    latency_ms = 0
    finish_reasons: list[str | None] = []
    raised_refusal = False
    refusal_text: str | None = None
    last_trace: ToolTrace | None = None

    for round_index in range(rounds):
        round_messages = (
            messages
            if round_index == 0
            else build_round_messages(item.example, prompt_text, round_index)
        )
        try:
            if round_messages is not None:
                result = await client.complete_messages_with_tools(
                    model=canonical_id,
                    messages=round_messages,
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
            # A partially replayed example is an unmeasured example: the rounds
            # that did complete are dropped, exactly as a failed single-shot
            # call records no trace. The round is named so the failure is
            # attributable without re-running; a single-shot call keeps the
            # bare provider error it always carried.
            return Call(
                run_id=run_id,
                prompt_id=item.prompt.id,
                example_id=item.example.id,
                model_id=canonical_id,
                role=item.role,
                sample_index=item.sample_index,
                error=f"round {round_index + 1}/{rounds}: {exc}" if rounds > 1 else str(exc),
            )

        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        cost_usd += result.cost_usd
        latency_ms += result.latency_ms
        finish_reasons.append(result.finish_reason)
        if result.trace.raised_refusal:
            raised_refusal = True
            if refusal_text is None:
                refusal_text = result.trace.refusal_text
        # sequence_index keeps counting across rounds: it stays unique per
        # trace, which ToolTrace validates. Snapshot the base before extending
        # — list.extend consumes a generator lazily, so len() must not be read
        # from inside it.
        base = len(merged_calls)
        merged_calls.extend(
            call.model_copy(
                update={"round_index": round_index, "sequence_index": base + offset},
            )
            for offset, call in enumerate(result.trace.calls)
        )
        last_trace = result.trace

    assert last_trace is not None  # rounds_to_replay() is always >= 1

    # A single-round replay keeps the client's trace object verbatim, so
    # nothing about a non-agentic-loop example changes shape.
    trace = (
        last_trace
        if rounds == 1
        else ToolTrace(
            calls=merged_calls,
            final_text=last_trace.final_text,
            raised_refusal=raised_refusal,
            refusal_text=refusal_text,
            round_count=rounds,
        )
    )
    # A truncated round poisons every comparison after it, so "length" anywhere
    # in the loop wins over the last round's own reason.
    finish_reason = "length" if "length" in finish_reasons else finish_reasons[-1]

    return Call(
        run_id=run_id,
        prompt_id=item.prompt.id,
        example_id=item.example.id,
        model_id=canonical_id,
        role=item.role,
        sample_index=item.sample_index,
        text=trace.final_text or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        trace=trace,
        finish_reason=finish_reason,
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
    "build_round_messages",
    "history_for_cache_key",
    "preflight_cost",
    "resolve_example_tools",
    "resolve_suite_tools",
    "run_orchestrator",
    "toolset_base_candidates",
]
