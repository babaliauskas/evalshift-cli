"""Pydantic models for run state and per-call records.

Two complementary types live here:

* :class:`RunState` — top-level state of an in-flight run. Persisted to
  ``state.json`` per the PDF §5.4 schema, used by the resume logic, and
  rendered in Rich progress UIs.
* :class:`Call` — one row of ``raw.jsonl``: a single LLM call's worth
  of data. The Phase 5 evaluators consume this stream and pair up the
  ``role="source"`` and ``role="target"`` rows for each
  ``(prompt_id, example_id)`` to produce evaluations.

We deliberately store one row per call (not per pair) because:

1. Resume logic stays simple — every crash leaves a coherent prefix of
   ``raw.jsonl``; we just skip what's already there.
2. The Phase 5 evaluators don't gain much from a pre-paired row — they
   need to walk every example anyway.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evalshift.evaluators.tool_models import ToolTrace

CallRole = Literal["source", "target"]
RunStatus = Literal["in_progress", "completed", "failed"]


class _StrictModel(BaseModel):
    """Forbid extras + validate on assignment.

    Mirrors the same-named bases in :mod:`evalshift.config.models` and
    :mod:`evalshift.suite.models`. We keep these in parallel rather than
    extracting a single shared base because the diverging defaults are
    likely (e.g. config wants strict, but ``Call`` may eventually allow
    extra provider-specific metadata).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunModels(_StrictModel):
    """Source/target model pair for a run.

    Stored under the ``models`` key in ``state.json`` for symmetry with
    the PDF §5.4 schema example.
    """

    source: str = Field(min_length=1, description="Canonical id of the source model.")
    target: str = Field(min_length=1, description="Canonical id of the target model.")


class UnmeasuredPair(_StrictModel):
    """One (prompt, example) pair an evaluator was handed but did not score.

    Attributes:
        prompt_id: Prompt the pair was produced under.
        example_id: Suite example the pair belongs to.
    """

    prompt_id: str
    example_id: str


class EvaluatorCoverage(_StrictModel):
    """What one evaluator *axis* was asked to score, and what it scored.

    One entry per ``(evaluator_name, kind)``, not per evaluator:
    ``tool_selection`` scores conformance and divergence independently and
    writes a row for each, so a single tally could report ``recorded``
    above ``attempted`` and would hide an axis that measured nothing behind
    one that measured everything.

    An evaluator that measures nothing on a pair writes no row, so
    ``scores.jsonl`` no longer carries any trace of the attempt. Without
    this, an evaluator that measured 2 of 10 pairs would report a rate over
    a denominator of 2 with the other 8 invisible, and one that measured
    *none* would vanish from the analysis entirely — taking its
    ``severity: insufficient`` verdict with it, which is the only thing
    standing between a no-measurement run and a silent pass.

    Attributes:
        evaluator_name: The user-chosen name, matching ``EvalRecord``'s.
        kind: The axis's type slug (see :class:`EvalRecord.kind`). Together
            with ``evaluator_name`` it is what the analysis layer groups
            comparisons by, so coverage lines up with the rows it accounts
            for.
        attempted: Pairs this axis was handed.
        recorded: Pairs that produced a row in ``scores.jsonl``, errored
            rows included — an error is a broken measurement, not an absent
            one. ``attempted - recorded`` is the not-applicable count.
        unmeasured: The attempted pairs that produced no row, named
            individually so the analysis layer can attribute them to the
            same slices a real row would have landed in. Empty on a healthy
            run, so this costs nothing until something is wrong.
        blocking: The evaluator's config ``blocking`` flag. Every scored row
            carries it, but an axis that measured *nothing* has no rows —
            this is then the only place the flag survives, and without it
            the policy layer named silent ``blocking: false`` evaluators as
            blind gates. Defaults ``True`` so a ``state.json`` written
            before the field existed reads as gating — over-warning is the
            conservative side.
    """

    evaluator_name: str
    kind: str = ""
    attempted: int = Field(ge=0)
    recorded: int = Field(ge=0)
    unmeasured: list[UnmeasuredPair] = Field(default_factory=list)
    blocking: bool = True


class RunState(_StrictModel):
    """Top-level state of an in-flight run.

    Persisted at ``.evalshift/runs/<run_id>/state.json`` after every
    checkpoint. The Phase 4 orchestrator writes it; ``--resume`` reads
    it.

    Attributes:
        run_id: Stable identifier (``r_YYYYMMDD_<6hex>``).
        status: ``"in_progress"`` while running, ``"completed"`` on
            clean exit, ``"failed"`` on unrecoverable error.
        config_hash: SHA-256 of the canonicalised ``evalshift.yaml`` +
            suite path. Resume aborts if this changes between attempts.
        started_at: UTC timestamp the run was first kicked off.
        last_checkpoint_at: UTC of the most recent state write, or
            ``None`` before the first checkpoint.
        models: Source and target canonical model ids.
        prompt_ids: Stable list of prompt ids in the run.
        suite_path: Original path to the JSONL suite, kept verbatim
            so reports can quote it back.
        suite_name: The ``suites:`` key the run was launched with
            (``--suite-name``), or ``None`` for a raw ``--suite <path>`` run.
            Recorded because evaluate/report/bundle all run *after* the CLI
            invocation is gone and each needs to resolve the same per-suite
            evaluator set via
            :meth:`~evalshift.config.models.EvalShiftConfig.evaluators_for` —
            the run must remember which suite it was, not just where the file
            sat.
        total_evaluations: Total LLM calls implied by the run shape
            (``len(prompts) * len(examples) * 2`` models).
        completed_evaluations: Calls completed so far. Drives the
            progress bar and the resume "skip" filter.
        non_deterministic_models: Canonical ids of models in this run that
            do not honour ``temperature``, so their outputs vary between
            identical calls. Recorded at run start rather than recomputed
            later: a bundle is immutable, and a LiteLLM upgrade between the
            run and a subsequent ``evalshift report`` must not rewrite what
            was true when the calls were made. Empty for every run where
            both arms sample deterministically.
        evaluator_coverage: Per-evaluator attempted-vs-recorded counts,
            written by the ``evaluate`` stage rather than the orchestrator —
            it is the one piece of run-level state only scoring knows. Empty
            until the run has been scored. See :class:`EvaluatorCoverage`.
    """

    run_id: str = Field(min_length=1)
    status: RunStatus = "in_progress"
    config_hash: str = Field(min_length=1)
    started_at: datetime
    last_checkpoint_at: datetime | None = None
    models: RunModels
    prompt_ids: list[str] = Field(min_length=1)
    suite_path: str
    # Defaulted so state.json files written before this field existed still
    # load under extra="forbid" and --resume keeps working across upgrades.
    suite_name: str | None = None
    total_evaluations: int = Field(ge=0)
    completed_evaluations: int = Field(default=0, ge=0)
    # Defaulted so state.json files written before this field existed still
    # load under extra="forbid" and --resume keeps working across upgrades.
    non_deterministic_models: list[str] = Field(default_factory=list)
    # Written by `evalshift evaluate`, which rewrites state.json once scoring
    # finishes. Defaulted because every state.json is written by the
    # orchestrator first, long before any evaluator has run.
    evaluator_coverage: list[EvaluatorCoverage] = Field(default_factory=list)


class Call(_StrictModel):
    """One row of ``raw.jsonl`` — a single completed LLM call.

    Successful calls have ``error=None`` and a populated ``text``.
    Failed calls record the exception message in ``error`` and leave
    ``text`` empty. Either way the call counts as "done" for resume
    purposes — we don't retry failed calls automatically across resumes,
    because the most common cause of a failure is a deterministic
    config/key issue that wouldn't be fixed by re-running.

    Attributes:
        run_id: Owning run.
        prompt_id: Prompt this call belongs to.
        example_id: Suite example whose inputs were rendered.
        model_id: Canonical id of the model that was called.
        role: ``"source"`` or ``"target"``.
        text: The model's response text. Empty string on error.
        input_tokens / output_tokens: From the provider response.
        cost_usd: Per-call cost (``litellm.completion_cost``); ``0.0``
            when the model isn't priced.
        latency_ms: Wall time of the live call. ``0`` for cache hits
            after the first run (we keep the *original* latency).
        cached: ``True`` if the response came from the local cache.
        error: ``None`` on success; the stringified error on failure.
        finish_reason: The provider's normalised stop reason. ``"length"``
            means the output was truncated by the ``max_tokens`` cap;
            :meth:`truncated` reports this. Truncated calls are excluded
            from the paired regression statistics (via the evaluator's
            error path) so a cut-off output can't manufacture a false
            regression.
    """

    run_id: str
    prompt_id: str
    example_id: str
    model_id: str
    role: CallRole
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    error: str | None = None
    # v0.2 — populated only for tool-aware calls; ``None`` for plain text
    # ones. The orchestrator switches between ``ModelClient.complete`` and
    # ``complete_with_tools`` per call, based on whether the dispatched
    # example's own resolved toolset (see
    # ``runner.orchestrator.resolve_example_tools``) is non-empty — not on
    # anything the owning prompt declares.
    trace: ToolTrace | None = None
    # Optional (defaulted) so pre-existing raw.jsonl lines still validate
    # on resume. ``"length"`` flags a token-cap truncation.
    finish_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        """True if this call has no error attached."""
        return self.error is None

    @property
    def truncated(self) -> bool:
        """True when the provider cut the output off at the token cap."""
        return self.finish_reason == "length"


__all__ = [
    "Call",
    "CallRole",
    "EvaluatorCoverage",
    "RunModels",
    "RunState",
    "RunStatus",
    "UnmeasuredPair",
]
