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

CallRole = Literal["source", "target"]
RunStatus = Literal["in_progress", "completed", "failed"]


class _StrictModel(BaseModel):
    """Forbid extras + validate on assignment.

    Mirrors the same-named bases in :mod:`aimigrate.config.models` and
    :mod:`aimigrate.suite.models`. We keep these in parallel rather than
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


class RunState(_StrictModel):
    """Top-level state of an in-flight run.

    Persisted at ``.aimigrate/runs/<run_id>/state.json`` after every
    checkpoint. The Phase 4 orchestrator writes it; ``--resume`` reads
    it.

    Attributes:
        run_id: Stable identifier (``r_YYYYMMDD_<6hex>``).
        status: ``"in_progress"`` while running, ``"completed"`` on
            clean exit, ``"failed"`` on unrecoverable error.
        config_hash: SHA-256 of the canonicalised ``aimigrate.yaml`` +
            suite path. Resume aborts if this changes between attempts.
        started_at: UTC timestamp the run was first kicked off.
        last_checkpoint_at: UTC of the most recent state write, or
            ``None`` before the first checkpoint.
        models: Source and target canonical model ids.
        prompt_ids: Stable list of prompt ids in the run.
        suite_path: Original path to the JSONL suite, kept verbatim
            so reports can quote it back.
        total_evaluations: Total LLM calls implied by the run shape
            (``len(prompts) * len(examples) * 2`` models).
        completed_evaluations: Calls completed so far. Drives the
            progress bar and the resume "skip" filter.
    """

    run_id: str = Field(min_length=1)
    status: RunStatus = "in_progress"
    config_hash: str = Field(min_length=1)
    started_at: datetime
    last_checkpoint_at: datetime | None = None
    models: RunModels
    prompt_ids: list[str] = Field(min_length=1)
    suite_path: str
    total_evaluations: int = Field(ge=0)
    completed_evaluations: int = Field(default=0, ge=0)


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

    @property
    def succeeded(self) -> bool:
        """True if this call has no error attached."""
        return self.error is None


__all__ = [
    "Call",
    "CallRole",
    "RunModels",
    "RunState",
    "RunStatus",
]
