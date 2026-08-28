"""Evaluator protocol + result types.

Every evaluator scores a ``(source_output, target_output)`` pair and
returns a :class:`PairedScore` — both halves of the pair get a number
between 0 and 1, and the analysis layer (Phase 6) computes
``delta = target - source`` to detect regressions.

An evaluator that measured *nothing* on a pair returns ``None`` instead,
and the harness writes no row. Absence is a first-class answer: the
alternative is a fabricated score, and every fabricated score this
project ever wrote was a maximum one that read downstream as a perfect
result over a comparison that never happened.

Per-evaluator semantics:

* **Structural** (json-schema / regex / length): each output is scored
  independently. Source might still be valid even when target breaks.
* **Semantic similarity**: the score is a *target preservation* metric.
  Source is implicitly 1.0 (the source preserves itself); target gets
  the cosine similarity to source. Delta < 0 means the target drifted.
* **LLM-as-judge**: target wins → ``(0, 1)``; tie → ``(0.5, 0.5)``;
  source wins → ``(1, 0)``. Delta direction is the same as a regression
  classifier: negative = target lost.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PairedScore(_StrictModel):
    """Per-output scores from one evaluator on one (prompt, example) pair."""

    source_score: float = Field(ge=0.0, le=1.0)
    target_score: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def delta(self) -> float:
        """``target_score - source_score`` (negative = regression)."""
        return self.target_score - self.source_score


class EvalRecord(_StrictModel):
    """One row in ``scores.jsonl``: one evaluator's verdict on one pair.

    ``blocking`` mirrors the evaluator's config flag: records from advisory
    evaluators (``blocking=False``) are reported but never gate the migration
    verdict. Rows written before the flag existed load as blocking.

    ``kind`` is the evaluator's *type* slug, as opposed to ``evaluator_name``
    which is whatever the user called it in ``evalshift.yaml``. The analysis
    layer selects rows on ``kind`` — selecting on a name prefix meant renaming
    an evaluator silently unhooked it from its policy budget.
    """

    run_id: str
    prompt_id: str
    example_id: str
    evaluator_name: str
    #: Stable evaluator-type slug, independent of the user-chosen
    #: ``evaluator_name``. Empty on records checkpointed before it existed;
    #: the policy layer falls back to the legacy name prefix for those.
    kind: str = ""
    source_score: float = Field(ge=0.0, le=1.0)
    target_score: float = Field(ge=0.0, le=1.0)
    delta: float
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    blocking: bool = True


class EvaluatorError(Exception):
    """Raised when an evaluator can't be constructed or fails to score."""


@runtime_checkable
class Evaluator(Protocol):
    """Async scorer for a ``(source_output, target_output)`` pair.

    Implementations must set ``self.name`` to a stable identifier used
    in reports (e.g. ``"structural.json_schema"`` or
    ``"llm_judge.factuality"``) and ``kind`` to their type slug, which the
    analysis layer selects rows on (see :class:`EvalRecord`).
    """

    name: str
    kind: str

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore | None:
        """Score the pair. Raise :class:`EvaluatorError` when scoring breaks.

        Returns ``None`` when this evaluator measured *nothing* on this pair
        — two empty outputs leave a similarity metric and a judge with no
        text to compare. The harness writes no :class:`EvalRecord` at all for
        ``None``, because the only alternative a non-optional return allows
        is inventing a score, and an invented ``1.0`` reads downstream as a
        perfect result over a comparison that never happened.

        A ``None`` is not an error. Raise :class:`EvaluatorError` when the
        measurement *broke*: the harness converts a raise into an
        ``error``-stamped record that the analysis layer excludes, whereas
        returning a neutral score would silently count a broken measurement
        as "equivalent".

        Args:
            history: Multi-turn — the verbatim conversation prefix that
                preceded this turn, as plain ``{"role": ..., "content": ...}``
                dicts (deliberately not the ``suite`` models — the evaluator
                layer must stay decoupled from suite loading). ``None`` for
                single-turn examples.
        """
        ...


__all__ = [
    "EvalRecord",
    "Evaluator",
    "EvaluatorError",
    "PairedScore",
]
