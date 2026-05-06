"""Evaluator protocol + result types.

Every evaluator scores a ``(source_output, target_output)`` pair and
returns a :class:`PairedScore` — both halves of the pair get a number
between 0 and 1, and the analysis layer (Phase 6) computes
``delta = target - source`` to detect regressions.

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
    """One row in ``scores.jsonl``: one evaluator's verdict on one pair."""

    run_id: str
    prompt_id: str
    example_id: str
    evaluator_name: str
    source_score: float = Field(ge=0.0, le=1.0)
    target_score: float = Field(ge=0.0, le=1.0)
    delta: float
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluatorError(Exception):
    """Raised when an evaluator can't be constructed or fails to score."""


@runtime_checkable
class Evaluator(Protocol):
    """Async scorer for a ``(source_output, target_output)`` pair.

    Implementations must set ``self.name`` to a stable identifier used
    in reports (e.g. ``"structural.json_schema"`` or
    ``"llm_judge.factuality"``).
    """

    name: str

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
    ) -> PairedScore:
        """Score the pair. Should never raise — return a 0/0 record on failure."""
        ...


__all__ = [
    "EvalRecord",
    "Evaluator",
    "EvaluatorError",
    "PairedScore",
]
