"""Evaluator: two independent questions about the tools a model called.

A migration asks both at once, and they are not the same question:

* **Conformance** (``tool_selection.conformance``) — did each side match
  the recorded ground truth? Each side is graded *absolutely*, against
  ``example.expected_tools`` (or against ``expected_no_tools``), so both
  can fail at once and the delta stays 0. That is correct: the migration
  did not cause a failure both models share.
* **Divergence** (``tool_selection.divergence``) — did the target do what
  the source did? Source is its own baseline at 1.0, so a target that
  behaves differently lands a negative delta — a regression.

Both are emitted for every pair. They used to be modes of one switch, and
the ground-truth branch won unconditionally: ``capture sync`` marks every
promoted row ``expected_no_tools``, so a suite built from captures took
that branch on every example, graded both sides absolutely, and reported
``0.0 / 0.0`` — a zero delta, filed as *equivalent* — for pairs where the
two models called entirely different tools.

Conformance strategies (``expected``, ``expected_set``) and divergence
strategies (``exact``, ``set``, ``first``) are configured independently;
either axis can be turned ``off``. See
:class:`~evalshift.config.models.ToolSelectionEvaluatorConfig`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from evalshift.config.models import ToolSelectionEvaluatorConfig
from evalshift.evaluators.base import EvalRecord, PairedScore
from evalshift.evaluators.failures import TOOL_GROUND_TRUTH_MISS, TOOL_SELECTION_DRIFT
from evalshift.evaluators.tool_models import ToolTrace
from evalshift.suite.models import SuiteExample

#: Family slug. Not stamped on any record — the two axes below are what
#: rows carry — but it names the evaluator family in config and reports.
KIND = "tool_selection"

#: Stable per-axis slugs stamped onto records. The analysis layer selects
#: policy rows on these, never on the user-chosen name, and keeps the two
#: axes in separate comparisons: they are different measurements against
#: different baselines and averaging them together restates the same bug.
KIND_CONFORMANCE = f"{KIND}.conformance"
KIND_DIVERGENCE = f"{KIND}.divergence"


class ToolSelectionEvaluator:
    """Score whether the right tools were called."""

    kind = KIND

    def __init__(self, config: ToolSelectionEvaluatorConfig) -> None:
        self.config = config
        self.name = config.name

    @property
    def kinds(self) -> tuple[str, ...]:
        """The axis slugs this evaluator attempts on every pair.

        The scoring stage reads this to book one coverage attempt per axis:
        "k of n pairs were not measurable" is a per-axis statement, and an
        axis that is configured ``off`` was never attempted at all.
        """
        out: list[str] = []
        if self.config.conformance != "off":
            out.append(KIND_CONFORMANCE)
        if self.config.divergence != "off":
            out.append(KIND_DIVERGENCE)
        return tuple(out)

    async def score_pair(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> list[EvalRecord]:
        """Score a (source, target) trace pair on every configured axis.

        Returns one record per axis that measured something — so zero, one
        or two. An axis that is ``off``, or that had no ground truth to
        compare against, contributes no row rather than an invented score.
        """
        records = []
        for record in (
            self._score_conformance(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            ),
            self._score_divergence(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            ),
        ):
            if record is not None:
                records.append(record)
        return records

    # ------------------------------------------------------------------
    # Axis dispatch
    # ------------------------------------------------------------------

    def _score_conformance(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        """Grade both sides against the example's recorded ground truth.

        ``expected_no_tools`` is this axis's *input*, not a control-flow
        branch over the whole evaluator: "the recording called nothing" is
        simply the expectation to grade against, under either strategy.
        """
        strategy = self.config.conformance
        if strategy == "off":
            return None
        if example.expected_no_tools:
            return self._score_no_tools(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        if strategy == "expected":
            return self._score_expected(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        # strategy == "expected_set"
        return self._score_expected_set(
            run_id=run_id,
            prompt_id=prompt_id,
            example=example,
            source_trace=source_trace,
            target_trace=target_trace,
        )

    def _score_divergence(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        """Compare the target to the source, which is its own baseline."""
        strategy = self.config.divergence
        if strategy == "off":
            return None
        if strategy == "exact":
            return self._score_exact(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        if strategy == "set":
            return self._score_set(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )
        # strategy == "first"
        return self._score_first(
            run_id=run_id,
            prompt_id=prompt_id,
            example=example,
            source_trace=source_trace,
            target_trace=target_trace,
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _score_no_tools(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        target_score = 1.0 if target_trace.call_count == 0 else 0.0
        source_score = 1.0 if source_trace.call_count == 0 else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_CONFORMANCE,
            paired=PairedScore(
                source_score=source_score,
                target_score=target_score,
                metadata={
                    "mode": "expected_no_tools",
                    "source_calls": source_trace.call_count,
                    "target_calls": target_trace.call_count,
                },
            ),
        )

    def _score_expected(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        if not example.expected_tools:
            # No ground truth to conform to, so nothing was measured. The
            # 1.0/1.0 this used to write claimed full conformance against an
            # expectation that does not exist.
            return None

        expected_names = [e.tool_name for e in example.expected_tools]
        source_score = _sequence_match(expected_names, source_trace.tool_names)
        target_score = _sequence_match(expected_names, target_trace.tool_names)
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_CONFORMANCE,
            paired=PairedScore(
                source_score=source_score,
                target_score=target_score,
                metadata={
                    "mode": "expected",
                    "expected_names": expected_names,
                    "source_names": source_trace.tool_names,
                    "target_names": target_trace.tool_names,
                },
            ),
        )

    def _score_expected_set(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        """Order-insensitive variant of :meth:`_score_expected`.

        Scores multiset recall of the expected tool names. Call order is
        ignored — a parallel fan-out that a model emits in a different
        sequence is the same behaviour, not a regression. Extra calls the
        model made beyond the expectation neither help nor hurt; the
        trace-structure evaluator is what polices call counts.

        Returns ``None`` when the example carries no expected tools — see
        :meth:`_score_expected`.
        """
        if not example.expected_tools:
            return None

        expected_names = [e.tool_name for e in example.expected_tools]
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_CONFORMANCE,
            paired=PairedScore(
                source_score=_multiset_match(expected_names, source_trace.tool_names),
                target_score=_multiset_match(expected_names, target_trace.tool_names),
                metadata={
                    "mode": "expected_set",
                    "expected_names": expected_names,
                    "source_names": source_trace.tool_names,
                    "target_names": target_trace.tool_names,
                },
            ),
        )

    def _score_exact(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        # Source-vs-itself = 1.0 by definition; target must match source's sequence.
        target_score = 1.0 if target_trace.tool_names == source_trace.tool_names else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_DIVERGENCE,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "exact",
                    "source_names": source_trace.tool_names,
                    "target_names": target_trace.tool_names,
                },
            ),
        )

    def _score_set(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        source_set = source_trace.tool_name_set
        target_set = target_trace.tool_name_set
        target_score = _jaccard(source_set, target_set)
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_DIVERGENCE,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "set",
                    "source_set": sorted(source_set),
                    "target_set": sorted(target_set),
                },
            ),
        )

    def _score_first(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        src_first = source_trace.tool_names[0] if source_trace.calls else None
        tgt_first = target_trace.tool_names[0] if target_trace.calls else None
        # Source is always its own baseline at 1.0.
        target_score = 1.0 if src_first is not None and tgt_first == src_first else 0.0
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            kind=KIND_DIVERGENCE,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "mode": "first",
                    "source_first": src_first,
                    "target_first": tgt_first,
                },
            ),
        )

    def _record(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example_id: str,
        kind: str,
        paired: PairedScore,
    ) -> EvalRecord:
        """Build one axis's record, tagging what went wrong on it.

        Two different failures live here and they must not share a label:

        * a negative delta is drift — the target lost ground the source
          held. On the divergence axis, whose source is 1.0 by
          construction, that is any target below 1.0;
        * both sides missing the ground truth is not a migration finding
          at all. Ground truth captured from the source model that the
          source model then fails means the harness is misconfigured, and
          ``0.0 / 0.0`` reads as *equivalent* to every downstream rate, so
          it needs a category of its own to be visible.
        """
        meta: dict[str, Any] = dict(paired.metadata)
        if self.config.severity_floor:
            meta["severity_floor"] = self.config.severity_floor
        categories: list[str] = []
        if paired.target_score < paired.source_score:
            categories.append(TOOL_SELECTION_DRIFT)
        if kind == KIND_CONFORMANCE and paired.source_score < 1.0 and paired.target_score < 1.0:
            categories.append(TOOL_GROUND_TRUTH_MISS)
        if categories:
            meta["failure_categories"] = categories
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example_id,
            evaluator_name=self.name,
            kind=kind,
            source_score=paired.source_score,
            target_score=paired.target_score,
            delta=paired.delta,
            explanation=paired.explanation,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sequence_match(expected: list[str], actual: list[str]) -> float:
    """How many ``expected`` names appear in ``actual`` in order, normalised."""
    if not expected:
        return 1.0
    i = j = matched = 0
    while i < len(expected) and j < len(actual):
        if expected[i] == actual[j]:
            matched += 1
            i += 1
            j += 1
        else:
            j += 1
    return matched / len(expected)


def _multiset_match(expected: list[str], actual: list[str]) -> float:
    """Fraction of ``expected`` names present in ``actual``, ignoring order.

    Duplicates count: two expected ``archive_project`` calls need two actual
    ``archive_project`` calls for a full score. Extra actual calls are
    ignored.
    """
    if not expected:
        return 1.0
    remaining = Counter(actual)
    matched = 0
    for name in expected:
        if remaining[name] > 0:
            remaining[name] -= 1
            matched += 1
    return matched / len(expected)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity. Returns 1.0 if both sets empty, else intersection/union."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


__all__ = ["KIND_CONFORMANCE", "KIND_DIVERGENCE", "ToolSelectionEvaluator"]
