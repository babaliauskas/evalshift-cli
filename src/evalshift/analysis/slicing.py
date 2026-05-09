"""Group evaluation records into slices for analysis.

A *slice* is a named subset of suite examples — typically defined by a
tag in ``evalshift.yaml``'s ``slices:`` block. The implicit ``"all"``
slice always exists and contains every example.

The output of :func:`build_slices` is a mapping from slice name to a
list of ``(prompt_id, evaluator_name, example_id, source_score,
target_score, delta)`` tuples that the statistics layer can pair up and
test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from evalshift.evaluators.base import EvalRecord
from evalshift.suite.models import Suite

ALL_SLICE: str = "all"


@dataclass(frozen=True, slots=True)
class SlicedScore:
    """One paired score belonging to a slice."""

    prompt_id: str
    evaluator_name: str
    example_id: str
    source_score: float
    target_score: float
    delta: float


@dataclass(frozen=True, slots=True)
class SliceAggregate:
    """Per-slice aggregate stats (computed without significance testing)."""

    name: str
    n: int
    source_mean: float
    target_mean: float
    delta_mean: float
    delta_min: float
    delta_max: float
    source_std: float
    target_std: float


def build_slices(
    *,
    records: list[EvalRecord],
    suite: Suite,
    tag_to_slice: dict[str, str] | None = None,
) -> dict[str, list[SlicedScore]]:
    """Group evaluation records into slices keyed by slice name.

    Args:
        records: Every :class:`EvalRecord` from ``scores.jsonl``.
        suite: The loaded suite, used to look up an example's ``tags``.
        tag_to_slice: Mapping from a configured slice's tag (the value
            of ``filter`` in MVP, simplified to a literal tag) to the
            slice name surfaced in reports. ``None`` falls back to a
            tag-name == slice-name identity mapping.

    Returns:
        ``{slice_name: [SlicedScore, ...]}`` always containing at
        least the implicit ``"all"`` slice.
    """
    by_id = {ex.id: ex for ex in suite.examples}
    out: dict[str, list[SlicedScore]] = defaultdict(list)

    for rec in records:
        # Skip records that errored at the evaluator layer — they don't
        # carry meaningful scores.
        if rec.error is not None:
            continue
        sliced = SlicedScore(
            prompt_id=rec.prompt_id,
            evaluator_name=rec.evaluator_name,
            example_id=rec.example_id,
            source_score=rec.source_score,
            target_score=rec.target_score,
            delta=rec.delta,
        )
        out[ALL_SLICE].append(sliced)
        example = by_id.get(rec.example_id)
        if example is None:
            continue
        for tag in example.tags:
            slice_name = tag_to_slice.get(tag, tag) if tag_to_slice else tag
            out[slice_name].append(sliced)

    return dict(out)


def aggregates(
    sliced: list[SlicedScore],
    name: str,
) -> SliceAggregate:
    """Compute simple aggregates over a list of paired scores."""
    n = len(sliced)
    if n == 0:
        return SliceAggregate(
            name=name,
            n=0,
            source_mean=0.0,
            target_mean=0.0,
            delta_mean=0.0,
            delta_min=0.0,
            delta_max=0.0,
            source_std=0.0,
            target_std=0.0,
        )
    source = [s.source_score for s in sliced]
    target = [s.target_score for s in sliced]
    deltas = [s.delta for s in sliced]
    return SliceAggregate(
        name=name,
        n=n,
        source_mean=_mean(source),
        target_mean=_mean(target),
        delta_mean=_mean(deltas),
        delta_min=min(deltas),
        delta_max=max(deltas),
        source_std=_std(source),
        target_std=_std(target),
    )


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return float((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)


__all__ = ["ALL_SLICE", "SliceAggregate", "SlicedScore", "aggregates", "build_slices"]
