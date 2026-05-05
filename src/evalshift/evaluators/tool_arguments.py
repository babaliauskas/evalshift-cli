"""Evaluator: did the target pass similar arguments to matched tool calls?

Greedy match by ``(tool_name, nearest sequence_index)`` — when both
source and target call the same tool multiple times we line them up by
position. Hungarian-algorithm matching is opt-in for v0.3 (PRD risk #3).

Per-field strategies:

* ``exact`` — string equality of repr-stringified values.
* ``subset`` — recursive structural subset (dict/list aware).
* ``numeric`` — relative-error decay clamped by ``numeric_tolerance``.
* ``semantic`` — cosine similarity of embedded values via an injected
  ``embeddings_fn``. Falls back to ``exact`` when ``embeddings_fn`` is
  ``None`` (typical in unit tests).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any

from evalshift.config.models import ToolArgumentsEvaluatorConfig
from evalshift.evaluators.base import EvalRecord, PairedScore
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.suite.models import SuiteExample

# Type for the embeddings function: (a, b) -> cosine similarity in [0, 1].
EmbeddingsFn = Callable[[str, str], Awaitable[float]]


class ToolArgumentsEvaluator:
    """Score how similar the target's tool arguments are to the source's."""

    def __init__(
        self,
        config: ToolArgumentsEvaluatorConfig,
        *,
        embeddings_fn: EmbeddingsFn | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self._embeddings_fn = embeddings_fn

    async def score_pair(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        matched = _match_calls(source_trace, target_trace)
        if not matched:
            # Either side has no calls of any matched tool name → no pairs
            # to compare. Treat as 1.0 if both empty, 0.0 if target made
            # uncomparable calls (likely a regression vs. source).
            target_score = 1.0 if target_trace.call_count == 0 else 0.0
            return EvalRecord(
                run_id=run_id,
                prompt_id=prompt_id,
                example_id=example.id,
                evaluator_name=self.name,
                source_score=1.0,
                target_score=target_score,
                delta=target_score - 1.0,
                metadata={"reason": "no matched calls between source and target"},
            )

        per_call: list[float] = []
        per_call_meta: list[dict[str, Any]] = []
        for src_call, tgt_call in matched:
            score, detail = await self._score_call_args(src_call, tgt_call)
            per_call.append(score)
            per_call_meta.append(detail)

        target_score = sum(per_call) / len(per_call)
        paired = PairedScore(
            source_score=1.0,
            target_score=target_score,
            metadata={"per_call": per_call_meta},
        )
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            evaluator_name=self.name,
            source_score=paired.source_score,
            target_score=paired.target_score,
            delta=paired.delta,
            metadata=paired.metadata,
        )

    # ------------------------------------------------------------------
    # Per-call / per-field scoring
    # ------------------------------------------------------------------

    async def _score_call_args(
        self,
        src: ToolCall,
        tgt: ToolCall,
    ) -> tuple[float, dict[str, Any]]:
        all_keys = set(src.arguments) | set(tgt.arguments)
        if not all_keys:
            return 1.0, {"tool_name": src.tool_name, "field_scores": {}}

        # Drop sentinel keys we use to record parser-level errors so they
        # don't pollute the comparison.
        all_keys = {k for k in all_keys if not k.startswith("_")}
        if not all_keys:
            return 0.0, {
                "tool_name": src.tool_name,
                "field_scores": {},
                "_parse_error": True,
            }

        field_scores: dict[str, float] = {}
        for key in all_keys:
            strategy = self.config.strategies.get(key, "exact")
            field_scores[key] = await self._score_field(
                src.arguments.get(key),
                tgt.arguments.get(key),
                strategy,
            )
        avg = sum(field_scores.values()) / len(field_scores)
        return avg, {
            "tool_name": src.tool_name,
            "field_scores": field_scores,
        }

    async def _score_field(self, src_val: Any, tgt_val: Any, strategy: str) -> float:
        if src_val is None and tgt_val is None:
            return 1.0
        if src_val is None or tgt_val is None:
            return 0.0
        if strategy == "exact":
            return 1.0 if src_val == tgt_val else 0.0
        if strategy == "subset":
            return 1.0 if _is_subset(src_val, tgt_val) else 0.0
        if strategy == "numeric":
            return self._score_numeric(src_val, tgt_val)
        if strategy == "semantic":
            return await self._score_semantic(src_val, tgt_val)
        # Unknown strategy: fall back to exact.
        return 1.0 if src_val == tgt_val else 0.0

    def _score_numeric(self, src_val: Any, tgt_val: Any) -> float:
        try:
            s = float(src_val)
            t = float(tgt_val)
        except TypeError, ValueError:
            return 0.0
        if s == 0 and t == 0:
            return 1.0
        denom = max(abs(s), abs(t), 1e-9)
        rel = abs(s - t) / denom
        tolerance = max(self.config.numeric_tolerance, 1e-9)
        # Linear decay: 1.0 at zero error, 0.0 once error == tolerance.
        return max(0.0, 1.0 - rel / tolerance)

    async def _score_semantic(self, src_val: Any, tgt_val: Any) -> float:
        if self._embeddings_fn is None:
            return 1.0 if str(src_val) == str(tgt_val) else 0.0
        try:
            sim = await self._embeddings_fn(str(src_val), str(tgt_val))
        except Exception:
            return 1.0 if str(src_val) == str(tgt_val) else 0.0
        if math.isnan(sim):
            return 0.0
        return max(0.0, min(1.0, float(sim)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_calls(
    source: ToolTrace,
    target: ToolTrace,
) -> list[tuple[ToolCall, ToolCall]]:
    """Greedy nearest-index match between same-named source and target calls.

    Documented v0.2 simplification — works perfectly when each tool
    appears at most once and degrades gracefully with repeats. Hungarian
    matching deferred to v0.3 (PRD risk #3).
    """
    matched: list[tuple[ToolCall, ToolCall]] = []
    used: set[int] = set()
    for src in source.calls:
        best: tuple[int, ToolCall] | None = None
        best_distance = math.inf
        for i, tgt in enumerate(target.calls):
            if i in used or tgt.tool_name != src.tool_name:
                continue
            distance = abs(tgt.sequence_index - src.sequence_index)
            if distance < best_distance:
                best_distance = distance
                best = (i, tgt)
        if best is not None:
            used.add(best[0])
            matched.append((src, best[1]))
    return matched


def _is_subset(a: Any, b: Any) -> bool:
    """Structural subset: every leaf in ``a`` is present (== or contains) in ``b``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and _is_subset(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return all(any(_is_subset(item, x) for x in b) for item in a)
    return bool(a == b)


__all__ = ["EmbeddingsFn", "ToolArgumentsEvaluator"]
