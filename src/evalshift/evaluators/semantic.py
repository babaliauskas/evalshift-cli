"""Semantic-similarity evaluator using LLM-provider embeddings.

We treat similarity as a *target preservation* score: source is 1.0 by
definition (it preserves itself), and target gets its cosine similarity
to source. ``delta = target - source`` is then in [-2, 0] and any
negative number indicates drift.

This deliberately doesn't make symmetric "both halves same score" pairs
because that would always produce delta=0 and defeat the regression
detection.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import litellm

from evalshift.evaluators.base import PairedScore
from evalshift.evaluators.failures import SEMANTIC_REGRESSION

log = logging.getLogger(__name__)


class CosineSimilarityEvaluator:
    """Cosine similarity between source and target output embeddings."""

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-small",
        name: str = "semantic.cosine",
    ) -> None:
        self.name = name
        self.embedding_model = embedding_model

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
    ) -> PairedScore:
        try:
            source_emb = await self._embed(source_output)
            target_emb = await self._embed(target_output)
        except Exception as exc:
            log.warning("embedding failed for %s/%s: %s", prompt_id, example_id, exc)
            return PairedScore(
                source_score=0.0,
                target_score=0.0,
                explanation=f"embedding failed: {exc}",
            )

        sim = _cosine(source_emb, target_emb)
        # Clamp to [0,1] so the pydantic constraint holds; cosine can
        # be slightly negative or above 1 due to float drift.
        sim_clamped = max(0.0, min(1.0, sim))
        return PairedScore(
            source_score=1.0,  # source preserves itself by definition
            target_score=sim_clamped,
            metadata={
                "raw_cosine": sim,
                "failure_categories": [SEMANTIC_REGRESSION] if sim_clamped < 1.0 else [],
            },
        )

    async def _embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=self.embedding_model,
            input=[text],
        )
        # LiteLLM returns either dict-like or object-like depending on
        # provider; handle both.
        data = response.data if hasattr(response, "data") else response["data"]
        first = data[0]
        embedding = first["embedding"] if isinstance(first, dict) else first.embedding
        return list(embedding)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, returning 0 for any degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


__all__ = ["CosineSimilarityEvaluator"]
