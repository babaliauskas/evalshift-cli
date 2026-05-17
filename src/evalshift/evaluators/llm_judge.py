"""LLM-as-judge evaluator: pairwise comparison via a prompted model."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from evalshift.evaluators.base import PairedScore
from evalshift.models.client import ModelClient

log = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge comparing two AI assistant outputs against a criterion.

Criterion: {criterion_prompt}

Output A:
\"\"\"
{output_a}
\"\"\"

Output B:
\"\"\"
{output_b}
\"\"\"

Decide which output better satisfies the criterion. Reply with strict JSON only,
no prose, on a single line:
{{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}
"""


class PairwiseJudgeEvaluator:
    """Ask a model which of two outputs better satisfies a criterion.

    Order randomization (target sometimes shown as A, sometimes as B)
    mitigates positional bias common in LLM judges. The mapping back to
    target-vs-source happens after the verdict so the caller never sees
    the randomization.
    """

    def __init__(
        self,
        *,
        criterion_name: str,
        criterion_prompt: str,
        judge_model: str = "gemini/gemini-3.1-flash-lite-preview",
        client: ModelClient | None = None,
        rng: random.Random | None = None,
        name: str | None = None,
    ) -> None:
        self.criterion_name = criterion_name
        self.criterion_prompt = criterion_prompt
        self.judge_model = judge_model
        self.name = name or f"llm_judge.{criterion_name}"
        self._client = client or ModelClient()
        self._rng = rng or random.Random()

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
    ) -> PairedScore:
        target_is_a = self._rng.random() < 0.5
        if target_is_a:
            output_a, output_b = target_output, source_output
        else:
            output_a, output_b = source_output, target_output

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            criterion_prompt=self.criterion_prompt,
            output_a=output_a,
            output_b=output_b,
        )

        try:
            response = await self._client.complete(
                model=self.judge_model,
                prompt=judge_prompt,
                temperature=0.0,
            )
            verdict = _parse_verdict(response.text)
        except Exception as exc:
            log.warning("judge call failed for %s/%s: %s", prompt_id, example_id, exc)
            return PairedScore(
                source_score=0.5,
                target_score=0.5,
                explanation=f"judge failed: {exc}",
                metadata={"target_was_a": target_is_a, "verdict": "error"},
            )

        if verdict == "tie":
            return PairedScore(
                source_score=0.5,
                target_score=0.5,
                explanation="tie",
                metadata={"target_was_a": target_is_a, "verdict": "tie"},
            )

        a_won = verdict == "A"
        target_won = a_won == target_is_a
        return PairedScore(
            source_score=0.0 if target_won else 1.0,
            target_score=1.0 if target_won else 0.0,
            explanation=f"target {'won' if target_won else 'lost'}",
            metadata={"target_was_a": target_is_a, "verdict": verdict},
        )


def _parse_verdict(text: str) -> str:
    """Pull a winner label out of the judge's response.

    Tolerant: tries strict JSON first, then a regex JSON-object scan,
    then a final keyword fallback.
    """
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(
                f"no JSON object found in judge response: {stripped[:200]!r}"
            ) from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"judge response contained malformed JSON: {match.group(0)[:200]!r}"
            ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"judge response wasn't a JSON object: {data!r}")

    winner = str(data.get("winner", "")).strip().upper()
    if winner == "A":
        return "A"
    if winner == "B":
        return "B"
    if winner in ("TIE", "DRAW", "EQUAL"):
        return "tie"
    raise ValueError(f"invalid winner value: {winner!r}")


__all__ = ["PairwiseJudgeEvaluator"]
