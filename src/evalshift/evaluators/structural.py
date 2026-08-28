"""Structural evaluators: deterministic, fast, no API calls."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError

from evalshift.evaluators.base import EvaluatorError, PairedScore
from evalshift.evaluators.failures import FORMAT_FAILURE


class JsonSchemaEvaluator:
    """Validate each output against a JSON schema. 1.0 if valid, 0.0 otherwise.

    Outputs that don't even parse as JSON score 0.0.
    """

    #: Stable evaluator-type slug the analysis layer selects policy rows
    #: on — never the user-chosen ``name``, which is free to change.
    kind = "structural"

    def __init__(
        self,
        *,
        schema_path: str | Path,
        name: str = "structural.json_schema",
    ) -> None:
        self.name = name
        try:
            text = Path(schema_path).read_text(encoding="utf-8")
            schema = json.loads(text)
            Draft7Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, SchemaError) as exc:
            raise EvaluatorError(f"failed to load schema {schema_path!r}: {exc}") from exc
        self._validator = Draft7Validator(schema)

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore:
        source_score = self._validate_one(source_output)
        target_score = self._validate_one(target_output)
        return PairedScore(
            source_score=source_score,
            target_score=target_score,
            metadata=_failure_metadata(source_score, target_score, FORMAT_FAILURE),
        )

    def _validate_one(self, output: str) -> float:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return 0.0
        return 1.0 if self._validator.is_valid(data) else 0.0


class RegexEvaluator:
    """1.0 if the output matches the regex (search), 0.0 otherwise."""

    kind = "structural"

    def __init__(self, *, pattern: str, name: str = "structural.regex") -> None:
        self.name = name
        try:
            self._pattern = re.compile(pattern)
        except re.error as exc:
            raise EvaluatorError(f"invalid regex {pattern!r}: {exc}") from exc

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore:
        source_score = 1.0 if self._pattern.search(source_output) else 0.0
        target_score = 1.0 if self._pattern.search(target_output) else 0.0
        return PairedScore(
            source_score=source_score,
            target_score=target_score,
            metadata=_failure_metadata(source_score, target_score, FORMAT_FAILURE),
        )


class LengthEvaluator:
    """Score 1.0 inside the bounds, distance-decayed outside.

    The decay is linear: at ``2 * boundary`` distance, the score hits 0.
    This is slightly more forgiving than a hard 0/1 threshold while
    still penalising obvious blow-ups.
    """

    kind = "structural"

    def __init__(
        self,
        *,
        min_chars: int | None = None,
        max_chars: int | None = None,
        name: str = "structural.length",
    ) -> None:
        if min_chars is None and max_chars is None:
            raise EvaluatorError(
                "LengthEvaluator needs at least one of min_chars/max_chars",
            )
        self.name = name
        self._min = min_chars
        self._max = max_chars

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore:
        source_score = self._score_length(len(source_output))
        target_score = self._score_length(len(target_output))
        return PairedScore(
            source_score=source_score,
            target_score=target_score,
            metadata=_failure_metadata(source_score, target_score, FORMAT_FAILURE),
        )

    def _score_length(self, n: int) -> float:
        if self._min is not None and n < self._min:
            denom = max(self._min, 1)
            return max(0.0, 1.0 - (self._min - n) / denom)
        if self._max is not None and n > self._max:
            denom = max(self._max, 1)
            return max(0.0, 1.0 - (n - self._max) / denom)
        return 1.0


def _failure_metadata(source_score: float, target_score: float, category: str) -> dict[str, Any]:
    if target_score < source_score:
        return {"failure_categories": [category]}
    return {}


__all__ = ["JsonSchemaEvaluator", "LengthEvaluator", "RegexEvaluator"]
