"""Tests for the evaluators package."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aimigrate.evaluators import semantic as semantic_module
from aimigrate.evaluators.base import EvalRecord, Evaluator, EvaluatorError, PairedScore
from aimigrate.evaluators.llm_judge import PairwiseJudgeEvaluator, _parse_verdict
from aimigrate.evaluators.semantic import CosineSimilarityEvaluator, _cosine
from aimigrate.evaluators.structural import (
    JsonSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from aimigrate.models.client import CompletionResult, ModelClient

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TestPairedScore:
    def test_delta_is_target_minus_source(self) -> None:
        ps = PairedScore(source_score=0.2, target_score=0.9)
        assert ps.delta == pytest.approx(0.7)

    def test_score_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PairedScore(source_score=1.5, target_score=0.5)


class TestEvalRecord:
    def test_round_trip(self) -> None:
        rec = EvalRecord(
            run_id="r1",
            prompt_id="p",
            example_id="e",
            evaluator_name="x",
            source_score=0.8,
            target_score=0.6,
            delta=-0.2,
        )
        assert EvalRecord.model_validate_json(rec.model_dump_json()) == rec


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


class TestJsonSchemaEvaluator:
    def _schema(self, tmp_path: Path) -> Path:
        schema = {
            "type": "object",
            "required": ["greeting"],
            "properties": {"greeting": {"type": "string"}},
        }
        path = tmp_path / "schema.json"
        path.write_text(json.dumps(schema), encoding="utf-8")
        return path

    async def test_valid_outputs(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"greeting": "hello"}',
            target_output='{"greeting": "hi"}',
        )
        assert score.source_score == 1.0
        assert score.target_score == 1.0

    async def test_invalid_target_only(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"greeting": "hello"}',
            target_output='{"missing_field": true}',
        )
        assert score.source_score == 1.0
        assert score.target_score == 0.0
        assert score.delta == pytest.approx(-1.0)

    async def test_unparseable_json_scores_zero(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="not json",
            target_output='{"greeting": "ok"}',
        )
        assert score.source_score == 0.0
        assert score.target_score == 1.0

    def test_missing_schema_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluatorError):
            JsonSchemaEvaluator(schema_path=tmp_path / "nope.json")


class TestRegexEvaluator:
    async def test_match_vs_no_match(self) -> None:
        ev = RegexEvaluator(pattern=r"^summary:")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="summary: ok",
            target_output="random text",
        )
        assert score.source_score == 1.0
        assert score.target_score == 0.0

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(EvaluatorError):
            RegexEvaluator(pattern="(unclosed")


class TestLengthEvaluator:
    async def test_within_bounds(self) -> None:
        ev = LengthEvaluator(min_chars=5, max_chars=20)
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="hello world",
            target_output="ok",  # too short
        )
        assert score.source_score == 1.0
        assert 0.0 <= score.target_score < 1.0

    async def test_too_long_decays(self) -> None:
        ev = LengthEvaluator(max_chars=10)
        # 30 chars is way past the boundary; decay drives score toward 0.
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="x" * 5,
            target_output="x" * 30,
        )
        assert score.source_score == 1.0
        assert score.target_score < 0.5

    def test_no_bounds_raises(self) -> None:
        with pytest.raises(EvaluatorError):
            LengthEvaluator()


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------


class TestCosineHelper:
    def test_identical_vectors(self) -> None:
        assert _cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine([1.0], [1.0, 2.0]) == 0.0


class TestCosineEvaluator:
    async def test_high_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock embeddings: identical → similarity 1.
        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            return type(
                "R",
                (),
                {
                    "data": [{"embedding": [1.0, 0.0, 0.0]}],
                },
            )()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="a",
            target_output="b",
        )
        assert score.source_score == 1.0
        assert score.target_score == pytest.approx(1.0)
        assert score.delta == pytest.approx(0.0)

    async def test_embedding_failure_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator()
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="a",
            target_output="b",
        )
        assert score.source_score == 0.0
        assert score.target_score == 0.0
        assert "embedding failed" in score.explanation


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_clean_json(self) -> None:
        assert _parse_verdict('{"winner": "A", "reason": "x"}') == "A"

    def test_b_winner(self) -> None:
        assert _parse_verdict('{"winner": "B", "reason": "y"}') == "B"

    def test_tie(self) -> None:
        assert _parse_verdict('{"winner": "tie", "reason": "z"}') == "tie"

    def test_lowercase_winner(self) -> None:
        assert _parse_verdict('{"winner": "a"}') == "A"

    def test_json_inside_prose(self) -> None:
        text = 'Here is my verdict: {"winner": "A", "reason": "..."} done.'
        assert _parse_verdict(text) == "A"

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            _parse_verdict("just prose")

    def test_invalid_winner_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid winner"):
            _parse_verdict('{"winner": "C"}')


class TestPairwiseJudge:
    def _client_returning(self, text: str) -> ModelClient:
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            return_value=CompletionResult(
                text=text,
                model_id="x",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            ),
        )
        return client

    async def test_target_wins_when_target_is_a_and_a_wins(self) -> None:
        # Force target_is_a = True via a deterministic RNG.
        rng = random.Random(0)
        # random.Random(0).random() == 0.84... but we need it < 0.5 for target=A.
        # Easier to override below the bias by patching random() itself:
        rng.random = lambda: 0.0  # type: ignore[method-assign]
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "A"}'),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == 1.0
        assert score.source_score == 0.0

    async def test_target_loses_when_target_is_a_and_b_wins(self) -> None:
        rng = random.Random()
        rng.random = lambda: 0.0  # target=A  # type: ignore[method-assign]
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "B"}'),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == 0.0
        assert score.source_score == 1.0

    async def test_tie_yields_half_half(self) -> None:
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "tie"}'),
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.source_score == 0.5
        assert score.target_score == 0.5
        assert score.delta == 0.0

    async def test_malformed_response_degrades_gracefully(self) -> None:
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning("just prose, no JSON anywhere"),
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        # Defensive: malformed → neutral 0.5/0.5 with explanation.
        assert score.source_score == 0.5
        assert score.target_score == 0.5
        assert "judge failed" in score.explanation


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_every_evaluator_satisfies_protocol(self, tmp_path: Path) -> None:
        schema_path = tmp_path / "s.json"
        schema_path.write_text('{"type": "object"}', encoding="utf-8")
        evaluators = [
            JsonSchemaEvaluator(schema_path=schema_path),
            RegexEvaluator(pattern="x"),
            LengthEvaluator(min_chars=1),
            CosineSimilarityEvaluator(),
            PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="x"),
        ]
        for ev in evaluators:
            assert isinstance(ev, Evaluator)
