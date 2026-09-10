"""Unit tests for :mod:`evalshift_cli.runner.models`.

The most important property: every model round-trips losslessly through
``model_dump_json`` / ``model_validate_json``. The orchestrator relies
on that for both ``state.json`` and ``raw.jsonl``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evalshift_cli.runner.models import (
    Call,
    EvaluatorCoverage,
    RunModels,
    RunState,
    representative_calls,
)

# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------


class TestCall:
    def test_minimum_valid(self) -> None:
        call = Call(
            run_id="r_20260601_abc123",
            prompt_id="greet",
            example_id="ex1",
            model_id="gemini/gemini-2.5-flash",
            role="source",
        )
        assert call.text == ""
        assert call.error is None
        assert call.succeeded is True
        assert call.cached is False

    def test_succeeded_false_when_error_present(self) -> None:
        call = Call(
            run_id="r1",
            prompt_id="p",
            example_id="ex",
            model_id="m",
            role="target",
            error="rate limited",
        )
        assert call.succeeded is False

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Call(
                run_id="r1",
                prompt_id="p",
                example_id="ex",
                model_id="m",
                role="judge",  # type: ignore[arg-type]
            )

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Call.model_validate(
                {
                    "run_id": "r1",
                    "prompt_id": "p",
                    "example_id": "ex",
                    "model_id": "m",
                    "role": "source",
                    "rogue": "field",
                },
            )

    def test_round_trip_through_json(self) -> None:
        original = Call(
            run_id="r1",
            prompt_id="p",
            example_id="ex",
            model_id="m",
            role="source",
            text="Hello!",
            input_tokens=11,
            output_tokens=4,
            cost_usd=0.0001,
            latency_ms=250,
            cached=True,
        )
        rebuilt = Call.model_validate_json(original.model_dump_json())
        assert rebuilt == original


# ---------------------------------------------------------------------------
# RunModels
# ---------------------------------------------------------------------------


class TestRunModels:
    def test_minimum_valid(self) -> None:
        rm = RunModels(source="a", target="b")
        assert rm.source == "a"
        assert rm.target == "b"

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunModels(source="", target="b")


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 6, 1, tzinfo=UTC)


class TestRunState:
    def test_minimum_valid(self) -> None:
        state = RunState(
            run_id="r_20260601_abc123",
            config_hash="abcd",
            started_at=_now(),
            models=RunModels(source="a", target="b"),
            prompt_ids=["p1"],
            suite_path="./golden.jsonl",
            total_evaluations=200,
        )
        assert state.status == "in_progress"
        assert state.completed_evaluations == 0
        assert state.last_checkpoint_at is None

    def test_suite_name_defaults_to_none(self) -> None:
        state = RunState(
            run_id="r_20260601_abc123",
            config_hash="abcd",
            started_at=_now(),
            models=RunModels(source="a", target="b"),
            prompt_ids=["p1"],
            suite_path="./golden.jsonl",
            total_evaluations=200,
        )
        assert state.suite_name is None

    def test_suite_name_round_trips_through_json(self) -> None:
        original = RunState(
            run_id="r_20260601_abc123",
            config_hash="abcd",
            started_at=_now(),
            models=RunModels(source="a", target="b"),
            prompt_ids=["p1"],
            suite_path="./golden.jsonl",
            suite_name="main_chat",
            total_evaluations=200,
        )
        rebuilt = RunState.model_validate_json(original.model_dump_json())
        assert rebuilt.suite_name == "main_chat"
        assert rebuilt == original

    def test_empty_prompt_ids_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunState(
                run_id="r1",
                config_hash="abcd",
                started_at=_now(),
                models=RunModels(source="a", target="b"),
                prompt_ids=[],
                suite_path="./golden.jsonl",
                total_evaluations=0,
            )

    def test_negative_completion_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunState(
                run_id="r1",
                config_hash="abcd",
                started_at=_now(),
                models=RunModels(source="a", target="b"),
                prompt_ids=["p1"],
                suite_path="./golden.jsonl",
                total_evaluations=10,
                completed_evaluations=-1,
            )

    def test_round_trip_through_json(self) -> None:
        original = RunState(
            run_id="r_20260601_abc123",
            status="in_progress",
            config_hash="abcd",
            started_at=_now(),
            last_checkpoint_at=_now(),
            models=RunModels(source="gemini/flash", target="gemini/pro"),
            prompt_ids=["p1", "p2"],
            suite_path="./golden.jsonl",
            total_evaluations=200,
            completed_evaluations=87,
        )
        rebuilt = RunState.model_validate_json(original.model_dump_json())
        assert rebuilt == original

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunState.model_validate(
                {
                    "run_id": "r1",
                    "status": "exploded",
                    "config_hash": "x",
                    "started_at": _now().isoformat(),
                    "models": {"source": "a", "target": "b"},
                    "prompt_ids": ["p1"],
                    "suite_path": "x.jsonl",
                    "total_evaluations": 0,
                },
            )

    def test_extra_top_level_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RunState.model_validate(
                {
                    "run_id": "r1",
                    "config_hash": "x",
                    "started_at": _now().isoformat(),
                    "models": {"source": "a", "target": "b"},
                    "prompt_ids": ["p1"],
                    "suite_path": "x.jsonl",
                    "total_evaluations": 0,
                    "rogue": True,
                },
            )


# ---------------------------------------------------------------------------
# EvaluatorCoverage
# ---------------------------------------------------------------------------


class TestEvaluatorCoverage:
    def test_an_entry_written_before_blocking_existed_loads_as_gating(self) -> None:
        """Old ``state.json`` files predate the flag. Unknown must read as
        gating — over-warning about a blind gate is the conservative side."""
        entry = EvaluatorCoverage.model_validate(
            {
                "evaluator_name": "semantic.cosine",
                "kind": "semantic",
                "attempted": 7,
                "recorded": 0,
            },
        )
        assert entry.blocking is True

    def test_blocking_round_trips_through_json(self) -> None:
        entry = EvaluatorCoverage(
            evaluator_name="semantic.cosine",
            kind="semantic",
            attempted=7,
            recorded=0,
            blocking=False,
        )
        restored = EvaluatorCoverage.model_validate_json(entry.model_dump_json())
        assert restored.blocking is False


class TestSampleIndex:
    """``samples_per_example`` (Task 7.1) adds a sample dimension to both rows."""

    def test_call_sample_index_defaults_to_zero(self) -> None:
        call = Call(run_id="r", prompt_id="p", example_id="e", model_id="m", role="source")
        assert call.sample_index == 0

    def test_call_written_before_the_field_existed_loads_as_sample_zero(self) -> None:
        raw = (
            '{"run_id": "r", "prompt_id": "p", "example_id": "e", '
            '"model_id": "m", "role": "source", "text": "x"}'
        )
        assert Call.model_validate_json(raw).sample_index == 0

    def test_call_sample_index_round_trips(self) -> None:
        call = Call(
            run_id="r", prompt_id="p", example_id="e", model_id="m", role="target", sample_index=2
        )
        assert Call.model_validate_json(call.model_dump_json()).sample_index == 2

    def test_call_negative_sample_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Call(
                run_id="r",
                prompt_id="p",
                example_id="e",
                model_id="m",
                role="source",
                sample_index=-1,
            )

    def test_run_state_samples_per_example_defaults_to_one(self) -> None:
        state = RunState(
            run_id="r_20260601_abc123",
            config_hash="abcd",
            started_at=_now(),
            models=RunModels(source="a", target="b"),
            prompt_ids=["p1"],
            suite_path="./golden.jsonl",
            total_evaluations=2,
        )
        assert state.samples_per_example == 1
        assert RunState.model_validate_json(state.model_dump_json()).samples_per_example == 1

    def test_run_state_samples_per_example_round_trips(self) -> None:
        state = RunState(
            run_id="r_20260601_abc123",
            config_hash="abcd",
            started_at=_now(),
            models=RunModels(source="a", target="b"),
            prompt_ids=["p1"],
            suite_path="./golden.jsonl",
            total_evaluations=6,
            samples_per_example=3,
        )
        assert RunState.model_validate_json(state.model_dump_json()).samples_per_example == 3


class TestRepresentativeCalls:
    """Consumers that show one output per (prompt, example, role) read sample 0."""

    def _call(self, *, example_id: str, role: str, sample_index: int) -> Call:
        return Call(
            run_id="r",
            prompt_id="p",
            example_id=example_id,
            model_id="m",
            role=role,  # type: ignore[arg-type]
            text=f"{example_id}/{role}/{sample_index}",
            sample_index=sample_index,
        )

    def test_keeps_only_sample_zero_in_order(self) -> None:
        calls = [
            self._call(example_id="a", role="source", sample_index=1),
            self._call(example_id="a", role="source", sample_index=0),
            self._call(example_id="a", role="target", sample_index=0),
            self._call(example_id="b", role="source", sample_index=2),
        ]
        kept = representative_calls(calls)
        assert [c.text for c in kept] == ["a/source/0", "a/target/0"]

    def test_single_sample_run_is_returned_unchanged(self) -> None:
        calls = [
            self._call(example_id="a", role="source", sample_index=0),
            self._call(example_id="a", role="target", sample_index=0),
        ]
        assert representative_calls(calls) == calls
