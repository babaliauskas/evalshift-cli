"""Unit tests for :mod:`aimigrate.runner.models`.

The most important property: every model round-trips losslessly through
``model_dump_json`` / ``model_validate_json``. The orchestrator relies
on that for both ``state.json`` and ``raw.jsonl``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aimigrate.runner.models import Call, RunModels, RunState

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
