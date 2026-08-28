"""Tests for recording which models in a run do not honour ``temperature``.

The flag is persisted rather than recomputed at report time: a bundle is
immutable, so a later LiteLLM upgrade must not rewrite what was true when the
calls were actually made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from evalshift.runner.models import RunModels, RunState


def _state(**overrides: Any) -> RunState:
    defaults: dict[str, Any] = {
        "run_id": "r_20260814_abcdef",
        "config_hash": "sha256:cafe",
        "started_at": datetime(2026, 8, 14, 9, 0, tzinfo=UTC),
        "models": RunModels(
            source="gemini/gemini-2.5-flash",
            target="gemini/gemini-3.5-flash-lite",
        ),
        "prompt_ids": ["greet"],
        "suite_path": "golden.jsonl",
        "total_evaluations": 2,
    }
    return RunState(**{**defaults, **overrides})


class TestRunStateField:
    def test_defaults_to_empty(self) -> None:
        assert _state().non_deterministic_models == []

    def test_records_the_affected_model_ids(self) -> None:
        state = _state(non_deterministic_models=["gemini/gemini-3.5-flash-lite"])
        assert state.non_deterministic_models == ["gemini/gemini-3.5-flash-lite"]

    def test_state_written_before_the_field_existed_still_loads(self) -> None:
        """Runs checkpointed by an earlier version must survive --resume."""
        legacy = {
            "run_id": "r_20260801_aaaaaa",
            "status": "in_progress",
            "config_hash": "sha256:cafe",
            "started_at": "2026-08-01T09:00:00Z",
            "models": {"source": "gemini/gemini-2.5-flash", "target": "openai/gpt-4o"},
            "prompt_ids": ["greet"],
            "suite_path": "golden.jsonl",
            "total_evaluations": 2,
            "completed_evaluations": 1,
        }
        state = RunState.model_validate(legacy)
        assert state.non_deterministic_models == []

    def test_round_trips_through_json(self) -> None:
        state = _state(non_deterministic_models=["gemini/gemini-3.5-flash-lite"])
        assert RunState.model_validate_json(state.model_dump_json()).non_deterministic_models == [
            "gemini/gemini-3.5-flash-lite"
        ]


class TestDetectionAtRunStart:
    """``non_deterministic_models`` is computed once, from both arms."""

    @pytest.mark.parametrize(
        ("honoured", "expected"),
        [
            ({"gemini/gemini-2.5-flash": True, "gemini/gemini-3.5-flash-lite": True}, []),
            (
                {"gemini/gemini-2.5-flash": True, "gemini/gemini-3.5-flash-lite": False},
                ["gemini/gemini-3.5-flash-lite"],
            ),
            (
                {"gemini/gemini-2.5-flash": False, "gemini/gemini-3.5-flash-lite": False},
                ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash-lite"],
            ),
        ],
        ids=["both-honour", "target-only", "both-affected"],
    )
    def test_collects_ids_that_do_not_honour_temperature(
        self,
        monkeypatch: pytest.MonkeyPatch,
        honoured: dict[str, bool],
        expected: list[str],
    ) -> None:
        from evalshift.runner import orchestrator

        monkeypatch.setattr(orchestrator, "honors_temperature", lambda model_id: honoured[model_id])
        assert (
            orchestrator.detect_non_deterministic_models(
                source="gemini/gemini-2.5-flash",
                target="gemini/gemini-3.5-flash-lite",
            )
            == expected
        )

    def test_deduplicates_when_both_arms_are_the_same_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A/A runs compare a model against itself; it must be listed once."""
        from evalshift.runner import orchestrator

        monkeypatch.setattr(orchestrator, "honors_temperature", lambda model_id: False)
        assert orchestrator.detect_non_deterministic_models(
            source="gemini/gemini-3.5-flash-lite",
            target="gemini/gemini-3.5-flash-lite",
        ) == ["gemini/gemini-3.5-flash-lite"]
