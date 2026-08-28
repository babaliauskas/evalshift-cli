"""Tests for the shared economics/methodology builders.

These live outside ``reports/json.py`` so ``report.json`` and the hosted
bundle cannot drift; this module pins the contract both consume.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from evalshift.reports.economics import (
    build_economics,
    is_empty_output,
    methodology_notes,
    role_economics_to_dict,
)
from evalshift.runner.models import Call, RunModels, RunState


def _state(**overrides: Any) -> RunState:
    """Build a ``RunState`` with sensible defaults and keyword overrides."""
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


def _call(**overrides: Any) -> Call:
    """Build a ``Call`` with sensible defaults and keyword overrides."""
    defaults: dict[str, Any] = {
        "run_id": "r_20260803_aaaaaa",
        "prompt_id": "greet",
        "example_id": "ex1",
        "model_id": "gemini/gemini-2.5-flash",
        "role": "source",
    }
    return Call(**{**defaults, **overrides})


def test_role_economics_splits_source_and_target() -> None:
    calls = [
        _call(role="source", cached=True, cost_usd=0.001, input_tokens=100, output_tokens=20),
        _call(role="target", cached=True, cost_usd=0.002, input_tokens=90, output_tokens=10),
    ]
    economics = build_economics(calls)
    assert economics.source.calls == 1
    assert economics.target.total_input_tokens == 90
    assert economics.source.cached_calls == 1


def test_cached_calls_do_not_skew_latency() -> None:
    """Cache hits replay from disk with latency_ms = 0 and must be excluded."""
    calls = [
        _call(role="source", cached=True, latency_ms=0),
        _call(role="source", cached=False, latency_ms=800),
    ]
    source = build_economics(calls).source
    assert source.live_calls == 1
    assert source.latency_ms_avg == 800.0


def test_empty_output_needs_tokens_and_a_stop_finish() -> None:
    assert is_empty_output(
        _call(text="", output_tokens=5, error=None, trace=None, finish_reason="stop")
    )
    assert not is_empty_output(
        _call(text="", output_tokens=5, error=None, trace=None, finish_reason="length")
    )


class TestDeterminismNote:
    """``temperature`` withdrawal must be stated, not left to the reader."""

    def test_absent_when_every_arm_samples_deterministically(self) -> None:
        notes = methodology_notes(_state())
        assert not any("non-deterministic" in note for note in notes)

    def test_names_the_affected_model(self) -> None:
        notes = methodology_notes(_state(non_deterministic_models=["gemini/gemini-3.5-flash-lite"]))
        note = next(n for n in notes if "non-deterministic" in n)
        assert "gemini/gemini-3.5-flash-lite" in note

    def test_warns_that_paired_tests_are_weakened(self) -> None:
        """The consequence is the point — a bare fact would be ignored."""
        notes = methodology_notes(_state(non_deterministic_models=["gemini/gemini-3.5-flash-lite"]))
        note = next(n for n in notes if "non-deterministic" in n)
        assert "under-powered" in note

    def test_one_note_per_affected_model(self) -> None:
        notes = methodology_notes(
            _state(
                non_deterministic_models=[
                    "gemini/gemini-2.5-flash",
                    "gemini/gemini-3.5-flash-lite",
                ]
            )
        )
        assert sum("non-deterministic" in note for note in notes) == 2

    def test_keeps_the_existing_statistical_contract(self) -> None:
        """The new note is additive; it must not displace the standing notes."""
        baseline = methodology_notes(_state())
        with_note = methodology_notes(
            _state(non_deterministic_models=["gemini/gemini-3.5-flash-lite"])
        )
        assert set(baseline) < set(with_note)


def test_role_economics_to_dict_is_json_ready() -> None:
    source = build_economics([_call(role="source")]).source
    payload = role_economics_to_dict(source)
    assert set(payload) == {
        "calls",
        "live_calls",
        "cached_calls",
        "failed_calls",
        "truncated_calls",
        "empty_output_calls",
        "total_cost_usd",
        "total_input_tokens",
        "total_output_tokens",
        "latency_ms_avg",
        "latency_ms_p95",
    }
