"""Tests for recording generation constraints the run's models cannot honour.

``models/client.py`` sets ``drop_params=True`` so a call carrying a parameter
the provider never accepted still succeeds. That is the right runtime
behaviour and the wrong reporting behaviour: the replay stops reproducing the
captured constraint and nothing says so. These tests pin the record.

Persisted at run start for the same reason as ``non_deterministic_models``: a
bundle is immutable, so a later LiteLLM upgrade must not rewrite what was true
when the calls were made.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.runner import orchestrator
from evalshift_cli.runner.models import RunModels, RunState
from evalshift_cli.suite.models import Suite, SuiteExample

_SOURCE = "gemini/gemini-2.5-flash"
_TARGET = "openai/gpt-4o-mini"


def _strict_tool() -> ToolSpec:
    return ToolSpec(name="lookup", strict=True)


def _state(**overrides: Any) -> RunState:
    defaults: dict[str, Any] = {
        "run_id": "r_20260814_abcdef",
        "config_hash": "sha256:cafe",
        "started_at": datetime(2026, 8, 14, 9, 0, tzinfo=UTC),
        "models": RunModels(source=_SOURCE, target=_TARGET),
        "prompt_ids": ["greet"],
        "suite_path": "golden.jsonl",
        "total_evaluations": 2,
    }
    return RunState(**{**defaults, **overrides})


def _suite(*configs: dict[str, Any] | None) -> Suite:
    return Suite(
        examples=[
            SuiteExample(
                id=f"ex{i}",
                inputs={},
                tools=[],
                generation_config=config,
            )
            for i, config in enumerate(configs)
        ]
    )


class TestRunStateField:
    def test_defaults_to_empty(self) -> None:
        assert _state().dropped_params == {}

    def test_records_params_per_model(self) -> None:
        state = _state(dropped_params={_TARGET: ["response_format", "tool_choice"]})
        assert state.dropped_params == {_TARGET: ["response_format", "tool_choice"]}

    def test_state_written_before_the_field_existed_still_loads(self) -> None:
        """Runs checkpointed by an earlier version must survive --resume."""
        legacy = {
            "run_id": "r_20260801_aaaaaa",
            "status": "in_progress",
            "config_hash": "sha256:cafe",
            "started_at": "2026-08-01T09:00:00Z",
            "models": {"source": _SOURCE, "target": _TARGET},
            "prompt_ids": ["greet"],
            "suite_path": "golden.jsonl",
            "total_evaluations": 2,
            "completed_evaluations": 1,
        }
        assert RunState.model_validate(legacy).dropped_params == {}

    def test_round_trips_through_json(self) -> None:
        state = _state(dropped_params={_TARGET: ["tool_choice"]})
        assert RunState.model_validate_json(state.model_dump_json()).dropped_params == {
            _TARGET: ["tool_choice"]
        }


class TestRequestedGenerationParams:
    """Recorded capture keys, mapped onto the names LiteLLM answers about."""

    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [
            ({"tool_choice": "any"}, ["tool_choice"]),
            ({"parallel_tool_calls": False}, ["parallel_tool_calls"]),
            # Gemini's spelling of tool_choice.
            ({"tool_config": {"mode": "ANY"}}, ["tool_choice"]),
            # Gemini's spelling of structured output.
            ({"response_mime_type": "application/json"}, ["response_format"]),
            ({"response_schema": {"type": "object"}}, ["response_format"]),
            ({"response_format": {"type": "json_object"}}, ["response_format"]),
            # Gemini's spelling of the completion cap.
            ({"max_output_tokens": 512}, ["max_tokens"]),
            ({"max_tokens": 512}, ["max_tokens"]),
            ({"top_p": 0.9}, ["top_p"]),
            # Unknown keys are not our business; they are never probed.
            ({"seed": 7, "candidate_count": 1}, []),
        ],
    )
    def test_maps_recorded_keys_to_litellm_param_names(
        self, recorded: dict[str, Any], expected: list[str]
    ) -> None:
        assert orchestrator.requested_generation_params(_suite(recorded)) == expected

    def test_two_spellings_of_one_constraint_collapse(self) -> None:
        suite = _suite({"response_mime_type": "application/json", "response_schema": {}})
        assert orchestrator.requested_generation_params(suite) == ["response_format"]

    def test_unions_across_examples_and_sorts(self) -> None:
        suite = _suite({"tool_choice": "any"}, {"max_output_tokens": 8}, None)
        assert orchestrator.requested_generation_params(suite) == ["max_tokens", "tool_choice"]

    def test_temperature_is_left_to_the_determinism_probe(self) -> None:
        """``non_deterministic_models`` owns that parameter and its own banner.

        Reporting it in both places would put two banners on one run saying the
        same thing, and the determinism probe is the stronger of the two: it
        runs whether or not a capture happened to record a temperature.
        """
        assert orchestrator.requested_generation_params(_suite({"temperature": 0.0})) == []

    def test_empty_for_a_suite_that_recorded_nothing(self) -> None:
        assert orchestrator.requested_generation_params(_suite(None, {})) == []


class TestDetectionAtRunStart:
    def test_records_params_the_target_cannot_honour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            orchestrator,
            "unsupported_params",
            lambda model_id, params: ["tool_choice"] if model_id == _TARGET else [],
        )
        assert orchestrator.detect_dropped_params(
            source=_SOURCE,
            target=_TARGET,
            suite=_suite({"tool_choice": "any"}),
        ) == {_TARGET: ["tool_choice"]}

    def test_probes_both_arms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The source arm can be the affected one — a replay is not a capture."""
        monkeypatch.setattr(
            orchestrator, "unsupported_params", lambda model_id, params: ["response_format"]
        )
        assert orchestrator.detect_dropped_params(
            source=_SOURCE,
            target=_TARGET,
            suite=_suite({"response_mime_type": "application/json"}),
        ) == {_SOURCE: ["response_format"], _TARGET: ["response_format"]}

    def test_models_that_honour_everything_are_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(orchestrator, "unsupported_params", lambda model_id, params: [])
        assert (
            orchestrator.detect_dropped_params(
                source=_SOURCE, target=_TARGET, suite=_suite({"tool_choice": "any"})
            )
            == {}
        )

    def test_no_probe_when_the_suite_recorded_no_constraints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No capture asked for anything, so nothing can have been dropped."""
        seen: list[str] = []

        def fake(model_id: str, params: Any) -> list[str]:
            seen.append(model_id)
            return ["tool_choice"]

        monkeypatch.setattr(orchestrator, "unsupported_params", fake)
        assert (
            orchestrator.detect_dropped_params(source=_SOURCE, target=_TARGET, suite=_suite(None))
            == {}
        )
        assert seen == []

    def test_an_a_a_run_lists_the_model_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            orchestrator, "unsupported_params", lambda model_id, params: ["tool_choice"]
        )
        assert orchestrator.detect_dropped_params(
            source=_TARGET, target=_TARGET, suite=_suite({"tool_choice": "any"})
        ) == {_TARGET: ["tool_choice"]}

    def test_warns_once_per_model_and_param(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One warning at run start, not one per dispatched call."""
        monkeypatch.setattr(
            orchestrator,
            "unsupported_params",
            lambda model_id, params: (
                ["response_format", "tool_choice"] if model_id == _TARGET else []
            ),
        )
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.orchestrator"):
            orchestrator.detect_dropped_params(
                source=_SOURCE,
                target=_TARGET,
                suite=_suite({"tool_choice": "any"}, {"response_format": {}}),
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        rendered = " ".join(r.getMessage() for r in warnings)
        assert _TARGET in rendered
        assert "tool_choice" in rendered
        assert "response_format" in rendered


class TestKnownLiteLLMGaps:
    """Constraints LiteLLM accepts, claims to support, and then never sends.

    The probe alone cannot see these: ``get_supported_openai_params`` lists
    ``parallel_tool_calls`` for Gemini, and tool strictness is not an OpenAI
    generation parameter at all, so it has no probe to fail. Both are dropped
    inside LiteLLM's Gemini transformer, which makes them exactly as invisible
    to the replay as a genuinely unsupported parameter — and therefore the
    same kind of caveat on the report.
    """

    @staticmethod
    def _no_probe_result(monkeypatch: pytest.MonkeyPatch) -> None:
        """LiteLLM says everything is supported; only the gap table can speak."""
        monkeypatch.setattr(orchestrator, "unsupported_params", lambda model_id, params: [])

    def test_gemini_target_records_parallel_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._no_probe_result(monkeypatch)
        assert orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite({"parallel_tool_calls": False}),
        ) == {_SOURCE: ["parallel_tool_calls"]}

    def test_parallel_tool_calls_only_when_a_capture_recorded_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gap is only a loss if the replay was going to send the constraint."""
        self._no_probe_result(monkeypatch)
        assert (
            orchestrator.detect_dropped_params(
                source=_TARGET, target=_SOURCE, suite=_suite({"tool_choice": "any"})
            )
            == {}
        )

    def test_gemini_target_records_strict_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_probe_result(monkeypatch)
        assert orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite(None),
            tools_by_example={"ex0": [_strict_tool()]},
        ) == {_SOURCE: ["tools.strict"]}

    def test_non_strict_tools_record_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_probe_result(monkeypatch)
        assert (
            orchestrator.detect_dropped_params(
                source=_TARGET,
                target=_SOURCE,
                suite=_suite(None),
                tools_by_example={"ex0": [ToolSpec(name="lookup")]},
            )
            == {}
        )

    def test_one_strict_example_is_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_probe_result(monkeypatch)
        assert orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite(None, None),
            tools_by_example={"ex0": [ToolSpec(name="lookup")], "ex1": [_strict_tool()]},
        ) == {_SOURCE: ["tools.strict"]}

    def test_non_gemini_arms_are_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._no_probe_result(monkeypatch)
        assert (
            orchestrator.detect_dropped_params(
                source=_TARGET,
                target="anthropic/claude-4.5-sonnet",
                suite=_suite({"parallel_tool_calls": False}),
                tools_by_example={"ex0": [_strict_tool()]},
            )
            == {}
        )

    def test_probe_and_table_merge_into_one_sorted_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            orchestrator,
            "unsupported_params",
            lambda model_id, params: ["response_format"] if model_id == _SOURCE else [],
        )
        assert orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite({"parallel_tool_calls": False, "response_mime_type": "application/json"}),
            tools_by_example={"ex0": [_strict_tool()]},
        ) == {_SOURCE: ["parallel_tool_calls", "response_format", "tools.strict"]}

    def test_the_pseudo_param_is_never_probed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``tools.strict`` is not an OpenAI parameter; LiteLLM has no answer for it."""
        asked: list[list[str]] = []

        def fake(model_id: str, params: Any) -> list[str]:
            asked.append(list(params))
            return []

        monkeypatch.setattr(orchestrator, "unsupported_params", fake)
        orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite({"tool_choice": "any"}),
            tools_by_example={"ex0": [_strict_tool()]},
        )
        assert asked and all("tools.strict" not in params for params in asked)

    def test_strict_tools_alone_still_skip_the_probe_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No recorded generation_config means nothing to ask LiteLLM about."""
        seen: list[str] = []

        def fake(model_id: str, params: Any) -> list[str]:
            seen.append(model_id)
            return []

        monkeypatch.setattr(orchestrator, "unsupported_params", fake)
        orchestrator.detect_dropped_params(
            source=_TARGET,
            target=_SOURCE,
            suite=_suite(None),
            tools_by_example={"ex0": [_strict_tool()]},
        )
        assert seen == []

    def test_warns_about_table_entries_too(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._no_probe_result(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.runner.orchestrator"):
            orchestrator.detect_dropped_params(
                source=_TARGET,
                target=_SOURCE,
                suite=_suite({"parallel_tool_calls": False}),
                tools_by_example={"ex0": [_strict_tool()]},
            )
        rendered = " ".join(r.getMessage() for r in caplog.records)
        assert "parallel_tool_calls" in rendered
        assert "tools.strict" in rendered
        assert _SOURCE in rendered
