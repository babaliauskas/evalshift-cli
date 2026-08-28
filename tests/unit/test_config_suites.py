"""Tests for the ``suites:`` config section (capture-lifecycle bridge).

A ``suites`` entry names a promoted capture suite so ``evalshift run
--suite-name <name>`` can resolve its path without the user retyping it.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalshift.config.models import (
    EvalShiftConfig,
    EvaluatorsConfig,
    SuiteEvaluatorsOverride,
    SuiteSource,
)


def _base_config(**extra: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "prompts": [{"id": "p", "detection": "manual", "content": "Hello {query}"}],
        **extra,
    }


def test_suites_defaults_to_empty() -> None:
    cfg = EvalShiftConfig.model_validate(_base_config())
    assert cfg.suites == {}


def test_suites_parses_captured_source() -> None:
    cfg = EvalShiftConfig.model_validate(
        _base_config(
            suites={
                "promoted": {
                    "source": "captured",
                    "path": ".evalshift/suites/support_agent/golden.jsonl",
                },
            },
        ),
    )
    entry = cfg.suites["promoted"]
    assert isinstance(entry, SuiteSource)
    assert entry.source == "captured"
    assert entry.path == ".evalshift/suites/support_agent/golden.jsonl"


def test_suites_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        EvalShiftConfig.model_validate(
            _base_config(suites={"x": {"source": "magic", "path": "p.jsonl"}}),
        )


def test_suites_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        EvalShiftConfig.model_validate(
            _base_config(suites={"x": {"source": "jsonl", "path": "p.jsonl", "bogus": 1}}),
        )


# ---------------------------------------------------------------------------
# Per-suite evaluator overrides (`suites.<name>.evaluators`)
# ---------------------------------------------------------------------------


_TOP_LEVEL_EVALUATORS: dict[str, Any] = {
    "semantic": {"min_similarity": 0.8},
    "llm_judge": [{"criterion_name": "helpful", "criterion_prompt": "Which is better?"}],
    "tool_selection": [{"name": "global_routing"}],
    "tool_arguments": [{"name": "global_args"}],
}


def _config_with_suite_evaluators(**suite_extra: Any) -> EvalShiftConfig:
    return EvalShiftConfig.model_validate(
        _base_config(
            evaluators=_TOP_LEVEL_EVALUATORS,
            suites={
                "main_chat": {"source": "captured", "path": "a.jsonl", **suite_extra},
                "briefing": {"source": "captured", "path": "b.jsonl"},
            },
        ),
    )


def test_suite_evaluators_defaults_to_none_and_managed_true() -> None:
    cfg = EvalShiftConfig.model_validate(
        _base_config(suites={"x": {"source": "jsonl", "path": "p.jsonl"}}),
    )
    assert cfg.suites["x"].evaluators is None
    assert cfg.suites["x"].managed is True


def test_suite_evaluators_parses_full_block() -> None:
    cfg = _config_with_suite_evaluators(
        managed=False,
        evaluators={
            "tool_selection": [{"name": "routing", "divergence": "set"}],
            "tool_arguments": [{"name": "routing_args", "against": "expected"}],
        },
    )
    override = cfg.suites["main_chat"].evaluators
    assert override is not None
    assert isinstance(override, SuiteEvaluatorsOverride)
    assert override.tool_selection is not None
    assert override.tool_selection[0].name == "routing"
    assert override.tool_arguments is not None
    assert override.tool_arguments[0].against == "expected"
    assert cfg.suites["main_chat"].managed is False


def test_suite_evaluators_rejects_typo_in_nested_block() -> None:
    with pytest.raises(ValidationError):
        _config_with_suite_evaluators(
            evaluators={"tool_selection": [{"name": "routing", "divergance": "set"}]},
        )


def test_suite_evaluators_rejects_unknown_family() -> None:
    with pytest.raises(ValidationError):
        _config_with_suite_evaluators(evaluators={"tool_arguemnts": []})


def test_evaluators_for_none_returns_top_level_identity() -> None:
    cfg = _config_with_suite_evaluators()
    assert cfg.evaluators_for(None) is cfg.evaluators


def test_evaluators_for_suite_without_block_returns_top_level_identity() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_selection": [{"name": "routing"}]},
    )
    assert cfg.evaluators_for("briefing") is cfg.evaluators


def test_evaluators_for_unknown_suite_returns_top_level() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_selection": [{"name": "routing"}]},
    )
    assert cfg.evaluators_for("not-a-suite") is cfg.evaluators


def test_evaluators_for_replaces_declared_family_and_inherits_rest() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_arguments": [{"name": "routing_args", "against": "expected"}]},
    )
    resolved = cfg.evaluators_for("main_chat")

    # Declared family fully replaced — the top-level `global_args` is gone.
    assert [e.name for e in resolved.tool_arguments] == ["routing_args"]
    # Undeclared families inherited verbatim.
    assert resolved.semantic is not None
    assert resolved.semantic.min_similarity == 0.8
    assert [e.criterion_name for e in resolved.llm_judge] == ["helpful"]
    assert [e.name for e in resolved.tool_selection] == ["global_routing"]
    # The top-level config is untouched by resolution.
    assert [e.name for e in cfg.evaluators.tool_arguments] == ["global_args"]


def test_evaluators_for_declared_empty_family_removes_it() -> None:
    cfg = _config_with_suite_evaluators(evaluators={"tool_selection": []})
    resolved = cfg.evaluators_for("main_chat")
    assert resolved.tool_selection == []
    # Still inherits everything it did not declare.
    assert [e.name for e in resolved.tool_arguments] == ["global_args"]


def test_evaluators_for_explicit_null_removes_semantic() -> None:
    cfg = _config_with_suite_evaluators(evaluators={"semantic": None})
    assert cfg.evaluators_for("main_chat").semantic is None
    assert cfg.evaluators.semantic is not None


def test_evaluators_for_reflects_suite_tool_evaluator_names() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={
            "tool_selection": [{"name": "routing"}],
            "tool_arguments": [{"name": "routing_args", "against": "expected"}],
        },
    )
    assert cfg.evaluators_for("main_chat").tool_evaluator_names == frozenset(
        {"routing", "routing_args"},
    )
    assert cfg.evaluators_for("briefing").tool_evaluator_names == frozenset(
        {"global_routing", "global_args"},
    )


def test_suite_override_covers_every_evaluator_family() -> None:
    """The override must stay field-for-field aligned with the top-level model."""
    assert set(SuiteEvaluatorsOverride.model_fields) == set(EvaluatorsConfig.model_fields)
