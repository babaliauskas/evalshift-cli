"""Unit tests for the pydantic models in :mod:`evalshift.config.models`.

These tests are the primary contract test for ``evalshift.yaml``: any
change to validation behaviour should land here first.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalshift.config.models import (
    DEFAULT_JUDGE_MODEL,
    Defaults,
    EvalShiftConfig,
    EvaluatorsConfig,
    LLMJudgeConfig,
    PromptDefinition,
    SemanticEvaluatorConfig,
    SliceConfig,
    StructuralEvaluatorConfig,
)

# ---------------------------------------------------------------------------
# PromptDefinition
# ---------------------------------------------------------------------------


class TestPromptDefinition:
    def test_manual_with_content_is_valid(self) -> None:
        p = PromptDefinition(id="cs", detection="manual", content="hello {name}")
        assert p.detection == "manual"
        assert p.content == "hello {name}"
        assert p.path is None
        assert p.variable is None
        assert p.variables == []

    def test_manual_without_content_fails(self) -> None:
        with pytest.raises(ValidationError, match="'content' is required"):
            PromptDefinition(id="cs", detection="manual")

    def test_manual_with_path_fails(self) -> None:
        with pytest.raises(ValidationError, match="must not be set when detection='manual'"):
            PromptDefinition(
                id="cs",
                detection="manual",
                content="hi",
                path="src/prompts.py",
            )

    def test_python_string_with_path_and_variable_is_valid(self) -> None:
        p = PromptDefinition(
            id="cs",
            detection="python_string",
            path="src/prompts.py",
            variable="MY_PROMPT",
            variables=["a", "b"],
        )
        assert p.path == "src/prompts.py"
        assert p.variable == "MY_PROMPT"
        assert p.variables == ["a", "b"]

    def test_python_string_missing_path_fails(self) -> None:
        with pytest.raises(ValidationError, match="'path' and 'variable' are required"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                variable="MY_PROMPT",
            )

    def test_python_string_missing_variable_fails(self) -> None:
        with pytest.raises(ValidationError, match="'path' and 'variable' are required"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                path="src/prompts.py",
            )

    def test_python_string_with_content_fails(self) -> None:
        with pytest.raises(ValidationError, match="'content' must not be set"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                path="src/prompts.py",
                variable="X",
                content="leaked",
            )

    def test_invalid_detection_value_fails(self) -> None:
        with pytest.raises(ValidationError):
            PromptDefinition(id="cs", detection="auto", content="hi")  # type: ignore[arg-type]

    def test_empty_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            PromptDefinition(id="", detection="manual", content="hi")

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PromptDefinition.model_validate(
                {
                    "id": "cs",
                    "detection": "manual",
                    "content": "hi",
                    "typo": "oops",
                },
            )


# ---------------------------------------------------------------------------
# StructuralEvaluatorConfig
# ---------------------------------------------------------------------------


class TestStructuralEvaluatorConfig:
    def test_json_schema_requires_schema_path(self) -> None:
        with pytest.raises(ValidationError, match="'schema_path' is required"):
            StructuralEvaluatorConfig(type="json_schema")

    def test_json_schema_with_schema_path_is_valid(self) -> None:
        c = StructuralEvaluatorConfig(type="json_schema", schema_path="schema.json")
        assert c.applies_to == ["*"]

    def test_regex_requires_pattern(self) -> None:
        with pytest.raises(ValidationError, match="'pattern' is required"):
            StructuralEvaluatorConfig(type="regex")

    def test_length_requires_at_least_one_bound(self) -> None:
        with pytest.raises(ValidationError, match="at least one of 'min_chars' or 'max_chars'"):
            StructuralEvaluatorConfig(type="length")

    def test_length_with_min_only_is_valid(self) -> None:
        c = StructuralEvaluatorConfig(type="length", min_chars=10)
        assert c.min_chars == 10
        assert c.max_chars is None

    def test_length_min_greater_than_max_fails(self) -> None:
        with pytest.raises(ValidationError, match="'min_chars' must be <= 'max_chars'"):
            StructuralEvaluatorConfig(type="length", min_chars=100, max_chars=10)

    def test_negative_length_bounds_fail(self) -> None:
        with pytest.raises(ValidationError):
            StructuralEvaluatorConfig(type="length", min_chars=-1)

    def test_applies_to_override(self) -> None:
        c = StructuralEvaluatorConfig(type="regex", pattern=r"^\d+$", applies_to=["cs-*"])
        assert c.applies_to == ["cs-*"]


# ---------------------------------------------------------------------------
# SemanticEvaluatorConfig & LLMJudgeConfig
# ---------------------------------------------------------------------------


class TestSemanticEvaluatorConfig:
    def test_defaults(self) -> None:
        c = SemanticEvaluatorConfig()
        assert c.embedding_model == "text-embedding-3-small"
        assert c.applies_to == ["*"]


class TestLLMJudgeConfig:
    def test_minimum_valid(self) -> None:
        c = LLMJudgeConfig(
            criterion_name="factuality",
            criterion_prompt="Which output preserves more factual detail?",
        )
        assert c.judge_model == DEFAULT_JUDGE_MODEL

    def test_empty_criterion_name_fails(self) -> None:
        with pytest.raises(ValidationError):
            LLMJudgeConfig(criterion_name="", criterion_prompt="x")

    def test_empty_criterion_prompt_fails(self) -> None:
        with pytest.raises(ValidationError):
            LLMJudgeConfig(criterion_name="x", criterion_prompt="")


# ---------------------------------------------------------------------------
# SliceConfig
# ---------------------------------------------------------------------------


class TestSliceConfig:
    def test_defaults(self) -> None:
        s = SliceConfig(name="long", filter="len(conversation) > 1000")
        assert s.applies_to == ["*"]

    def test_empty_name_fails(self) -> None:
        with pytest.raises(ValidationError):
            SliceConfig(name="", filter="True")

    def test_empty_filter_fails(self) -> None:
        with pytest.raises(ValidationError):
            SliceConfig(name="long", filter="")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_factory_defaults(self) -> None:
        d = Defaults()
        assert d.source_model is None
        assert d.target_model is None
        assert d.judge_model == DEFAULT_JUDGE_MODEL
        assert d.concurrency == 10
        assert d.cache is True
        assert d.max_cost_usd == 50.0

    @pytest.mark.parametrize("bad_concurrency", [0, -1, 65, 1000])
    def test_concurrency_bounds(self, bad_concurrency: int) -> None:
        with pytest.raises(ValidationError):
            Defaults(concurrency=bad_concurrency)

    def test_negative_max_cost_fails(self) -> None:
        with pytest.raises(ValidationError):
            Defaults(max_cost_usd=-1.0)


# ---------------------------------------------------------------------------
# EvalShiftConfig — top level
# ---------------------------------------------------------------------------


class TestEvalShiftConfig:
    def test_minimal_valid(self) -> None:
        cfg = EvalShiftConfig(
            prompts=[
                PromptDefinition(id="cs", detection="manual", content="hi {n}"),
            ],
        )
        assert cfg.version == 1
        assert isinstance(cfg.defaults, Defaults)
        assert isinstance(cfg.evaluators, EvaluatorsConfig)
        assert cfg.slices == []
        assert cfg.project is None
        assert cfg.thresholds == {}

    def test_hosted_project_and_thresholds_are_valid(self) -> None:
        cfg = EvalShiftConfig(
            prompts=[
                PromptDefinition(id="cs", detection="manual", content="hi {n}"),
            ],
            project="acme/model-migration",
            thresholds={"pass_rate_min": 0.91, "slices": {"security": 0.95}},
        )

        assert cfg.project == "acme/model-migration"
        assert cfg.thresholds == {"pass_rate_min": 0.91, "slices": {"security": 0.95}}

    def test_invalid_hosted_project_slug_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig(
                prompts=[
                    PromptDefinition(id="cs", detection="manual", content="hi {n}"),
                ],
                project="Acme/model migration",
            )

    def test_round_trip_through_dump(self) -> None:
        original = EvalShiftConfig(
            prompts=[
                PromptDefinition(
                    id="cs",
                    detection="python_string",
                    path="src/prompts.py",
                    variable="MY_PROMPT",
                    variables=["conversation"],
                ),
            ],
            evaluators=EvaluatorsConfig(
                structural=[
                    StructuralEvaluatorConfig(type="json_schema", schema_path="s.json"),
                ],
                semantic=SemanticEvaluatorConfig(),
                llm_judge=[
                    LLMJudgeConfig(criterion_name="fact", criterion_prompt="?"),
                ],
            ),
            slices=[SliceConfig(name="long", filter="len(conversation)>1000")],
        )
        recreated = EvalShiftConfig.model_validate(original.model_dump())
        assert recreated == original

    def test_empty_prompts_list_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig(prompts=[])

    def test_duplicate_prompt_ids_fail(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate prompt ids: \['cs'\]"):
            EvalShiftConfig(
                prompts=[
                    PromptDefinition(id="cs", detection="manual", content="a"),
                    PromptDefinition(id="cs", detection="manual", content="b"),
                ],
            )

    def test_wrong_version_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig.model_validate(
                {
                    "version": 2,
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                },
            )

    def test_extra_top_level_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvalShiftConfig.model_validate(
                {
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                    "rogue_section": {"foo": "bar"},
                },
            )

    def test_nested_validation_error_propagates(self) -> None:
        # An invalid evaluator should fail the top-level validation, not
        # silently slip through.
        with pytest.raises(ValidationError, match="'schema_path' is required"):
            EvalShiftConfig.model_validate(
                {
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                    "evaluators": {
                        "structural": [{"type": "json_schema"}],
                    },
                },
            )
