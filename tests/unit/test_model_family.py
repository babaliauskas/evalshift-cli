"""Tests for :mod:`evalshift_cli.models.family` — judge/arm model-family overlap."""

from __future__ import annotations

from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.models.family import (
    JudgeFamilyOverlap,
    configured_judge_models,
    describe_overlap,
    judge_family_overlaps,
    shared_judge_family,
)


class TestSharedJudgeFamily:
    def test_same_provider_as_target_names_the_target(self) -> None:
        roles = shared_judge_family(
            judge_model="gemini/gemini-3.1-flash-lite-preview",
            source_model="anthropic/claude-sonnet-4-5",
            target_model="gemini/gemini-2.5-pro",
        )
        assert roles == ["target"]

    def test_same_provider_as_both_arms_names_both_in_order(self) -> None:
        roles = shared_judge_family(
            judge_model="gemini-3.1-flash-lite-preview",
            source_model="gemini/gemini-2.5-flash",
            target_model="gemini/gemini-2.5-pro",
        )
        assert roles == ["source", "target"]

    def test_third_family_judge_shares_nothing(self) -> None:
        roles = shared_judge_family(
            judge_model="openai/gpt-4.1-mini",
            source_model="anthropic/claude-sonnet-4-5",
            target_model="gemini/gemini-2.5-pro",
        )
        assert roles == []

    def test_unknown_provider_never_matches(self) -> None:
        # ``other`` means "could not tell", and two unknowns are not the
        # same family just because neither was recognised.
        roles = shared_judge_family(
            judge_model="llama3.1:8b",
            source_model="mistral-large",
            target_model="llama3.1:70b",
        )
        assert roles == []

    def test_alias_and_canonical_id_resolve_to_one_provider(self) -> None:
        # A registry alias for the judge and a bare prefix-inferred id for
        # the target land on the same provider.
        roles = shared_judge_family(
            judge_model="claude-4.5-sonnet",
            source_model="gpt-4.1",
            target_model="claude-opus-4-1",
        )
        assert roles == ["target"]


class TestJudgeFamilyOverlaps:
    def test_dedupes_judges_and_keeps_first_seen_order(self) -> None:
        overlaps = judge_family_overlaps(
            judge_models=["gemini/g-a", "openai/gpt-4.1", "gemini/g-a", "gemini/g-b"],
            source_model="openai/gpt-4o",
            target_model="gemini/gemini-2.5-pro",
        )
        assert overlaps == [
            JudgeFamilyOverlap(judge_model="gemini/g-a", provider="google", roles=("target",)),
            JudgeFamilyOverlap(judge_model="openai/gpt-4.1", provider="openai", roles=("source",)),
            JudgeFamilyOverlap(judge_model="gemini/g-b", provider="google", roles=("target",)),
        ]

    def test_empty_when_no_judge_overlaps(self) -> None:
        assert (
            judge_family_overlaps(
                judge_models=["anthropic/claude-haiku-4-5"],
                source_model="openai/gpt-4o",
                target_model="gemini/gemini-2.5-pro",
            )
            == []
        )


class TestDescribeOverlap:
    def test_names_judge_roles_provider_and_the_bias(self) -> None:
        text = describe_overlap(
            JudgeFamilyOverlap(judge_model="gemini/g", provider="google", roles=("target",))
        )
        assert "gemini/g" in text
        assert "target" in text
        assert "google" in text
        assert "self-preference" in text

    def test_both_roles_are_joined(self) -> None:
        text = describe_overlap(
            JudgeFamilyOverlap(
                judge_model="gemini/g", provider="google", roles=("source", "target")
            )
        )
        assert "source and target" in text


class TestConfiguredJudgeModels:
    def test_collects_top_level_and_per_suite_judges_deduped(self) -> None:
        cfg = EvalShiftConfig.model_validate(
            {
                "prompts": [{"id": "p", "detection": "manual", "content": "hi"}],
                "evaluators": {
                    "llm_judge": [
                        {"criterion_name": "a", "criterion_prompt": "x", "judge_model": "j1"},
                        {"criterion_name": "b", "criterion_prompt": "y", "judge_model": "j1"},
                    ]
                },
                "suites": {
                    "s": {
                        "path": "s.jsonl",
                        "evaluators": {
                            "llm_judge": [
                                {
                                    "criterion_name": "c",
                                    "criterion_prompt": "z",
                                    "judge_model": "j2",
                                }
                            ]
                        },
                    }
                },
            }
        )
        assert configured_judge_models(cfg) == ["j1", "j2"]

    def test_empty_when_no_llm_judge_is_configured(self) -> None:
        cfg = EvalShiftConfig.model_validate(
            {"prompts": [{"id": "p", "detection": "manual", "content": "hi"}]}
        )
        assert configured_judge_models(cfg) == []
