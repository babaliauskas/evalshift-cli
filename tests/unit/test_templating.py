"""Tests for :mod:`evalshift.utils.templating`."""

from __future__ import annotations

import pytest

from evalshift.parsers.base import PromptTemplate
from evalshift.suite.models import Suite, SuiteExample
from evalshift.utils.templating import (
    CompatibilityIssue,
    MissingTemplateVariableError,
    SuiteCompatibilityError,
    extract_variables,
    render,
    validate_suite_against_prompts,
)

# ---------------------------------------------------------------------------
# extract_variables
# ---------------------------------------------------------------------------


class TestExtractVariables:
    def test_no_placeholders(self) -> None:
        assert extract_variables("plain text") == set()

    def test_single_placeholder(self) -> None:
        assert extract_variables("Hello {name}") == {"name"}

    def test_multiple_placeholders(self) -> None:
        assert extract_variables("{a} and {b} and {a}") == {"a", "b"}

    def test_escaped_braces_are_literal(self) -> None:
        # `{{name}}` renders to `{name}` and must NOT count as a variable.
        assert extract_variables("This is {{name}}, not a placeholder") == set()

    def test_attribute_access_uses_root_name(self) -> None:
        assert extract_variables("Hi {user.name}") == {"user"}

    def test_index_access_uses_root_name(self) -> None:
        assert extract_variables("First: {items[0]}") == {"items"}

    def test_format_spec_passthrough(self) -> None:
        # A format spec like ':>10' shouldn't affect variable detection.
        assert extract_variables("Padded: {n:>10}") == {"n"}

    def test_conversion_flags_passthrough(self) -> None:
        assert extract_variables("Repr: {x!r}") == {"x"}

    def test_empty_and_positional_placeholders_ignored(self) -> None:
        # EvalShift templates use named placeholders only; positional
        # patterns shouldn't surface as variable names.
        assert extract_variables("{} and {0} and {1}") == set()


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    def test_happy_path(self) -> None:
        assert render("Hello {name}", {"name": "Alex"}) == "Hello Alex"

    def test_extra_keys_are_ignored(self) -> None:
        assert render("Hello {name}", {"name": "Alex", "extra": "ignored"}) == "Hello Alex"

    def test_escaped_braces_are_preserved(self) -> None:
        assert render("Use {{like this}}", {}) == "Use {like this}"

    def test_missing_single_variable_raises(self) -> None:
        with pytest.raises(MissingTemplateVariableError) as info:
            render("Hello {name}", {})
        assert info.value.missing == {"name"}

    def test_missing_multiple_variables_collected(self) -> None:
        with pytest.raises(MissingTemplateVariableError) as info:
            render("{a} {b} {c}", {"a": "1"})
        assert info.value.missing == {"b", "c"}

    def test_non_string_values_are_stringified(self) -> None:
        assert render("n={n}", {"n": 42}) == "n=42"

    def test_missing_template_variable_error_is_a_keyerror(self) -> None:
        # KeyError subclassing means existing `except KeyError` paths
        # still catch missing-template-variable failures.
        with pytest.raises(KeyError):
            render("{x}", {})


# ---------------------------------------------------------------------------
# validate_suite_against_prompts
# ---------------------------------------------------------------------------


def _template(
    template_id: str,
    content: str,
    declared: list[str] | None = None,
) -> PromptTemplate:
    return PromptTemplate(
        id=template_id,
        content=content,
        declared_variables=declared or [],
    )


class TestValidateSuiteAgainstPrompts:
    def test_happy_path_every_pair_compatible(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="ex1", inputs={"name": "Alex", "tone": "formal"}),
                SuiteExample(id="ex2", inputs={"name": "Sam", "tone": "casual"}),
            ],
        )
        templates = [_template("greet", "Hello {name} ({tone})", ["name", "tone"])]
        # Should not raise.
        validate_suite_against_prompts(suite, templates)

    def test_no_templates_or_examples_is_a_noop(self) -> None:
        validate_suite_against_prompts(Suite(), [])

    def test_one_missing_var_in_one_example(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="ex1", inputs={"name": "Alex"}),
                SuiteExample(id="ex2", inputs={"name": "Sam", "tone": "casual"}),
            ],
        )
        templates = [_template("greet", "Hi {name} ({tone})", ["name", "tone"])]
        with pytest.raises(SuiteCompatibilityError) as info:
            validate_suite_against_prompts(suite, templates)
        assert len(info.value.issues) == 1
        issue = info.value.issues[0]
        assert issue.prompt_id == "greet"
        assert issue.example_id == "ex1"
        assert issue.missing == frozenset({"tone"})

    def test_multiple_issues_collected_at_once(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="ex1", inputs={}),
                SuiteExample(id="ex2", inputs={"a": 1}),
            ],
        )
        templates = [
            _template("p1", "{a} and {b}", ["a", "b"]),
            _template("p2", "{x}", ["x"]),
        ]
        with pytest.raises(SuiteCompatibilityError) as info:
            validate_suite_against_prompts(suite, templates)
        # 2 prompts x 2 examples = 4 pairs; every pair has at least one missing.
        assert len(info.value.issues) == 4

    def test_declared_variables_treated_as_required(self) -> None:
        # If the user *declares* `extra` in `variables:` even though it's
        # not in the template body, we still treat it as required so the
        # config and the suite stay in sync.
        suite = Suite(
            examples=[SuiteExample(id="ex1", inputs={"name": "Alex"})],
        )
        templates = [_template("greet", "Hi {name}", ["name", "extra"])]
        with pytest.raises(SuiteCompatibilityError) as info:
            validate_suite_against_prompts(suite, templates)
        assert info.value.issues[0].missing == frozenset({"extra"})

    def test_extra_inputs_in_example_are_allowed(self) -> None:
        suite = Suite(
            examples=[SuiteExample(id="ex1", inputs={"name": "Alex", "extra": "ok"})],
        )
        templates = [_template("greet", "Hi {name}", ["name"])]
        validate_suite_against_prompts(suite, templates)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


class TestErrorFormatting:
    def test_format_plain_lists_each_issue(self) -> None:
        err = SuiteCompatibilityError(
            issues=[
                CompatibilityIssue(
                    prompt_id="p1",
                    example_id="ex1",
                    missing=frozenset({"a", "b"}),
                ),
                CompatibilityIssue(
                    prompt_id="p2",
                    example_id="ex2",
                    missing=frozenset({"x"}),
                ),
            ],
        )
        text = err.format_plain()
        assert "p1" in text and "ex1" in text
        assert "a, b" in text
        assert "p2" in text and "ex2" in text

    def test_str_uses_plain_format(self) -> None:
        err = SuiteCompatibilityError(
            issues=[
                CompatibilityIssue("p", "e", frozenset({"x"})),
            ],
        )
        assert "missing {x}" in str(err)

    def test_format_rich_smoke(self) -> None:
        # Just check it returns a renderable without exploding.
        err = SuiteCompatibilityError(
            issues=[CompatibilityIssue("p", "e", frozenset({"x"}))],
        )
        assert err.format_rich() is not None
