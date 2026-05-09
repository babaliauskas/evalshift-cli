"""Unit tests for :mod:`evalshift.suite.models`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalshift.suite.models import Suite, SuiteExample

# ---------------------------------------------------------------------------
# SuiteExample
# ---------------------------------------------------------------------------


class TestSuiteExample:
    def test_minimal_construction(self) -> None:
        ex = SuiteExample(id="ex1")
        assert ex.id == "ex1"
        assert ex.inputs == {}
        assert ex.tags == []
        assert ex.expected is None

    def test_with_all_fields(self) -> None:
        ex = SuiteExample(
            id="ex1",
            inputs={"name": "Alex", "tone": "formal"},
            tags=["formal", "english"],
            expected={"summary": "Hi Alex"},
        )
        assert ex.inputs["name"] == "Alex"
        assert "english" in ex.tags
        assert ex.expected == {"summary": "Hi Alex"}

    def test_empty_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            SuiteExample(id="")

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SuiteExample.model_validate(
                {"id": "ex1", "inputs": {}, "rogue_key": True},
            )

    def test_inputs_accept_arbitrary_json_values(self) -> None:
        ex = SuiteExample(
            id="ex1",
            inputs={
                "string": "x",
                "number": 1.5,
                "bool": True,
                "null": None,
                "list": [1, 2, 3],
                "object": {"nested": "yes"},
            },
        )
        assert ex.inputs["object"]["nested"] == "yes"
        assert ex.inputs["list"] == [1, 2, 3]

    def test_equality(self) -> None:
        a = SuiteExample(id="ex1", inputs={"k": 1}, tags=["t"])
        b = SuiteExample(id="ex1", inputs={"k": 1}, tags=["t"])
        c = SuiteExample(id="ex2", inputs={"k": 1}, tags=["t"])
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class TestSuite:
    def test_default_is_empty(self) -> None:
        suite = Suite()
        assert len(suite) == 0
        assert suite.ids() == set()
        assert suite.by_tag("anything") == []

    def test_construction_from_examples(self) -> None:
        examples = [
            SuiteExample(id="ex1", inputs={"n": 1}),
            SuiteExample(id="ex2", inputs={"n": 2}),
        ]
        suite = Suite(examples=examples)
        assert len(suite) == 2
        assert suite.ids() == {"ex1", "ex2"}

    def test_by_tag_returns_only_matching_examples(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="a", tags=["formal"]),
                SuiteExample(id="b", tags=["casual"]),
                SuiteExample(id="c", tags=["formal", "english"]),
                SuiteExample(id="d", tags=[]),
            ],
        )
        formal = suite.by_tag("formal")
        assert {e.id for e in formal} == {"a", "c"}

    def test_by_tag_supports_multi_tag_membership(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="a", tags=["formal", "english"]),
            ],
        )
        # Same example appears in both slices.
        assert suite.by_tag("formal") == suite.by_tag("english")

    def test_by_tag_returns_examples_in_suite_order(self) -> None:
        suite = Suite(
            examples=[
                SuiteExample(id="z", tags=["t"]),
                SuiteExample(id="a", tags=["t"]),
                SuiteExample(id="m", tags=["t"]),
            ],
        )
        assert [e.id for e in suite.by_tag("t")] == ["z", "a", "m"]

    def test_by_tag_unknown_tag_returns_empty(self) -> None:
        suite = Suite(examples=[SuiteExample(id="a", tags=["x"])])
        assert suite.by_tag("y") == []

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate example ids: \['ex1'\]"):
            Suite(
                examples=[
                    SuiteExample(id="ex1"),
                    SuiteExample(id="ex2"),
                    SuiteExample(id="ex1"),
                ],
            )

    def test_round_trip_through_dump(self) -> None:
        original = Suite(
            examples=[
                SuiteExample(id="a", inputs={"k": 1}, tags=["t"]),
                SuiteExample(id="b", inputs={"k": 2}),
            ],
        )
        recreated = Suite.model_validate(original.model_dump())
        assert recreated == original

    def test_extra_top_level_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Suite.model_validate(
                {"examples": [{"id": "a"}], "rogue": 42},
            )
