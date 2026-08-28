"""Tests for the human-readable failure-category vocabulary."""

from __future__ import annotations

from evalshift.evaluators import failures
from evalshift.evaluators.failures import CATEGORY_LABELS, category_label


class TestCategoryLabels:
    def test_every_declared_category_has_a_label(self) -> None:
        """Each machine label in ``__all__`` maps to plain-language prose."""
        declared = [name for name in failures.__all__ if name.isupper() and name in vars(failures)]
        categories = [
            name for name in declared if name not in {"BROKEN_HARNESS_CAUSES", "CATEGORY_LABELS"}
        ]
        assert categories, "sanity: the module declares categories"
        for name in categories:
            label = CATEGORY_LABELS[getattr(failures, name)]
            assert label
            assert "_" not in label, f"{name} label still reads as an identifier: {label!r}"

    def test_label_lookup_returns_the_mapping(self) -> None:
        assert category_label(failures.TOOL_SELECTION_DRIFT) == "Different tools chosen"

    def test_unknown_category_is_humanised_not_echoed(self) -> None:
        """A category this version doesn't know still renders as words."""
        assert category_label("SOME_FUTURE_DRIFT") == "Some future drift"

    def test_unknown_snake_case_is_humanised(self) -> None:
        assert category_label("missing_field") == "Missing field"
