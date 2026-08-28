"""Tests for the shared suite helpers in ``cli.commands._suites``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift.captures.toolset import EMPTY_TOOLSET_FINGERPRINT
from evalshift.cli.commands._suites import (
    SUITE_FILENAME,
    AmbiguousSuiteError,
    UnknownSuiteNameError,
    derive_suite_evaluators,
    derive_suite_slug,
    parse_suites_region,
    render_suites_region,
    render_suites_yaml,
    resolve_suite_path,
    suite_entry_payload,
)
from evalshift.config.models import EvalShiftConfig
from evalshift.evaluators.tool_models import ToolSpec
from evalshift.suite.models import ExpectedToolCall, SuiteExample


def _config(**extra: Any) -> EvalShiftConfig:
    return EvalShiftConfig.model_validate(
        {
            "version": 1,
            "prompts": [{"id": "p", "detection": "manual", "content": "Hi {q}"}],
            **extra,
        },
    )


class TestResolveSuitePath:
    _CFG_PATH = Path("/proj/evalshift.yaml")

    def _resolve(self, cfg: EvalShiftConfig, **kw: Any) -> Path:
        return resolve_suite_path(
            suite_path=kw.get("suite_path"),
            suite_name=kw.get("suite_name"),
            cfg=cfg,
            config_path=self._CFG_PATH,
        )

    def test_explicit_suite_path_wins(self) -> None:
        cfg = _config(suites={"a": {"path": "a.jsonl"}, "b": {"path": "b.jsonl"}})
        got = self._resolve(cfg, suite_path=Path("custom.jsonl"))
        assert got == Path("custom.jsonl")

    def test_suite_name_resolves_relative_to_config(self) -> None:
        cfg = _config(suites={"promoted": {"path": ".evalshift/suites/x/golden.jsonl"}})
        got = self._resolve(cfg, suite_name="promoted")
        assert got == Path("/proj/.evalshift/suites/x/golden.jsonl")

    def test_unknown_suite_name_raises(self) -> None:
        cfg = _config(suites={"promoted": {"path": "x.jsonl"}})
        with pytest.raises(UnknownSuiteNameError):
            self._resolve(cfg, suite_name="missing")

    def test_single_wired_suite_auto_selected(self) -> None:
        # Bare `evalshift all` after capture sync: exactly one suite, no flag.
        cfg = _config(suites={"only": {"path": ".evalshift/suites/only/golden.jsonl"}})
        got = self._resolve(cfg)
        assert got == Path("/proj/.evalshift/suites/only/golden.jsonl")

    def test_multiple_wired_suites_are_ambiguous(self) -> None:
        cfg = _config(suites={"a": {"path": "a.jsonl"}, "b": {"path": "b.jsonl"}})
        with pytest.raises(AmbiguousSuiteError) as excinfo:
            self._resolve(cfg)
        # Error names the choices and the fix.
        assert "a" in str(excinfo.value)
        assert "--suite-name" in str(excinfo.value)

    def test_no_suites_falls_back_to_golden_file(self) -> None:
        cfg = _config()
        got = self._resolve(cfg)
        assert got == Path(SUITE_FILENAME)


class TestDeriveSuiteSlug:
    def test_prefers_explicit_suite_name(self) -> None:
        slug = derive_suite_slug(
            suite_name="note_search",
            suite_path=Path(".evalshift/suites/note_search/golden.jsonl"),
        )
        assert slug == "note_search"

    def test_uses_parent_dir_for_golden_layout(self) -> None:
        slug = derive_suite_slug(
            suite_name=None,
            suite_path=Path(".evalshift/suites/note_search/golden.jsonl"),
        )
        assert slug == "note_search"

    def test_falls_back_to_file_stem(self) -> None:
        slug = derive_suite_slug(
            suite_name=None,
            suite_path=Path("datasets/smoke.jsonl"),
        )
        assert slug == "smoke"


# ---------------------------------------------------------------------------
# derive_suite_evaluators
# ---------------------------------------------------------------------------


def _example(idx: int, **kw: Any) -> SuiteExample:
    """A minimal suite row; ``kw`` carries the toolset/ground-truth under test."""
    return SuiteExample(id=f"ex{idx}", inputs={"q": "hi"}, **kw)


def _tool_row(idx: int, **kw: Any) -> SuiteExample:
    """A row whose agent was offered one tool."""
    return _example(idx, tools=[ToolSpec(name="search_orders")], **kw)


class TestDeriveSuiteEvaluators:
    def test_no_tools_offered_yields_no_block(self) -> None:
        """A prose suite gets no tool evaluators — an empty denominator is not a verdict."""
        assert derive_suite_evaluators([_example(1, tools=[]), _example(2, tools=[])]) is None

    def test_empty_toolset_ref_yields_no_block(self) -> None:
        """The ``toolset_ref`` spelling of "no tools offered" reads the same."""
        rows = [_example(1, toolset_ref=EMPTY_TOOLSET_FINGERPRINT)]
        assert derive_suite_evaluators(rows) is None

    def test_no_rows_yields_no_block(self) -> None:
        assert derive_suite_evaluators([]) is None

    def test_any_toolset_row_yields_tool_selection(self) -> None:
        override = derive_suite_evaluators([_example(1, tools=[]), _tool_row(2)])

        assert override is not None
        assert override.tool_selection is not None
        (selection,) = override.tool_selection
        assert selection.name == "routing"
        assert selection.conformance == "expected"
        # `set`, not `off`: reordered identical calls are the same behaviour,
        # but two models failing the ground truth differently is not.
        assert selection.divergence == "set"

    def test_toolset_without_recorded_arguments_yields_no_tool_arguments(self) -> None:
        rows = [_tool_row(1, expected_tools=[ExpectedToolCall(tool_name="search_orders")])]
        override = derive_suite_evaluators(rows)

        assert override is not None
        assert override.tool_arguments is None

    def test_recorded_arguments_yield_tool_arguments(self) -> None:
        rows = [
            _tool_row(1, expected_tools=[ExpectedToolCall(tool_name="search_orders")]),
            _tool_row(
                2,
                expected_tools=[
                    ExpectedToolCall(tool_name="search_orders", arguments={"customer_id": "c42"}),
                ],
            ),
        ]
        override = derive_suite_evaluators(rows)

        assert override is not None
        assert override.tool_arguments is not None
        (arguments,) = override.tool_arguments
        assert arguments.name == "routing_args"
        assert arguments.against == "expected"
        # No `strategies:` — `default_strategy: auto` already grades free text.
        assert arguments.strategies == {}

    def test_declares_nothing_about_the_families_it_does_not_generate(self) -> None:
        """Undeclared families inherit the top-level block (see ``evaluators_for``)."""
        override = derive_suite_evaluators([_tool_row(1)])

        assert override is not None
        assert override.semantic is None
        assert override.llm_judge is None
        assert override.structural is None

    def test_names_are_stable_across_calls(self) -> None:
        """Reports key on evaluator names across runs, so regeneration must not rename."""
        rows = [
            _tool_row(
                1,
                expected_tools=[
                    ExpectedToolCall(tool_name="search_orders", arguments={"customer_id": "c42"}),
                ],
            ),
        ]
        first = derive_suite_evaluators(rows)
        second = derive_suite_evaluators(rows)

        assert first is not None
        assert second is not None
        assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# suites: region rendering
# ---------------------------------------------------------------------------


class TestSuiteEntryPayload:
    def test_omits_evaluators_when_none(self) -> None:
        payload = suite_entry_payload(path="suites/alpha/golden.jsonl", evaluators=None)
        assert payload == {"source": "captured", "path": "suites/alpha/golden.jsonl"}

    def test_carries_only_the_generated_keys(self) -> None:
        """Defaults stay unwritten so a future default change still reaches the config."""
        override = derive_suite_evaluators([_tool_row(1)])
        payload = suite_entry_payload(path="p.jsonl", evaluators=override)

        assert payload["evaluators"] == {
            "tool_selection": [
                {"name": "routing", "conformance": "expected", "divergence": "set"},
            ],
        }


class TestRenderSuitesYaml:
    def test_empty_mapping(self) -> None:
        assert render_suites_yaml({}).strip() == "suites: {}"

    def test_indents_sequences_under_their_key(self) -> None:
        rendered = render_suites_yaml(
            {
                "alpha": suite_entry_payload(
                    path="p.jsonl", evaluators=derive_suite_evaluators([_tool_row(1)])
                )
            },
        )

        assert rendered.splitlines() == [
            "suites:",
            "  alpha:",
            "    source: captured",
            "    path: p.jsonl",
            "    evaluators:",
            "      tool_selection:",
            "        - name: routing",
            "          conformance: expected",
            "          divergence: set",
        ]

    def test_sorted_by_suite_name(self) -> None:
        rendered = render_suites_yaml(
            {
                "beta": suite_entry_payload(path="b.jsonl", evaluators=None),
                "alpha": suite_entry_payload(path="a.jsonl", evaluators=None),
            },
        )
        assert rendered.index("alpha") < rendered.index("beta")

    def test_round_trips_through_the_region_parser(self) -> None:
        entries = {
            "alpha": suite_entry_payload(
                path="a.jsonl",
                evaluators=derive_suite_evaluators([_tool_row(1)]),
            ),
            "beta": suite_entry_payload(path="b.jsonl", evaluators=None),
        }
        rendered = render_suites_yaml(entries)

        assert parse_suites_region(render_suites_region(rendered)) == entries
        # ...and re-rendering what was parsed is byte-identical (idempotency).
        assert render_suites_yaml(parse_suites_region(render_suites_region(rendered))) == rendered


class TestParseSuitesRegion:
    def test_missing_markers_yield_no_entries(self) -> None:
        assert parse_suites_region("version: 1\n") == {}

    def test_empty_region_yields_no_entries(self) -> None:
        assert parse_suites_region(render_suites_region("suites: {}")) == {}

    def test_ignores_config_outside_the_region(self) -> None:
        """Only the managed region is ours to carry forward."""
        text = "suites:\n  hand_written:\n    path: h.jsonl\n\n" + render_suites_region(
            "suites:\n  alpha:\n    source: captured\n    path: a.jsonl"
        )
        assert parse_suites_region(text) == {
            "alpha": {"source": "captured", "path": "a.jsonl"},
        }

    def test_unparseable_region_yields_no_entries(self) -> None:
        assert parse_suites_region(render_suites_region("suites: [oops")) == {}
