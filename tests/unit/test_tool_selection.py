"""Tests for :class:`evalshift.evaluators.tool_selection.ToolSelectionEvaluator`."""

from __future__ import annotations

import pytest

from evalshift.config.models import ToolSelectionEvaluatorConfig
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.failures import TOOL_GROUND_TRUTH_MISS, TOOL_SELECTION_DRIFT
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.evaluators.tool_selection import (
    KIND_CONFORMANCE,
    KIND_DIVERGENCE,
    ToolSelectionEvaluator,
    _jaccard,
    _multiset_match,
    _sequence_match,
)
from evalshift.suite.models import ExpectedToolCall, SuiteExample
from tests.scoring_fixtures import (
    DIVERGENT_EXAMPLES,
    PROMPT_ID,
    RECORDED_TOOL_CALLS,
    RUN_ID,
    load_pairs,
    records_of,
)
from tests.unit.suite_examples import suite_example


def _trace(*tool_names: str) -> ToolTrace:
    return ToolTrace(
        calls=[
            ToolCall(tool_name=n, arguments={}, sequence_index=i) for i, n in enumerate(tool_names)
        ],
    )


def _example(
    *,
    expected: list[str] | None = None,
    expected_no_tools: bool = False,
    tags: list[str] | None = None,
) -> SuiteExample:
    return suite_example(
        id="ex1",
        inputs={},
        tags=tags or [],
        expected_tools=([ExpectedToolCall(tool_name=n) for n in expected] if expected else None),
        expected_no_tools=expected_no_tools,
    )


def _evaluator(
    *,
    conformance: str = "expected",
    divergence: str = "set",
    severity_floor: str | None = None,
) -> ToolSelectionEvaluator:
    """Build the evaluator with both axes on, as a real project gets it."""
    cfg = ToolSelectionEvaluatorConfig(
        name="tool_selection",
        conformance=conformance,  # type: ignore[arg-type]
        divergence=divergence,  # type: ignore[arg-type]
        severity_floor=severity_floor,  # type: ignore[arg-type]
    )
    return ToolSelectionEvaluator(cfg)


async def _axis(
    evaluator: ToolSelectionEvaluator,
    kind: str,
    *,
    example: SuiteExample,
    source: ToolTrace,
    target: ToolTrace,
) -> EvalRecord:
    """The one record ``kind`` produced, asserting it produced exactly one."""
    matching = [
        record
        for record in await evaluator.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=source,
            target_trace=target,
        )
        if record.kind == kind
    ]
    assert len(matching) == 1, f"expected one {kind} record, got {matching}"
    return matching[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSequenceMatch:
    def test_perfect_in_order(self) -> None:
        assert _sequence_match(["a", "b"], ["a", "b"]) == 1.0

    def test_partial_in_order(self) -> None:
        assert _sequence_match(["a", "b"], ["a"]) == 0.5

    def test_extra_intermediate_calls_ok(self) -> None:
        # Expected ["a","b"] is satisfied even with noise between.
        assert _sequence_match(["a", "b"], ["a", "x", "b"]) == 1.0

    def test_no_expected(self) -> None:
        assert _sequence_match([], []) == 1.0


class TestMultisetMatch:
    @pytest.mark.parametrize(
        ("expected", "actual", "score"),
        [
            (["a", "a", "b"], ["b", "a", "a"], 1.0),  # permutation is free
            (["a", "a", "b"], ["a", "b"], 2 / 3),  # one duplicate missing
            (["a", "a"], ["a", "a", "a"], 1.0),  # extra calls do not subtract
            (["archive", "archive", "get"], ["get"], 1 / 3),  # the run's real target
            ([], ["a"], 1.0),  # nothing expected
        ],
    )
    def test_multiset_match(self, expected: list[str], actual: list[str], score: float) -> None:
        assert _multiset_match(expected, actual) == pytest.approx(score)


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert _jaccard({"a"}, {"a"}) == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_overlap(self) -> None:
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert _jaccard(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# Axis: conformance (expected / expected_set / expected_no_tools)
# ---------------------------------------------------------------------------


class TestConformanceExpected:
    async def test_perfect_match(self) -> None:
        record = await _axis(
            _evaluator(),
            KIND_CONFORMANCE,
            example=_example(expected=["search_orders"]),
            source=_trace("search_orders"),
            target=_trace("search_orders"),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0
        assert record.delta == 0.0

    async def test_target_misses_expected_tool(self) -> None:
        record = await _axis(
            _evaluator(),
            KIND_CONFORMANCE,
            example=_example(expected=["notify_security_team"]),
            source=_trace("notify_security_team"),
            target=_trace("send_email"),  # wrong tool
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0
        assert record.delta == -1.0
        assert record.metadata["failure_categories"] == [TOOL_SELECTION_DRIFT]

    async def test_both_sides_missing_ground_truth_is_a_broken_harness(self) -> None:
        """Neither side conformed, so the delta is 0 — not a migration finding.

        The ground truth was captured from the source model; the source
        model failing it means the harness is wrong, and ``0.0 / 0.0`` reads
        as equivalent to every rate downstream unless it is labelled.
        """
        record = await _axis(
            _evaluator(),
            KIND_CONFORMANCE,
            example=_example(expected=["notify_security_team"]),
            source=_trace("send_email"),
            target=_trace("archive_project"),
        )
        assert (record.source_score, record.target_score, record.delta) == (0.0, 0.0, 0.0)
        assert record.metadata["failure_categories"] == [TOOL_GROUND_TRUTH_MISS]

    async def test_severity_floor_recorded(self) -> None:
        record = await _axis(
            _evaluator(severity_floor="high"),
            KIND_CONFORMANCE,
            example=_example(expected=["a"]),
            source=_trace("a"),
            target=_trace("a"),
        )
        assert record.metadata.get("severity_floor") == "high"


class TestConformanceExpectedSet:
    async def test_ignores_call_order(self) -> None:
        record = await _axis(
            _evaluator(conformance="expected_set"),
            KIND_CONFORMANCE,
            example=_example(expected=["archive_project", "archive_project", "get_projects"]),
            source=_trace("get_projects", "archive_project", "archive_project"),
            target=_trace("get_projects"),
        )
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(1 / 3)
        assert record.metadata["mode"] == "expected_set"

    async def test_a_single_correct_call_at_the_wrong_index_still_scores(self) -> None:
        # The defect this strategy fixes: `expected` walks the expected calls
        # in order, so a target that called one of them first scores 0.
        example = _example(expected=["archive_project", "get_projects"])
        ordered = await _axis(
            _evaluator(),
            KIND_CONFORMANCE,
            example=example,
            source=_trace("archive_project", "get_projects"),
            target=_trace("get_projects"),
        )
        unordered = await _axis(
            _evaluator(conformance="expected_set"),
            KIND_CONFORMANCE,
            example=example,
            source=_trace("archive_project", "get_projects"),
            target=_trace("get_projects"),
        )
        assert ordered.target_score == 0.0
        assert unordered.target_score == pytest.approx(0.5)


class TestConformanceExpectedNoTools:
    """``expected_no_tools`` is the conformance axis's *input*.

    It used to short-circuit the whole evaluator before the configured mode
    was read, which is what made a captured suite — where every promoted row
    carries it — score two divergent models as equivalent.
    """

    @pytest.mark.parametrize("conformance", ["expected", "expected_set"])
    async def test_grades_against_no_tools_under_either_strategy(self, conformance: str) -> None:
        record = await _axis(
            _evaluator(conformance=conformance),
            KIND_CONFORMANCE,
            example=_example(expected_no_tools=True),
            source=_trace(),
            target=_trace("send_email"),
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0
        assert record.metadata["mode"] == "expected_no_tools"

    async def test_target_compliant(self) -> None:
        record = await _axis(
            _evaluator(),
            KIND_CONFORMANCE,
            example=_example(expected_no_tools=True),
            source=_trace(),
            target=_trace(),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0

    async def test_it_no_longer_suppresses_the_divergence_axis(self) -> None:
        """The headline defect in miniature: the two models called different
        tools, both violated the recorded ground truth, and the run reported
        a zero delta because only the conformance answer was ever written."""
        records = await _evaluator().score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected_no_tools=True),
            source_trace=_trace("get_projects"),
            target_trace=_trace("add_note"),
        )
        assert {r.kind for r in records} == {KIND_CONFORMANCE, KIND_DIVERGENCE}
        by_kind = {r.kind: r for r in records}
        assert by_kind[KIND_CONFORMANCE].delta == 0.0
        assert by_kind[KIND_DIVERGENCE].delta == -1.0


# ---------------------------------------------------------------------------
# Axis: divergence (exact / set / first)
# ---------------------------------------------------------------------------


class TestDivergence:
    async def test_exact_mismatch(self) -> None:
        record = await _axis(
            _evaluator(divergence="exact"),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a", "b"),
            target=_trace("a", "c"),
        )
        assert record.target_score == 0.0

    async def test_exact_match(self) -> None:
        record = await _axis(
            _evaluator(divergence="exact"),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a", "b"),
            target=_trace("a", "b"),
        )
        assert record.target_score == 1.0

    async def test_set_jaccard(self) -> None:
        record = await _axis(
            _evaluator(divergence="set"),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a", "b"),
            target=_trace("b", "c"),
        )
        assert record.target_score == pytest.approx(1 / 3)

    async def test_the_default_does_not_read_reordering_as_drift(self) -> None:
        """Why `set` is the default rather than `exact`."""
        record = await _axis(
            _evaluator(),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a", "b"),
            target=_trace("b", "a"),
        )
        assert record.target_score == 1.0
        assert record.delta == 0.0

    async def test_first_only_first_matters(self) -> None:
        record = await _axis(
            _evaluator(divergence="first"),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a", "b"),
            target=_trace("a", "c"),
        )
        assert record.target_score == 1.0  # first matches → ok regardless of rest

    async def test_first_mismatch(self) -> None:
        record = await _axis(
            _evaluator(divergence="first"),
            KIND_DIVERGENCE,
            example=_example(),
            source=_trace("a"),
            target=_trace("b"),
        )
        assert record.target_score == 0.0

    async def test_the_source_is_never_blamed_for_missing_ground_truth(self) -> None:
        """Divergence has no ground truth in it, so it never tags the harness."""
        record = await _axis(
            _evaluator(),
            KIND_DIVERGENCE,
            example=_example(expected=["something_else"]),
            source=_trace("a"),
            target=_trace("b"),
        )
        assert record.metadata["failure_categories"] == [TOOL_SELECTION_DRIFT]


# ---------------------------------------------------------------------------
# Axis wiring
# ---------------------------------------------------------------------------


class TestAxesAreIndependent:
    async def test_both_axes_emit_by_default(self) -> None:
        evaluator = _evaluator()
        assert evaluator.kinds == (KIND_CONFORMANCE, KIND_DIVERGENCE)
        records = await evaluator.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=["a"]),
            source_trace=_trace("a"),
            target_trace=_trace("b"),
        )
        assert [r.kind for r in records] == [KIND_CONFORMANCE, KIND_DIVERGENCE]

    @pytest.mark.parametrize(
        ("conformance", "divergence", "expected_kinds"),
        [
            ("off", "set", (KIND_DIVERGENCE,)),
            ("expected", "off", (KIND_CONFORMANCE,)),
        ],
    )
    async def test_an_axis_switched_off_writes_no_row_and_is_never_attempted(
        self, conformance: str, divergence: str, expected_kinds: tuple[str, ...]
    ) -> None:
        evaluator = _evaluator(conformance=conformance, divergence=divergence)
        assert evaluator.kinds == expected_kinds
        records = await evaluator.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(expected=["a"]),
            source_trace=_trace("a"),
            target_trace=_trace("b"),
        )
        assert tuple(r.kind for r in records) == expected_kinds


# ---------------------------------------------------------------------------
# Regression fixture: the run that called a different tool nine times out of
# ten and reported behavioural equivalence
# ---------------------------------------------------------------------------


class TestProjectInsightsDivergence:
    """Pins ``r_20260820_project_insights_143a5f`` (see ``tests/scoring_fixtures``).

    The project configured ``mode: expected``. Every promoted row carries
    ``expected_no_tools: true``, so :meth:`score_pair` short-circuits into
    ``_score_no_tools`` before the configured mode is ever read, grades both
    sides *absolutely* against ground truth, and lands on ``0.0/0.0`` — a
    zero delta — for a pair where the two models called entirely different
    tools. Conformance and divergence are different questions and the
    evaluator currently answers only the first.
    """

    def test_fixture_matches_the_recorded_run(self) -> None:
        """The checked-in data still says what the plan says it says."""
        pairs = load_pairs()
        assert len(pairs) == 10
        assert {
            p.short_id: (p.source_trace.tool_names[0], p.target_trace.tool_names[0]) for p in pairs
        } == RECORDED_TOOL_CALLS
        # Tool-only turns: every output is empty, which is what makes every
        # text evaluator on this run a non-measurement.
        assert [p.source_text for p in pairs] == [""] * 10
        assert [p.target_text for p in pairs] == [""] * 10
        assert all(p.example.expected_no_tools for p in pairs)
        assert all(p.example.expected_tools is None for p in pairs)

    async def test_nine_of_ten_pairs_produce_a_negative_delta(self) -> None:
        """The headline defect: nine complete divergences, zero regressions.

        Fails today with an empty set — every pair scores ``0.0/0.0``.
        # S2: the divergence axis scores target against a source baseline of
        # 1.0, so each of the nine lands a -1.0 delta on its divergence record.
        """
        ev = _evaluator()  # the run configured ``mode: expected``, also the default
        regressed = {
            pair.short_id
            for pair in load_pairs()
            if any(
                record.delta < 0
                for record in records_of(
                    await ev.score_pair(
                        run_id=RUN_ID,
                        prompt_id=PROMPT_ID,
                        example=pair.example,
                        source_trace=pair.source_trace,
                        target_trace=pair.target_trace,
                    )
                )
            )
        }
        assert regressed == DIVERGENT_EXAMPLES

    async def test_a_divergence_is_categorised_as_a_failure(self) -> None:
        """``_record`` tags drift only on a *relative* drop, so 0.0/0.0 is silent.

        Fails today: no record on any of the nine carries a failure category.
        # S2: the divergence axis tags ``TOOL_SELECTION_DRIFT`` on any negative
        # delta, which is what puts the nine into ``failure_categories``.
        """
        ev = _evaluator()
        uncategorised = []
        for pair in load_pairs():
            if pair.short_id not in DIVERGENT_EXAMPLES:
                continue
            records = records_of(
                await ev.score_pair(
                    run_id=RUN_ID,
                    prompt_id=PROMPT_ID,
                    example=pair.example,
                    source_trace=pair.source_trace,
                    target_trace=pair.target_trace,
                )
            )
            if not any(
                TOOL_SELECTION_DRIFT in record.metadata.get("failure_categories", [])
                for record in records
            ):
                uncategorised.append(pair.short_id)
        assert uncategorised == []


class TestNothingMeasuredEmitsNothing:
    """An evaluator that measured nothing must write no row at all.

    A promoted capture carries no ``expected_tools``, so ground-truth modes
    have nothing to compare. They used to invent ``1.0/1.0`` and tag
    ``metadata["skipped"]``, and the row still landed in ``scores.jsonl``,
    the report, and the hosted bundle reading as full conformance.
    # S1: ``score_pair`` returns ``None`` and no record is written.
    """

    @pytest.mark.parametrize("conformance", ["expected", "expected_set"])
    async def test_no_conformance_record_without_ground_truth(self, conformance: str) -> None:
        records = records_of(
            await _evaluator(conformance=conformance, divergence="off").score_pair(
                run_id=RUN_ID,
                prompt_id=PROMPT_ID,
                example=_example(expected=None),
                source_trace=_trace("get_projects"),
                target_trace=_trace("add_note"),
            )
        )
        assert records == []

    async def test_the_divergence_axis_still_measures_without_ground_truth(self) -> None:
        """Divergence needs no ground truth — that is the point of the split.

        The pair below is the shape every row of the personalButler run had:
        no ``expected_tools`` to conform to, and two models calling entirely
        different tools.
        """
        records = records_of(
            await _evaluator().score_pair(
                run_id=RUN_ID,
                prompt_id=PROMPT_ID,
                example=_example(expected=None),
                source_trace=_trace("get_projects"),
                target_trace=_trace("add_note"),
            )
        )
        assert [(r.kind, r.delta) for r in records] == [(KIND_DIVERGENCE, -1.0)]
