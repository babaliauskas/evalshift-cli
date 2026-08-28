"""Tests for :class:`evalshift.evaluators.tool_arguments.ToolArgumentsEvaluator`."""

from __future__ import annotations

from typing import Any

import pytest

from evalshift.config.models import ToolArgumentsEvaluatorConfig
from evalshift.evaluators.failures import ARGUMENT_VALUE_DRIFT
from evalshift.evaluators.tool_arguments import (
    ToolArgumentsEvaluator,
    _is_subset,
    _match_calls,
    _schema_strategy,
)
from evalshift.evaluators.tool_models import ToolCall, ToolSpec, ToolTrace
from evalshift.suite.models import ExpectedToolCall, SuiteExample
from tests.unit.suite_examples import suite_example


def _trace(*calls: tuple[str, dict[str, Any]]) -> ToolTrace:
    return ToolTrace(
        calls=[
            ToolCall(tool_name=name, arguments=args, sequence_index=i)
            for i, (name, args) in enumerate(calls)
        ],
    )


def _ev(
    *,
    strategies: dict[str, str] | None = None,
    default_strategy: str = "exact",
    numeric_tolerance: float = 0.05,
    embeddings_fn: Any = None,
) -> ToolArgumentsEvaluator:
    cfg = ToolArgumentsEvaluatorConfig(
        name="tool_arguments",
        strategies=strategies or {},  # type: ignore[arg-type]
        default_strategy=default_strategy,  # type: ignore[arg-type]
        numeric_tolerance=numeric_tolerance,
    )
    return ToolArgumentsEvaluator(cfg, embeddings_fn=embeddings_fn)


def _example() -> SuiteExample:
    return suite_example(id="ex1", inputs={})


def _resolve_add_event(_example: SuiteExample) -> list[ToolSpec]:
    """A stand-in for the toolset resolver ``evaluate.py`` injects."""
    return [_add_event_spec()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestIsSubset:
    def test_dict_subset(self) -> None:
        assert _is_subset({"a": 1}, {"a": 1, "b": 2})

    def test_dict_not_subset(self) -> None:
        assert not _is_subset({"a": 1, "c": 3}, {"a": 1, "b": 2})

    def test_nested_dict(self) -> None:
        assert _is_subset({"x": {"y": 1}}, {"x": {"y": 1, "z": 2}})

    def test_list_subset_unordered(self) -> None:
        assert _is_subset([1, 2], [3, 1, 2])

    def test_scalar_equality(self) -> None:
        assert _is_subset("foo", "foo")
        assert not _is_subset("foo", "bar")


class TestMatchCalls:
    def test_one_to_one(self) -> None:
        s = _trace(("a", {}), ("b", {}))
        t = _trace(("a", {}), ("b", {}))
        matched = _match_calls(s, t)
        assert len(matched) == 2
        assert [m[0].tool_name for m in matched] == ["a", "b"]

    def test_unmatched_dropped(self) -> None:
        s = _trace(("a", {}), ("b", {}))
        t = _trace(("a", {}))  # no b in target
        matched = _match_calls(s, t)
        assert len(matched) == 1

    def test_repeated_tool_pairs_by_index(self) -> None:
        s = _trace(("a", {"q": 1}), ("a", {"q": 2}))
        t = _trace(("a", {"q": 1}), ("a", {"q": 2}))
        matched = _match_calls(s, t)
        # Greedy nearest-index → src[0]→tgt[0], src[1]→tgt[1]
        assert matched[0][0].arguments == {"q": 1}
        assert matched[0][1].arguments == {"q": 1}
        assert matched[1][0].arguments == {"q": 2}
        assert matched[1][1].arguments == {"q": 2}


# ---------------------------------------------------------------------------
# Score scenarios
# ---------------------------------------------------------------------------


class TestExactStrategy:
    async def test_perfect_match(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "ACME"})),
        )
        assert record.target_score == 1.0
        assert record.source_score == 1.0

    async def test_value_drift(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "Initech"})),
        )
        assert record.target_score == 0.0
        assert record.source_score == 1.0


class TestSubsetStrategy:
    async def test_target_has_extras_passes(self) -> None:
        ev = _ev(strategies={"q": "subset"})
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "ACME"})),
        )
        assert record.target_score == 1.0


class TestNumericStrategy:
    async def test_within_tolerance(self) -> None:
        ev = _ev(strategies={"qty": "numeric"}, numeric_tolerance=0.1)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 100})),
            target_trace=_trace(("ship", {"qty": 105})),
        )
        # 5% relative error within 10% tolerance → high score.
        assert record.target_score > 0.4

    async def test_outside_tolerance(self) -> None:
        ev = _ev(strategies={"qty": "numeric"}, numeric_tolerance=0.05)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 10})),
            target_trace=_trace(("ship", {"qty": 100})),  # huge drift
        )
        assert record.target_score == 0.0


class TestSemanticStrategy:
    async def test_falls_back_to_exact_when_no_embeddings(self) -> None:
        ev = _ev(strategies={"text": "semantic"}, embeddings_fn=None)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"text": "hello"})),
            target_trace=_trace(("ping", {"text": "hello"})),
        )
        assert record.target_score == 1.0

    async def test_uses_embeddings_fn(self) -> None:
        async def fake_emb(a: str, b: str) -> float:
            return 0.92

        ev = _ev(strategies={"text": "semantic"}, embeddings_fn=fake_emb)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"text": "hello"})),
            target_trace=_trace(("ping", {"text": "hi"})),
        )
        assert record.target_score == pytest.approx(0.92)


class TestAutoStrategy:
    """``auto`` grades free text instead of asking for byte equality.

    The ladder is: normalized exact → graded similarity (embeddings when a
    semantic evaluator lent us a model, ``difflib`` when it did not) for
    strings; ``numeric`` for numbers; ``subset`` for containers; ``exact``
    for everything else.
    """

    async def test_case_and_whitespace_differences_score_full_marks(self) -> None:
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "find people"})),
            target_trace=_trace(("search", {"q": "Find  people"})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_the_reference_capitalization_case_scores_one(self) -> None:
        """Regression: ``cap_177fa25b`` scored ``delta -0.5`` on this pair."""
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "find people that would use Evalshift"})),
            target_trace=_trace(("search", {"q": "Find people that would use Evalshift"})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_distinct_strings_get_partial_credit_without_embeddings(self) -> None:
        """No embedding model configured must not collapse to a 0/1 verdict."""
        ev = _ev(default_strategy="auto", embeddings_fn=None)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "quarterly revenue report"})),
            target_trace=_trace(("search", {"q": "quarterly revenue summary"})),
        )
        assert 0.0 < record.target_score < 1.0

    async def test_unrelated_strings_still_score_near_zero(self) -> None:
        ev = _ev(default_strategy="auto", embeddings_fn=None)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "ACME"})),
            target_trace=_trace(("search", {"q": "zzzz"})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_distinct_strings_use_embeddings_when_available(self) -> None:
        calls: list[tuple[str, str]] = []

        async def fake_emb(a: str, b: str) -> float:
            calls.append((a, b))
            return 0.88

        ev = _ev(default_strategy="auto", embeddings_fn=fake_emb)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "weather in Madrid tomorrow"})),
            target_trace=_trace(("search", {"q": "Madrid forecast for Tuesday"})),
        )
        assert calls == [("weather in Madrid tomorrow", "Madrid forecast for Tuesday")]
        assert record.target_score == pytest.approx(0.88)

    async def test_embedding_similarity_is_clamped(self) -> None:
        async def wild_emb(a: str, b: str) -> float:
            return 1.7

        ev = _ev(default_strategy="auto", embeddings_fn=wild_emb)
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "alpha"})),
            target_trace=_trace(("search", {"q": "beta"})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_equal_numbers_score_one(self) -> None:
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 100})),
            target_trace=_trace(("ship", {"qty": 100})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_wildly_different_numbers_score_zero(self) -> None:
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"qty": 10})),
            target_trace=_trace(("ship", {"qty": 100})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_booleans_are_compared_exactly_not_numerically(self) -> None:
        """``True`` is not ``1`` here — a flag flip is a wrong value."""
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ship", {"express": True})),
            target_trace=_trace(("ship", {"express": False})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_containers_are_compared_as_subsets(self) -> None:
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("filter", {"tags": ["a"], "where": {"x": 1}})),
            target_trace=_trace(("filter", {"tags": ["a", "b"], "where": {"x": 1, "y": 2}})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_mismatched_types_fall_back_to_exact(self) -> None:
        ev = _ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("filter", {"limit": 10})),
            target_trace=_trace(("filter", {"limit": "10"})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_per_field_strategy_overrides_the_default(self) -> None:
        ev = _ev(default_strategy="auto", strategies={"q": "exact"})
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "find people"})),
            target_trace=_trace(("search", {"q": "Find people"})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_exact_default_reproduces_the_pre_auto_behaviour(self) -> None:
        ev = _ev(default_strategy="exact")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "find people"})),
            target_trace=_trace(("search", {"q": "Find people"})),
        )
        assert record.target_score == pytest.approx(0.0)

    async def test_the_shipped_config_default_is_the_auto_ladder(self) -> None:
        """A config the user never touched must already grade free text."""
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="tool_arguments"))
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("search", {"q": "find people"})),
            target_trace=_trace(("search", {"q": "Find people"})),
        )
        assert record.target_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Schema-informed dispatch
# ---------------------------------------------------------------------------


def _add_event_spec() -> ToolSpec:
    """A toolset the way a capture records one — the ``add_event`` reference tool."""
    return ToolSpec(
        name="add_event",
        description="Create a calendar event.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "format": "date-time"},
                "attendee_ids": {"type": "array", "items": {"type": "string"}},
                "location": {"type": "object"},
                "project_id": {"type": "integer"},
                "duration_minutes": {"type": "integer"},
                "status": {"type": "string", "enum": ["confirmed", "tentative"]},
                "owner_id": {"type": "string"},
                "seats": {"type": ["integer", "null"]},
            },
        },
    )


class TestSchemaStrategy:
    """``_schema_strategy`` reads the recorded JSON Schema, nothing else."""

    @staticmethod
    def _strategy(field: str, *, tool: str = "add_event") -> str | None:
        return _schema_strategy([_add_event_spec()], tool, field)

    def test_date_time_format_is_exact(self) -> None:
        assert self._strategy("start_time") == "exact"

    def test_enum_string_is_exact(self) -> None:
        assert self._strategy("status") == "exact"

    def test_id_suffixed_string_is_exact(self) -> None:
        """A free-text ladder would give a wrong id partial credit."""
        assert self._strategy("owner_id") == "exact"

    def test_id_suffixed_integer_is_exact_not_numeric(self) -> None:
        """Numeric tolerance on an identifier is nonsense — 41 is not nearly 42."""
        assert self._strategy("project_id") == "exact"

    def test_plain_integer_is_numeric(self) -> None:
        assert self._strategy("duration_minutes") == "numeric"

    def test_nullable_integer_is_numeric(self) -> None:
        """``type: [integer, null]`` is an integer with a null option."""
        assert self._strategy("seats") == "numeric"

    def test_object_is_subset(self) -> None:
        assert self._strategy("location") == "subset"

    def test_array_is_subset(self) -> None:
        """Containers compare structurally even when the name says ids.

        ``_is_subset`` compares leaves with ``==``, so each id inside is
        still matched exactly; only the extras are forgiven.
        """
        assert self._strategy("attendee_ids") == "subset"

    def test_free_string_falls_through(self) -> None:
        assert self._strategy("title") is None

    def test_unknown_field_falls_through(self) -> None:
        assert self._strategy("nonexistent") is None

    def test_unknown_tool_falls_through(self) -> None:
        assert self._strategy("title", tool="delete_event") is None

    def test_no_toolset_falls_through(self) -> None:
        assert _schema_strategy(None, "add_event", "start_time") is None

    def test_empty_toolset_falls_through(self) -> None:
        assert _schema_strategy([], "add_event", "start_time") is None


class TestAutoSchemaDispatch:
    """``auto`` step 2: the toolset the capture recorded picks the strategy."""

    @staticmethod
    def _ev_with_schema(resolver: Any = _resolve_add_event) -> ToolArgumentsEvaluator:
        return ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name="tool_arguments"),
            toolset_resolver=resolver,
        )

    @staticmethod
    async def _score(ev: ToolArgumentsEvaluator, src: Any, tgt: Any, *, field: str) -> float:
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("add_event", {field: src})),
            target_trace=_trace(("add_event", {field: tgt})),
        )
        assert record is not None
        return record.target_score

    async def test_a_reworded_timestamp_is_wrong_not_similar(self) -> None:
        score = await self._score(
            self._ev_with_schema(),
            "2026-01-02T09:00:00Z",
            "2026-01-02T09:00:00+00:00",
            field="start_time",
        )
        assert score == pytest.approx(0.0)

    async def test_a_wrong_enum_value_scores_zero(self) -> None:
        score = await self._score(self._ev_with_schema(), "confirmed", "tentative", field="status")
        assert score == pytest.approx(0.0)

    async def test_a_near_miss_id_scores_zero(self) -> None:
        score = await self._score(
            self._ev_with_schema(), "usr_abc123", "usr_abc124", field="owner_id"
        )
        assert score == pytest.approx(0.0)

    async def test_a_close_integer_gets_numeric_tolerance(self) -> None:
        score = await self._score(self._ev_with_schema(), 60, 61, field="duration_minutes")
        assert 0.0 < score < 1.0

    async def test_a_container_is_scored_as_a_subset(self) -> None:
        score = await self._score(
            self._ev_with_schema(), ["u1"], ["u1", "u2"], field="attendee_ids"
        )
        assert score == pytest.approx(1.0)

    async def test_a_free_string_still_falls_through_to_graded_similarity(self) -> None:
        score = await self._score(
            self._ev_with_schema(), "quarterly review", "quarterly review sync", field="title"
        )
        assert 0.0 < score < 1.0

    async def test_normalized_exact_still_wins_before_the_schema(self) -> None:
        """Step 1 runs first: a cosmetic difference is never a wrong value."""
        score = await self._score(self._ev_with_schema(), "Confirmed", "confirmed", field="status")
        assert score == pytest.approx(1.0)

    async def test_no_resolver_falls_back_to_the_phase_one_ladder(self) -> None:
        score = await self._score(
            self._ev_with_schema(resolver=None),
            "2026-01-02T09:00:00Z",
            "2026-01-02T09:00:00+00:00",
            field="start_time",
        )
        assert 0.0 < score < 1.0

    async def test_a_resolver_returning_none_falls_back(self) -> None:
        score = await self._score(
            self._ev_with_schema(resolver=lambda _example: None),
            "2026-01-02T09:00:00Z",
            "2026-01-02T09:00:00+00:00",
            field="start_time",
        )
        assert 0.0 < score < 1.0

    async def test_a_raising_resolver_falls_back_instead_of_erroring(self) -> None:
        def _boom(_example: SuiteExample) -> list[ToolSpec] | None:
            raise RuntimeError("sidecar is gone")

        score = await self._score(
            self._ev_with_schema(resolver=_boom),
            "2026-01-02T09:00:00Z",
            "2026-01-02T09:00:00+00:00",
            field="start_time",
        )
        assert 0.0 < score < 1.0

    async def test_the_resolver_is_consulted_once_per_pair(self) -> None:
        """Resolution is per example, not per field — it can touch disk."""
        calls: list[str] = []

        def _counting(example: SuiteExample) -> list[ToolSpec] | None:
            calls.append(example.id)
            return [_add_event_spec()]

        ev = self._ev_with_schema(resolver=_counting)
        await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("add_event", {"title": "a", "status": "confirmed"})),
            target_trace=_trace(("add_event", {"title": "b", "status": "tentative"})),
        )
        assert calls == ["ex1"]

    async def test_dispatch_also_applies_against_expected_ground_truth(self) -> None:
        ev = ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name="tool_arguments", against="expected"),
            toolset_resolver=_resolve_add_event,
        )
        example = suite_example(
            id="ex1",
            inputs={},
            expected_tools=[
                ExpectedToolCall(
                    tool_name="add_event",
                    arguments={"status": "confirmed"},
                    match_strategy="subset",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("add_event", {"status": "confirmed"})),
            target_trace=_trace(("add_event", {"status": "tentative"})),
        )
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(0.0)


class TestNoMatch:
    async def test_target_calls_different_tool(self) -> None:
        ev = _ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("a", {})),
            target_trace=_trace(("b", {})),
        )
        # No matched calls; target made calls so it's a regression.
        assert record.target_score == 0.0


class TestParseError:
    async def test_parse_error_args_score_zero(self) -> None:
        ev = _ev()
        # arguments dict with sentinel _parse_error → all real keys gone.
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("ping", {"_raw": "{bad", "_parse_error": True})),
            target_trace=_trace(("ping", {"_raw": "{bad", "_parse_error": True})),
        )
        # Both halves had only sentinel keys → fallback 0.0.
        assert record.target_score == 0.0


# ---------------------------------------------------------------------------
# against: expected — score both sides against recorded ground truth
# ---------------------------------------------------------------------------


def _expected_ev(
    *,
    strategies: dict[str, str] | None = None,
    default_strategy: str = "exact",
    optional_fields_scored: str = "lenient",
    embeddings_fn: Any = None,
) -> ToolArgumentsEvaluator:
    cfg = ToolArgumentsEvaluatorConfig(
        name="routing_args",
        against="expected",
        strategies=strategies or {},  # type: ignore[arg-type]
        default_strategy=default_strategy,  # type: ignore[arg-type]
        optional_fields_scored=optional_fields_scored,  # type: ignore[arg-type]
    )
    return ToolArgumentsEvaluator(cfg, embeddings_fn=embeddings_fn)


class TestScoringAgainstExpected:
    """``against: expected`` measures correctness, not drift-from-source.

    The default ``source`` path pins ``source_score`` at 1.0 by construction,
    so a source model that hallucinated an argument value defines the yardstick
    and scores perfectly against itself.
    """

    async def test_a_drifting_source_is_penalised(self) -> None:
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="archive_project",
                    arguments={"project_name": "Series A Fundraise"},
                    match_strategy="subset",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(
                ("archive_project", {"project_name": "Series A Fundraise (March 29, 2026)"}),
            ),
            target_trace=_trace(("archive_project", {"project_name": "Series A Fundraise"})),
        )
        assert record.source_score == pytest.approx(0.0)
        assert record.target_score == pytest.approx(1.0)
        assert record.metadata["against"] == "expected"

    async def test_no_ground_truth_arguments_writes_no_record(self) -> None:
        """No expectation to score against is a non-measurement, not a match.

        The 1.0/1.0 this used to write read as a perfect argument match on a
        suite that never recorded any arguments.
        """
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("a", {"k": 1})),
            target_trace=_trace(("a", {"k": 2})),
        )
        assert record is None

    async def test_a_call_the_model_never_made_scores_zero(self) -> None:
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(tool_name="archive_project", arguments={"project_name": "A"}),
                ExpectedToolCall(tool_name="archive_project", arguments={"project_name": "B"}),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(
                ("archive_project", {"project_name": "A"}),
                ("archive_project", {"project_name": "B"}),
            ),
            target_trace=_trace(("archive_project", {"project_name": "A"})),
        )
        assert record.source_score == pytest.approx(1.0)
        # One of two expected calls made, with correct arguments → 0.5.
        assert record.target_score == pytest.approx(0.5)
        assert record.metadata["expected_calls"] == 2

    async def test_an_expectation_without_arguments_is_ignored(self) -> None:
        """Name-only expectations are ``tool_selection``'s business, not ours."""
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(tool_name="get_projects"),
                ExpectedToolCall(tool_name="archive_project", arguments={"project_name": "A"}),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("archive_project", {"project_name": "A"})),
            target_trace=_trace(("archive_project", {"project_name": "A"})),
        )
        assert record.metadata["expected_calls"] == 1
        assert record.target_score == pytest.approx(1.0)

    async def test_subset_expectations_ignore_extra_arguments(self) -> None:
        """``match_strategy: subset`` — what promotion writes — scores expected keys only."""
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="get_projects",
                    arguments={"status": "active"},
                    match_strategy="subset",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("get_projects", {"status": "active"})),
            target_trace=_trace(("get_projects", {"status": "active", "limit": 10})),
        )
        assert record.target_score == pytest.approx(1.0)

    async def test_exact_expectations_penalise_extra_arguments(self) -> None:
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="get_projects",
                    arguments={"status": "active"},
                    match_strategy="exact",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("get_projects", {"status": "active"})),
            target_trace=_trace(("get_projects", {"status": "active", "limit": 10})),
        )
        # status 1.0, the undeclared `limit` 0.5 under lenient presence scoring.
        assert record.target_score == pytest.approx(0.75)

    async def test_field_strategies_still_apply(self) -> None:
        async def fake_emb(a: str, b: str) -> float:
            return 0.94

        cfg = ToolArgumentsEvaluatorConfig(
            name="routing_args",
            against="expected",
            strategies={"query": "semantic"},
        )
        ev = ToolArgumentsEvaluator(cfg, embeddings_fn=fake_emb)
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="search_web",
                    arguments={"query": "weather in Madrid tomorrow"},
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("search_web", {"query": "weather in Madrid tomorrow"})),
            target_trace=_trace(("search_web", {"query": "Madrid weather Tuesday"})),
        )
        assert record.target_score == pytest.approx(0.94)

    async def test_records_still_carry_the_evaluator_kind(self) -> None:
        # Needs real ground truth: without it the evaluator measures nothing
        # and writes no record to carry a slug at all.
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[ExpectedToolCall(tool_name="a", arguments={"k": 1})],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(("a", {"k": 1})),
            target_trace=_trace(("a", {"k": 1})),
        )
        assert record is not None
        assert record.kind == "tool_arguments"
        assert record.evaluator_name == "routing_args"


class TestAutoStrategyAgainstExpected:
    """The ladder has to reach the ground-truth path too — that is where
    ``capture sync`` puts free-text arguments."""

    async def test_capitalization_only_drift_is_not_a_regression(self) -> None:
        ev = _expected_ev(default_strategy="auto")
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="search_web",
                    arguments={"query": "find people that would use Evalshift"},
                    match_strategy="subset",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(
                ("search_web", {"query": "find people that would use Evalshift"}),
            ),
            target_trace=_trace(
                ("search_web", {"query": "Find people that would use Evalshift"}),
            ),
        )
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(1.0)
        assert record.delta == pytest.approx(0.0)

    async def test_a_reworded_argument_gets_partial_credit(self) -> None:
        ev = _expected_ev(default_strategy="auto")
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="archive_project",
                    arguments={"project_name": "Series A Fundraise"},
                    match_strategy="subset",
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(
                ("archive_project", {"project_name": "Series A Fundraise (March 29, 2026)"}),
            ),
            target_trace=_trace(("archive_project", {"project_name": "Series A Fundraise"})),
        )
        assert record is not None
        # Under ``exact`` the source scored a flat 0.0; the extra suffix is a
        # partial mismatch, not a total one.
        assert 0.0 < record.source_score < 1.0
        assert record.target_score == pytest.approx(1.0)


class TestDriftIsARegression:
    """``ARGUMENT_VALUE_DRIFT`` marks a source→target regression.

    Stamping every target below ground truth double-counted rows where both
    models were equally wrong: the live run reported the category twice on
    one migration where only one row was an actual regression.
    """

    @staticmethod
    def _example() -> SuiteExample:
        return suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(tool_name="archive_project", arguments={"project_name": "A"}),
            ],
        )

    async def test_both_sides_equally_wrong_is_not_drift(self) -> None:
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(("archive_project", {"project_name": "wrong"})),
            target_trace=_trace(("archive_project", {"project_name": "also wrong"})),
        )
        assert record is not None
        assert record.source_score == pytest.approx(0.0)
        assert record.delta == pytest.approx(0.0)
        assert record.metadata["failure_categories"] == []

    async def test_a_target_below_the_source_is_stamped(self) -> None:
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(("archive_project", {"project_name": "A"})),
            target_trace=_trace(("archive_project", {"project_name": "wrong"})),
        )
        assert record is not None
        assert record.delta < 0
        assert record.metadata["failure_categories"] == [ARGUMENT_VALUE_DRIFT]

    async def test_a_target_above_the_source_is_not_stamped(self) -> None:
        """The target is still short of ground truth, but it improved."""
        ev = _expected_ev(default_strategy="auto")
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(("archive_project", {"project_name": "totally different"})),
            target_trace=_trace(("archive_project", {"project_name": "A "})),
        )
        assert record is not None
        assert record.delta > 0
        assert record.metadata["failure_categories"] == []


class TestUnmeasurableGroundTruthFields:
    """A ground-truth field neither side produced is a stale expectation.

    Scoring it against both sides caps the call below 1.0 for good — no model
    change can lift it — which presents a suite-quality problem as a model
    signal. It leaves the denominator instead, and is disclosed in metadata.
    """

    @staticmethod
    def _example() -> SuiteExample:
        return suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="add_event",
                    arguments={
                        "title": "Standup",
                        "start_time": "2026-08-27T09:00:00Z",
                        "end_time": "2026-08-27T09:15:00Z",
                    },
                ),
            ],
        )

    @staticmethod
    def _call(**overrides: Any) -> tuple[str, dict[str, Any]]:
        args: dict[str, Any] = {"title": "Standup", "start_time": "2026-08-27T09:00:00Z"}
        args.update(overrides)
        return ("add_event", args)

    async def test_absent_from_both_sides_leaves_the_denominator(self) -> None:
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(self._call()),
            target_trace=_trace(self._call()),
        )
        assert record is not None
        # Both sides scored 0.833 while ``end_time`` sat in the denominator.
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(1.0)
        assert record.metadata["per_call"][0]["unmeasured_fields"] == ["end_time"]
        assert "end_time" not in record.metadata["per_call"][0]["field_scores"]

    async def test_absent_from_one_side_only_still_counts(self) -> None:
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(self._call(end_time="2026-08-27T09:15:00Z")),
            target_trace=_trace(self._call()),
        )
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        # ``lenient`` scores the omitted field 0.5, over all three fields.
        assert record.target_score == pytest.approx(5 / 6)
        assert "unmeasured_fields" not in record.metadata["per_call"][0]
        assert record.metadata["failure_categories"] == [ARGUMENT_VALUE_DRIFT]

    async def test_a_wholly_unmeasurable_call_scores_one(self) -> None:
        """Nothing comparable left is the existing empty-keys case: 1.0."""
        ev = _expected_ev()
        example = suite_example(
            id="x",
            expected_tools=[
                ExpectedToolCall(
                    tool_name="add_event",
                    arguments={"end_time": "2026-08-27T09:15:00Z"},
                ),
            ],
        )
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=_trace(self._call()),
            target_trace=_trace(self._call()),
        )
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(1.0)
        assert record.metadata["per_call"][0]["field_scores"] == {}
        assert record.metadata["per_call"][0]["unmeasured_fields"] == ["end_time"]
        assert record.metadata["per_call_source"][0]["unmeasured_fields"] == ["end_time"]

    async def test_a_missing_call_is_unaffected(self) -> None:
        """No matched call on one side: the other side's omissions still count."""
        ev = _expected_ev()
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=self._example(),
            source_trace=_trace(self._call(end_time="2026-08-27T09:15:00Z")),
            target_trace=_trace(("other_tool", {})),
        )
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(0.0)
        assert record.metadata["per_call"][0]["missing"] is True


class TestGroundTruthProvenanceStamp:
    """Every ``against: expected`` row records where its ground truth came from.

    The analysis layer never sees suite rows, only records, so the disclosure
    it prints about source-derived ground truth has to travel on the record --
    same channel as ``failure_categories``.
    """

    @staticmethod
    def _ev() -> ToolArgumentsEvaluator:
        return ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name="tool_arguments", against="expected"),
        )

    @staticmethod
    def _example(*provenance: str) -> SuiteExample:
        return suite_example(
            id="ex1",
            inputs={},
            expected_tools=[
                ExpectedToolCall(
                    tool_name=f"t{i}",
                    arguments={"q": "x"},
                    provenance=p,  # type: ignore[arg-type]
                )
                for i, p in enumerate(provenance)
            ],
        )

    async def _score(self, example: SuiteExample) -> Any:
        names = [f"t{i}" for i in range(len(example.expected_tools or []))]
        trace = _trace(*[(name, {"q": "x"}) for name in names])
        return await self._ev().score_pair(
            run_id="r",
            prompt_id="p",
            example=example,
            source_trace=trace,
            target_trace=trace,
        )

    async def test_all_captured(self) -> None:
        record = await self._score(self._example("captured", "captured"))
        assert record is not None
        assert record.metadata["gt_provenance"] == "captured"

    async def test_all_reviewed(self) -> None:
        record = await self._score(self._example("reviewed", "reviewed"))
        assert record is not None
        assert record.metadata["gt_provenance"] == "reviewed"

    async def test_mixed_expectations_are_neither(self) -> None:
        record = await self._score(self._example("captured", "reviewed"))
        assert record is not None
        assert record.metadata["gt_provenance"] == "mixed"

    async def test_against_source_rows_carry_no_stamp(self) -> None:
        """Drift-vs-source never claimed to measure correctness, so nothing to disclose."""
        record = await _ev().score_pair(
            run_id="r",
            prompt_id="p",
            example=_example(),
            source_trace=_trace(("t0", {"q": "x"})),
            target_trace=_trace(("t0", {"q": "x"})),
        )
        assert record is not None
        assert "gt_provenance" not in record.metadata
