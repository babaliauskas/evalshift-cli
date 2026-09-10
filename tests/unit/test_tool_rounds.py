"""Tests for :mod:`evalshift_cli.evaluators.tool_rounds`."""

from __future__ import annotations

from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.evaluators.tool_rounds import (
    expected_names,
    expected_rounds,
    is_multi_round,
    padded_rounds,
    replayed_rounds,
)
from evalshift_cli.suite.models import ExpectedToolCall
from tests.unit.suite_examples import suite_example


def _multi(*rounds: list[str]) -> ToolTrace:
    calls: list[ToolCall] = []
    for round_index, names in enumerate(rounds):
        for name in names:
            calls.append(
                ToolCall(
                    tool_name=name,
                    arguments={},
                    sequence_index=len(calls),
                    round_index=round_index,
                ),
            )
    return ToolTrace(calls=calls, round_count=len(rounds))


class TestIsMultiRound:
    def test_two_single_round_traces_are_not(self) -> None:
        assert not is_multi_round(_multi(["a"]), _multi(["b"]))

    def test_either_side_is_enough(self) -> None:
        assert is_multi_round(_multi(["a"], ["b"]), _multi(["b"]))
        assert is_multi_round(_multi(["a"]), _multi(["b"], ["c"]))


class TestReplayedRounds:
    def test_the_longer_replay_wins(self) -> None:
        assert replayed_rounds(_multi(["a"], ["b"], ["c"]), _multi(["a"])) == 3


class TestExpectedRounds:
    def test_the_recorded_rounds_are_used_when_present(self) -> None:
        example = suite_example(
            id="ex",
            expected_tools=[ExpectedToolCall(tool_name="a")],
            expected_tool_rounds=[
                [ExpectedToolCall(tool_name="a")],
                [ExpectedToolCall(tool_name="b")],
            ],
        )
        assert [[c.tool_name for c in r] for r in expected_rounds(example)] == [["a"], ["b"]]

    def test_a_suite_without_rounds_falls_back_to_expected_tools(self) -> None:
        example = suite_example(id="ex", expected_tools=[ExpectedToolCall(tool_name="a")])
        assert [[c.tool_name for c in r] for r in expected_rounds(example)] == [["a"]]

    def test_no_ground_truth_at_all_is_empty(self) -> None:
        assert expected_rounds(suite_example(id="ex")) == []


class TestExpectedNames:
    def test_past_the_recorded_loop_expects_nothing(self) -> None:
        rounds = [[ExpectedToolCall(tool_name="a")]]
        assert expected_names(rounds, 0) == ["a"]
        assert expected_names(rounds, 1) == []


class TestPaddedRounds:
    def test_a_shorter_replay_is_padded_with_empty_rounds(self) -> None:
        padded = padded_rounds(_multi(["a"]), 3)
        assert [trace.tool_names for trace in padded] == [["a"], [], []]

    def test_each_round_is_renumbered_from_zero(self) -> None:
        padded = padded_rounds(_multi(["a"], ["b", "c"]), 2)
        assert [call.sequence_index for call in padded[1].calls] == [0, 1]
