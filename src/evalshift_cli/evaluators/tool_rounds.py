"""Shared round arithmetic for scoring a teacher-forced multi-round replay.

Three evaluators read tool traces (``tool_selection``, ``tool_arguments``,
``tool_trace_structure``) and each of them has to answer the same four
questions before it can score a pair: is this a multi-round replay at all,
how many rounds does the pair span, what did the recording expect in round
*k*, and what did one side do in round *k* when it replayed fewer rounds than
the other. The answers live here so the three agree by construction.

A single-round pair (``round_count == 1`` on both sides, which is every trace
written before teacher-forced replay existed) takes none of these paths: the
evaluators keep their original code for it, so its records stay byte-identical.
"""

from __future__ import annotations

from evalshift_cli.evaluators.tool_models import ToolTrace
from evalshift_cli.suite.models import ExpectedToolCall, SuiteExample

__all__ = [
    "expected_names",
    "expected_rounds",
    "is_multi_round",
    "padded_rounds",
    "replayed_rounds",
]


def is_multi_round(source_trace: ToolTrace, target_trace: ToolTrace) -> bool:
    """True when either side of the pair replayed more than one round.

    The example's ``expected_tool_rounds`` deliberately does not enter this:
    a suite can carry a whole recorded loop while the run replayed only its
    first round (``--rounds first``, the default), and grading that run per
    round would score it against rounds it was never asked for.
    """
    return source_trace.round_count > 1 or target_trace.round_count > 1


def replayed_rounds(source_trace: ToolTrace, target_trace: ToolTrace) -> int:
    """How many rounds the pair spans — the longer of the two replays."""
    return max(source_trace.round_count, target_trace.round_count)


def expected_rounds(example: SuiteExample) -> list[list[ExpectedToolCall]]:
    """The example's tool-call ground truth, grouped by round.

    Args:
        example: The suite row being scored.

    Returns:
        One list of expectations per recorded round. Falls back to
        ``expected_tools`` as a single round for a suite written before
        ``expected_tool_rounds`` existed, and is empty when the row carries no
        tool-call ground truth at all.
    """
    if example.expected_tool_rounds:
        return [list(round_calls) for round_calls in example.expected_tool_rounds]
    if example.expected_tools:
        return [list(example.expected_tools)]
    return []


def expected_names(rounds: list[list[ExpectedToolCall]], index: int) -> list[str]:
    """The tool names ``rounds`` expects in round ``index``.

    Empty past the last recorded round — that is the answer round, where the
    recording produced its final text and called nothing.
    """
    if index >= len(rounds):
        return []
    return [call.tool_name for call in rounds[index]]


def padded_rounds(trace: ToolTrace, count: int) -> list[ToolTrace]:
    """``trace`` split into single-round traces, padded out to ``count``.

    A side that replayed fewer rounds than the other contributes an empty
    round rather than dropping out of the comparison: "the target stopped
    calling tools two rounds early" is a finding, not a missing measurement.
    """
    split = trace.rounds()
    return [split[index] if index < len(split) else ToolTrace() for index in range(count)]
