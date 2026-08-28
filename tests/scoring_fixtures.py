"""The run that reported ``equivalent_rate: 1.0`` while nine of ten pairs diverged.

``r_20260820_project_insights_143a5f`` is a real personalButler run. Its
``raw.jsonl`` is frozen verbatim (trimmed to the four fields the scoring layer
reads) at ``tests/unit/fixtures/scoring/project_insights_pairs.jsonl`` so the
defect it exposed stays pinned without reaching outside this repo.

What the run did, and why it matters:

* every suite row carries ``expected_no_tools: true`` and no ``expected_tools``
  — that is what ``capture sync`` promotes for a turn whose recording made no
  tool call;
* every model output is ``text: ""`` — both models answered with a tool call
  and no prose, so every *text* evaluator has nothing to measure;
* nine of the ten pairs called a **different** tool on each side, and the run
  still reported behavioural equivalence.

This module lives at the ``tests/`` root rather than under ``tests/unit/``
because both the unit suite (``tests/unit/test_tool_selection.py``) and the
integration suite (``tests/integration/test_scoring_semantics.py``) read the
same fixture, and an integration test importing from ``tests.unit`` would be
backwards. The *data file* stays under ``tests/unit/fixtures/`` with every
other checked-in fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.tool_models import ToolTrace
from evalshift.suite.models import SuiteExample

#: The frozen run's identifiers, reused so a fixture-built run directory is
#: indistinguishable from the real one to every stage that reads it.
RUN_ID = "r_20260820_project_insights_143a5f"
PROMPT_ID = "replay"
SOURCE_MODEL = "gemini/gemini-3.5-flash-lite"
TARGET_MODEL = "gemini/gemini-3.7-flash"

#: Tags ``capture promote`` stamped on every row of the promoted suite.
SUITE_TAGS = ("captured", "project_insights")

#: Evaluator names from the project's real ``evalshift.yaml``. The two text
#: evaluators are named by their defaults, which is why they carry no ``kind``
#: slug in ``scores.jsonl`` — see the plan's correction C2.
TOOL_SELECTION_NAME = "routing"
TOOL_ARGUMENTS_NAME = "routing_args"
SEMANTIC_NAME = "semantic.cosine"
JUDGE_NAME = "llm_judge.equivalence"

PAIRS_PATH = (
    Path(__file__).parent / "unit" / "fixtures" / "scoring" / "project_insights_pairs.jsonl"
)

#: Ground truth, transcribed from ``EVALUATOR_SCORING_SEMANTICS_PLAN.md``:
#: short example id → (tool the source called, tool the target called).
#: ``test_fixture_matches_the_recorded_run`` proves the data file still says
#: this, so a later phase cannot quietly re-cut the fixture into agreement
#: with whatever the code does at the time.
RECORDED_TOOL_CALLS: dict[str, tuple[str, str]] = {
    "16bd3ed793": ("get_recent_files", "display_info"),
    "17693757b7": ("start_new_conversation", "get_daily_briefing"),
    "32f5d70b14": ("get_projects", "get_daily_briefing"),
    "5f66cf7a24": ("get_projects", "display_info"),
    "95831e8916": ("get_schedule", "get_daily_briefing"),
    "a46ec12303": ("get_schedule", "add_note"),
    "a52f6bf152": ("get_schedule", "add_note"),
    "a5b89feea7": ("start_new_conversation", "display_info"),
    "b5b390017b": ("get_projects", "get_projects"),
    "ebe69dc6a3": ("start_new_conversation", "add_note"),
}

#: The only pair whose two sides agreed.
CONVERGENT_EXAMPLE = "b5b390017b"

#: The nine that did not. Every one of these must produce a negative delta on
#: some record once the divergence axis exists (S2).
DIVERGENT_EXAMPLES = frozenset(RECORDED_TOOL_CALLS) - {CONVERGENT_EXAMPLE}


@dataclass(frozen=True, slots=True)
class PairFixture:
    """One example's (source, target) halves as the scoring layer sees them."""

    example_id: str
    source_text: str
    target_text: str
    source_trace: ToolTrace
    target_trace: ToolTrace

    @property
    def short_id(self) -> str:
        """The 10-char form the plan's table uses, e.g. ``16bd3ed793``."""
        return self.example_id.removeprefix("cap_")[:10]

    @property
    def example(self) -> SuiteExample:
        """The promoted suite row: no ground truth beyond "called nothing"."""
        return SuiteExample(
            id=self.example_id,
            tags=list(SUITE_TAGS),
            expected_no_tools=True,
            # This run predates per-call toolset capture (see
            # PER_CALL_TOOLSET_CAPTURE_PLAN.md) -- there is no recorded
            # toolset to carry, so tools=[] is the closest honest filler for
            # a frozen historical fixture. Unrelated to what this fixture
            # exercises: tool_selection's conformance/divergence scoring
            # never reads toolset_ref/tools.
            tools=[],
        )

    @property
    def diverged(self) -> bool:
        """Whether the two models called different tools."""
        return self.source_trace.tool_names != self.target_trace.tool_names


def load_pairs() -> tuple[PairFixture, ...]:
    """Load the ten frozen pairs, ordered by example id."""
    by_example: dict[str, dict[str, dict[str, Any]]] = {}
    for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_example.setdefault(row["example_id"], {})[row["role"]] = row

    return tuple(
        PairFixture(
            example_id=example_id,
            source_text=roles["source"]["text"],
            target_text=roles["target"]["text"],
            source_trace=ToolTrace.model_validate(roles["source"]["trace"]),
            target_trace=ToolTrace.model_validate(roles["target"]["trace"]),
        )
        for example_id, roles in sorted(by_example.items())
    )


def empty_output_pair() -> tuple[str, str]:
    """The skip case, taken from the run rather than invented.

    Both halves of every frozen pair are ``""``: the models answered with a
    tool call and no prose. A text evaluator handed these measured nothing and
    must emit nothing — today ``semantic`` reports ``1.0/1.0`` and ``llm_judge``
    reports a ``tie``.
    """
    first = load_pairs()[0]
    return first.source_text, first.target_text


def records_of(result: object) -> list[EvalRecord]:
    """Normalise a tool evaluator's ``score_pair`` return into a record list.

    ``score_pair`` returns exactly one :class:`EvalRecord` today. S1 lets it
    return ``None`` (nothing was measured) and S2 turns it into a list (the
    conformance and divergence axes are separate records). Tests assert over
    *records emitted*, which is the same question under all three shapes, so
    they keep running across the change instead of dying at the call site.
    """
    if result is None:
        return []
    if isinstance(result, EvalRecord):
        return [result]
    if isinstance(result, list | tuple):
        return [r for r in result if isinstance(r, EvalRecord)]
    raise TypeError(f"unexpected score_pair return: {type(result)!r}")
