"""Pydantic models for the golden suite.

A "suite" is a JSONL file where each non-blank line is an example: a
``{prompt-template-variable -> value}`` mapping plus optional metadata
(``id``, ``tags``, ``expected``). The orchestrator (Phase 4) iterates over
the suite, renders each prompt template against each example, and dispatches
the rendered prompts to the source/target models.

The schema is deliberately minimal — engineers tend to bring their own
labels and downstream tooling, and forcing more structure at this layer
would just block adoption. The only invariants we enforce:

* ``id`` is a non-empty string and unique across the suite.
* ``inputs`` is a flat ``dict[str, Any]`` (we don't validate value types
  because template variables can take any JSON-serialisable shape).
* Unknown top-level keys are rejected (``extra='forbid'``) so typos in
  example rows fail fast instead of being silently dropped.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalshift.evaluators.tool_models import ToolSpec
from evalshift.suite.tags import RESERVED_SLICE_NAME


class _StrictModel(BaseModel):
    """Forbid extra keys + validate on assignment.

    Mirrors the same-named base in :mod:`evalshift.config.models`. We keep a
    parallel definition here rather than extracting a shared base because
    the two modules have different audiences (config vs. suite data) and
    diverging defaults are likely. If a third module ever needs the same
    pattern, that's the moment to consolidate.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HistoryToolCall(_StrictModel):
    """A tool call recorded on an ``assistant`` turn of a conversation prefix.

    Attributes:
        id: Provider call id, used to pair this call with the ``tool``
            message carrying its result. ``None`` when the recording had
            no id; the replay then synthesises one from position.
        name: Tool name the assistant invoked.
        arguments: Arguments the assistant passed, as recorded.
    """

    id: str | None = None
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(_StrictModel):
    """A single turn in a recorded conversation prefix.

    Attributes:
        role: Who spoke the turn. ``system`` may only appear as the first
            message of a :attr:`SuiteExample.history` list (enforced there).
            ``tool`` carries the result of a previous ``assistant`` tool call.
        content: The verbatim turn text. Defaults to the empty string. For a
            ``tool`` message this is the serialised tool result.
        tool_calls: v0.3 — tool calls this ``assistant`` turn emitted. Only
            valid on ``assistant`` messages.
        tool_call_id: v0.3 — the :attr:`HistoryToolCall.id` this ``tool``
            message answers. Required on ``tool`` messages, forbidden
            elsewhere.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[HistoryToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_tool_fields_match_role(self) -> Self:
        """Reject tool metadata on a role that cannot carry it.

        A ``tool`` message with no ``tool_call_id`` cannot be paired with the
        call it answers, and providers reject the resulting message list — so
        an unpairable turn must fail at load time, not at dispatch.
        """
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls may only appear on an assistant message")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message requires tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id may only appear on a tool message")
        return self


class ExpectedToolCall(_StrictModel):
    """Ground-truth expectation for a single tool call (v0.2).

    Attributes:
        tool_name: Name of the tool the model is expected to call.
        arguments: Optional ground-truth args. ``None`` means only the
            tool name is checked; argument matching is skipped.
        match_strategy:
            * ``exact`` — every argument key + value matches exactly.
            * ``subset`` — expected args ⊆ actual args (recursive).
            * ``contains_per_field`` — every expected field is present;
              values compared per-field by the configured strategy.
        provenance: Where this expectation came from. ``captured`` (the
            default, and what ``capture promote``/``sync`` write) means the
            arguments were transcribed verbatim from the source model's own
            recorded call — nobody has checked that they are *right*. Scoring
            such a row ``against: expected`` therefore pins the source at 1.0
            by construction and measures target deviation from source, which
            the run discloses rather than silently letting ``source_score:
            1.0`` read as evidence. Set ``reviewed`` once a human has
            confirmed the row's arguments; scoring is identical either way.
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] | None = None
    match_strategy: Literal["exact", "subset", "contains_per_field"] = "subset"
    provenance: Literal["captured", "reviewed"] = "captured"


class SuiteExample(_StrictModel):
    """A single row from a golden suite file.

    Attributes:
        id: Stable identifier surfaced in reports and error messages.
            Must be unique within the suite.
        inputs: Mapping of template-variable name to value. The orchestrator
            substitutes these into each prompt template via
            :func:`evalshift.utils.templating.render`.
        tags: Optional labels used by :class:`evalshift.config.models.SliceConfig`
            to group examples for slice-level statistical analysis. An
            example may carry multiple tags.
        expected: Optional reference output for evaluators that take an
            "expected" answer (most evaluators in the MVP do not).
        expected_tools: v0.2 — ground-truth list of tool calls the model
            should make. ``None`` means no expectation (tool evaluators
            skip).
        expected_tool_rounds: v0.3 — the full recorded agent loop, grouped
            into rounds (one list per model turn that emitted tool calls).
            ``None`` for suites promoted before v0.3 or for captures that
            called no tools.
        expected_tool_count: v0.2 — exact total tool-call count expected.
            Used by the trace-structure evaluator.
        expected_no_tools: v0.2 — set ``True`` when the example expects a
            text-only response (refusal-equivalent for agent prompts).
        expected_parallel: v0.2 — when set, the trace-structure evaluator
            checks the parallel-vs-sequential flag of the model output.
        history: Multi-turn — verbatim conversation prefix prepended before
            the rendered current turn at dispatch (teacher-forced replay); a
            ``system`` message, if any, must come first. ``None`` means the
            example is single-turn; an empty list means no prefix but keeps
            the example marked as conversational.
        conversation_id: Multi-turn — id of the recorded conversation this
            example's turn was captured from. ``None`` for suites without
            conversation provenance.
        turn_index: Multi-turn — zero-based position of this turn within its
            source conversation.
        generation_config: Recorded generation config from the source
            capture's first model call (``temperature``,
            ``response_mime_type``, ``response_schema``, ...). Stored
            verbatim by ``capture promote``/``sync``; the runner translates
            it to provider kwargs at dispatch. ``None`` means no override.
        toolset_ref: Content-addressed pointer (``sha256:<hex>``) to a
            toolset sidecar under ``<base>/toolsets/`` -- what ``capture
            promote``/``sync`` writes, carried verbatim from the source
            capture's first model call. Exactly one of ``toolset_ref`` /
            ``tools`` must be set; see :meth:`_check_exactly_one_toolset_field`.
        tools: The example's toolset, inlined -- what a person writes by hand
            in a hand-authored suite. ``[]`` is a real, valid value: "this
            example's agent was offered no tools" is a first-class assertion,
            not an absence. Exactly one of ``toolset_ref`` / ``tools`` must be
            set.
    """

    id: str = Field(min_length=1, description="Stable, unique identifier.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Template-variable values injected into the prompt.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Slice labels for group-level analysis.",
    )
    expected: dict[str, Any] | None = Field(
        default=None,
        description="Optional reference output (unused by most evaluators).",
    )
    # v0.2 — tool-call ground truth (all optional; v0.1 suites still load).
    expected_tools: list[ExpectedToolCall] | None = None
    #: v0.3 — the recorded agent loop, one entry per tool-emitting round.
    #: ``expected_tools`` is normally ``expected_tool_rounds[0]``: a
    #: single-shot replay can only ever produce the first round, so that is
    #: the only fair yardstick. Later rounds are retained here for
    #: teacher-forced multi-round replay and for report context.
    expected_tool_rounds: list[list[ExpectedToolCall]] | None = None
    expected_tool_count: int | None = Field(default=None, ge=0)
    expected_no_tools: bool = False
    expected_parallel: bool | None = None
    # Multi-turn — conversation prefix + provenance (all optional; single-turn
    # suites still load unchanged).
    history: list[ChatMessage] | None = None
    conversation_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    generation_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Recorded generation config from the source capture's first model call "
            "(temperature, response_mime_type, response_schema, ...). The runner "
            "translates it to provider kwargs at dispatch; None means no override."
        ),
    )
    # Exactly one of these two must be set -- see
    # _check_exactly_one_toolset_field. Two spellings for two genuinely
    # different authors: `capture sync` (a deduplicating machine) writes
    # toolset_ref; a person hand-editing JSONL writes tools inline. Not a
    # compatibility path -- nothing reads an older, toolset-less shape.
    toolset_ref: str | None = Field(
        default=None,
        description="Content-addressed pointer to a toolset sidecar (sha256:<hex>).",
    )
    tools: list[ToolSpec] | None = Field(
        default=None,
        description="This example's toolset, inlined. [] is a valid 'no tools offered' value.",
    )

    @model_validator(mode="after")
    def _reject_reserved_slice_tag(self) -> Self:
        """Refuse a tag that would become the reserved ``overall`` slice.

        Tags are slice names: ``analysis.slicing._slices_of`` maps an untranslated
        tag straight through. The bundle contract reserves ``overall`` for the
        run-level scope, so the suite is where the collision has to be caught --
        the alternative is a full run followed by a rejection at finalize.
        """
        if RESERVED_SLICE_NAME in self.tags:
            raise ValueError(
                f"tag {RESERVED_SLICE_NAME!r} is reserved: it names the run-level scope "
                "in the run bundle, so it cannot also name a slice -- rename the tag",
            )
        return self

    @model_validator(mode="after")
    def _check_exactly_one_toolset_field(self) -> Self:
        """Require exactly one of ``toolset_ref`` / ``tools``.

        Every model call records the toolset it was offered (the empty
        toolset included, as a real fingerprinted value); a suite example is
        the promoted or hand-authored record of one such call, so it must
        carry that toolset too. Neither present is a load error, not a
        default -- a suite that forgot its toolset must fail immediately and
        visibly, the same way an empty ``id`` does, rather than silently
        behaving as though no tools existed. Both present is rejected too:
        the two spellings are mutually exclusive, not a fallback pair.
        """
        has_ref = self.toolset_ref is not None
        has_tools = self.tools is not None
        if has_ref and has_tools:
            raise ValueError(
                "toolset_ref and tools are mutually exclusive -- a promoted example "
                "references a sidecar (toolset_ref) OR a hand-authored one inlines "
                "tools directly, never both",
            )
        if not has_ref and not has_tools:
            raise ValueError(
                "exactly one of toolset_ref or tools is required -- every model call "
                "records the toolset it was offered, so a suite example must carry it "
                "too (pass tools: [] to assert 'no tools were offered')",
            )
        return self

    @model_validator(mode="after")
    def _check_tool_expectations_consistent(self) -> Self:
        """Reject tool-call ground truth paired with an assertion of "no tools".

        Three spellings assert "no tools were offered or expected", and all
        three are incompatible with any tool-call ground truth (a non-empty
        ``expected_tools``/``expected_tool_rounds``, or ``expected_tool_count
        != 0``) for the same reason: a call dispatched with no tools can
        produce no tool calls, so ground truth expecting one could never be
        satisfied -- this is the hand-authored mirror of the bug that
        motivated per-call toolset capture (a suite asserting or dispatching
        a toolset it did not have).

        * ``expected_no_tools=True`` -- "tools were offered and the model
          correctly called none".
        * ``tools == []`` -- the example's own inline toolset *is* empty (see
          :meth:`_check_exactly_one_toolset_field`).
        * ``toolset_ref == EMPTY_TOOLSET_FINGERPRINT`` -- the ``toolset_ref``
          spelling of the identical assertion. The empty toolset has exactly
          one possible fingerprint (a property of the hashing algorithm, not
          of any one sidecar), so this needs no disk I/O to check -- unlike a
          ``toolset_ref`` naming some other toolset, which still escapes this
          load-time validator because resolving what it actually contains
          needs the sidecar. That resolution-time check is
          :func:`evalshift.captures.reader.load_toolset`'s job, not this
          one's.
        """
        # Deferred import: evalshift.captures (the package __init__) imports
        # captures.models, which imports evalshift.suite.models for
        # SuiteExample -- so a module-level import here would make the first
        # `import evalshift.suite.models` (or anything importing it before
        # evalshift.captures has already been loaded, e.g. the `evalshift`
        # CLI entry point itself) try to re-enter this module while it is
        # still mid-initialization and fail with ImportError. Deferring the
        # import to call time (well after both modules have finished
        # loading, however either was reached first) breaks the cycle
        # without restructuring either package. Confirmed empirically: a
        # top-level import here breaks `import evalshift.cli.main` cold.
        from evalshift.captures.toolset import EMPTY_TOOLSET_FINGERPRINT

        reasons: list[str] = []
        if self.expected_no_tools:
            reasons.append("expected_no_tools=True")
        if self.tools == []:
            reasons.append("tools=[] (no tools offered)")
        if self.toolset_ref == EMPTY_TOOLSET_FINGERPRINT:
            reasons.append("toolset_ref=<empty toolset> (no tools offered)")

        for reason in reasons:
            if self.expected_tools:
                raise ValueError(
                    f"{reason} is incompatible with a non-empty expected_tools list",
                )
            if self.expected_tool_count not in (None, 0):
                raise ValueError(f"{reason} is incompatible with expected_tool_count != 0")
            if self.expected_tool_rounds:
                raise ValueError(
                    f"{reason} is incompatible with a non-empty expected_tool_rounds",
                )
        return self

    @model_validator(mode="after")
    def _check_history_system_message_placement(self) -> Self:
        """Reject malformed conversation prefixes.

        At most one ``system`` message is allowed in ``history``, and if
        present it must be the first element. Malformed prefixes must fail
        loudly at load time rather than silently producing garbage stats
        downstream.
        """
        if self.history is None:
            return self
        system_indices = [i for i, msg in enumerate(self.history) if msg.role == "system"]
        if len(system_indices) > 1:
            raise ValueError("history may contain at most one system message")
        if system_indices and system_indices[0] != 0:
            raise ValueError("history system message must be the first element")
        return self


class Suite(_StrictModel):
    """An ordered, id-unique collection of :class:`SuiteExample`.

    Construct directly from a list of examples or load from a JSONL file via
    :func:`evalshift.suite.loader.load_jsonl`.
    """

    examples: list[SuiteExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        """Reject suites containing the same example id more than once."""
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate example ids: {duplicates}")
        return self

    def by_tag(self, tag: str) -> list[SuiteExample]:
        """Return every example whose ``tags`` list contains ``tag``.

        Returns an empty list if no example matches; callers can rely on
        the result being a stable copy in suite order.
        """
        return [e for e in self.examples if tag in e.tags]

    def ids(self) -> set[str]:
        """Return the set of example ids in this suite."""
        return {e.id for e in self.examples}

    def __len__(self) -> int:
        return len(self.examples)


__all__ = [
    "ChatMessage",
    "ExpectedToolCall",
    "HistoryToolCall",
    "Suite",
    "SuiteExample",
]
