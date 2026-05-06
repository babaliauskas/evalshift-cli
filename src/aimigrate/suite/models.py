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


class _StrictModel(BaseModel):
    """Forbid extra keys + validate on assignment.

    Mirrors the same-named base in :mod:`aimigrate.config.models`. We keep a
    parallel definition here rather than extracting a shared base because
    the two modules have different audiences (config vs. suite data) and
    diverging defaults are likely. If a third module ever needs the same
    pattern, that's the moment to consolidate.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] | None = None
    match_strategy: Literal["exact", "subset", "contains_per_field"] = "subset"


class SuiteExample(_StrictModel):
    """A single row from a golden suite file.

    Attributes:
        id: Stable identifier surfaced in reports and error messages.
            Must be unique within the suite.
        inputs: Mapping of template-variable name to value. The orchestrator
            substitutes these into each prompt template via
            :func:`aimigrate.utils.templating.render`.
        tags: Optional labels used by :class:`aimigrate.config.models.SliceConfig`
            to group examples for slice-level statistical analysis. An
            example may carry multiple tags.
        expected: Optional reference output for evaluators that take an
            "expected" answer (most evaluators in the MVP do not).
        expected_tools: v0.2 — ground-truth list of tool calls the model
            should make. ``None`` means no expectation (tool evaluators
            skip).
        expected_tool_count: v0.2 — exact total tool-call count expected.
            Used by the trace-structure evaluator.
        expected_no_tools: v0.2 — set ``True`` when the example expects a
            text-only response (refusal-equivalent for agent prompts).
        expected_parallel: v0.2 — when set, the trace-structure evaluator
            checks the parallel-vs-sequential flag of the model output.
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
    expected_tool_count: int | None = Field(default=None, ge=0)
    expected_no_tools: bool = False
    expected_parallel: bool | None = None

    @model_validator(mode="after")
    def _check_tool_expectations_consistent(self) -> Self:
        """Reject mutually-exclusive tool expectations.

        If ``expected_no_tools`` is ``True``, ``expected_tools`` must be
        ``None`` or an empty list (you can't expect zero AND expect a
        specific tool to be called).
        """
        if self.expected_no_tools and self.expected_tools:
            raise ValueError(
                "expected_no_tools=True is incompatible with a non-empty expected_tools list",
            )
        if self.expected_no_tools and self.expected_tool_count not in (None, 0):
            raise ValueError(
                "expected_no_tools=True is incompatible with expected_tool_count != 0",
            )
        return self


class Suite(_StrictModel):
    """An ordered, id-unique collection of :class:`SuiteExample`.

    Construct directly from a list of examples or load from a JSONL file via
    :func:`aimigrate.suite.loader.load_jsonl`.
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


__all__ = ["ExpectedToolCall", "Suite", "SuiteExample"]
