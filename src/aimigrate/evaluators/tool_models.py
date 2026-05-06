"""Provider-agnostic data models for tool calls and traces.

These models normalise the shape of tool-use responses across Anthropic,
OpenAI, and Gemini so the rest of the v0.2 pipeline (parser, evaluators,
report) speaks one language regardless of provider.

The wire format mirrors Anthropic's tool spec (``name`` / ``description`` /
``input_schema``), which OpenAI's adapter accepts as input. We provide
``ToolSpec.to_anthropic()`` / ``.to_openai()`` for outbound serialisation
and ``ToolSpec.from_dict()`` for inbound deserialisation that accepts
either shape.

This module is dependency-free beyond pydantic so the evaluator package
stays cheap to import.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Forbid extras + validate on assignment.

    Mirrors the same-named bases in :mod:`aimigrate.config.models`,
    :mod:`aimigrate.runner.models`, and :mod:`aimigrate.suite.models`.
    Kept parallel rather than extracted into a shared location because
    the v0.1 → v0.2 cut deliberately avoids mid-flight refactors of
    cross-cutting types.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolSpec(_StrictModel):
    """A tool definition the model can call.

    Wire format mirrors Anthropic's tool spec, which OpenAI's adapter
    accepts as input via ``to_openai()``. Use :meth:`from_dict` to
    deserialise either shape.
    """

    name: str = Field(min_length=1, description="Tool name; must match the model's understanding.")
    description: str = Field(
        min_length=1,
        description="Human-readable description; the model uses this to decide when to call.",
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the tool's arguments.",
    )

    def to_anthropic(self) -> dict[str, Any]:
        """Serialise in Anthropic / LiteLLM-Anthropic-path shape."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict[str, Any]:
        """Serialise in OpenAI function-calling shape."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Accept either Anthropic or OpenAI shape and produce a :class:`ToolSpec`.

        Args:
            payload: A single tool dict in either provider shape.

        Returns:
            The normalised :class:`ToolSpec`.

        Raises:
            ValueError: If the payload doesn't look like either shape.
        """
        if "function" in payload:
            fn = payload["function"]
            if not isinstance(fn, dict):
                raise ValueError("OpenAI-shape tool payload missing 'function' object")
            return cls(
                name=fn["name"],
                description=fn.get("description", ""),
                input_schema=fn.get("parameters", {}),
            )
        if "name" not in payload:
            raise ValueError("tool payload missing 'name' field")
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            input_schema=payload.get("input_schema", payload.get("parameters", {})),
        )


class ToolCall(_StrictModel):
    """A single tool invocation parsed from a model response.

    Attributes:
        tool_name: The name of the tool the model invoked.
        arguments: The arguments the model passed. ``{"_parse_error": True}``
            indicates the underlying provider returned a malformed JSON
            argument string; downstream evaluators handle this.
        call_id: The provider-assigned id for this call (when available).
        parent_call_id: For chained / nested calls. ``None`` for top-level
            calls. v0.2 treats two top-level calls (``parent_call_id is None``)
            as parallel.
        sequence_index: 0-indexed position within the trace.
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    parent_call_id: str | None = None
    sequence_index: int = Field(ge=0)


class ToolTrace(_StrictModel):
    """The full sequence of tool calls in one model response.

    A trace can be:

    * **Empty** (``calls=[]``) — the model produced text only.
    * **Tool-only** (``calls=[...]``, ``final_text=None``) — the model
      issued tool calls without a final text answer.
    * **Mixed** (``calls=[...]``, ``final_text="..."``) — both happened.
    * **Refusal** (``raised_refusal=True``) — the model declined to act.

    Attributes:
        calls: Tool invocations in the order the model emitted them.
        final_text: Free-form text the model produced after / instead of
            tool calls. ``None`` if the response was tool-only.
        raised_refusal: True if the response indicates refusal.
        refusal_text: Optional text accompanying a refusal.
    """

    calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None
    raised_refusal: bool = False
    refusal_text: str | None = None

    @property
    def call_count(self) -> int:
        """Number of tool calls in this trace."""
        return len(self.calls)

    @property
    def tool_names(self) -> list[str]:
        """Tool names in the order they were called."""
        return [c.tool_name for c in self.calls]

    @property
    def tool_name_set(self) -> set[str]:
        """Distinct tool names called."""
        return {c.tool_name for c in self.calls}

    def has_parallel_calls(self) -> bool:
        """True iff the trace contains two or more top-level calls.

        v0.2 treats "top-level" as ``parent_call_id is None``. Provider
        adapters may refine this once we have richer parent/child signals
        from the underlying SDKs.
        """
        top_level = [c for c in self.calls if c.parent_call_id is None]
        return len(top_level) >= 2

    def calls_by_tool(self, tool_name: str) -> list[ToolCall]:
        """Return every call for the given ``tool_name`` (in trace order)."""
        return [c for c in self.calls if c.tool_name == tool_name]

    @model_validator(mode="after")
    def _check_sequence_indices_unique(self) -> Self:
        """Reject traces where two calls share a ``sequence_index``.

        Provider adapters always assign sequential indices; a duplicate
        indicates a parser bug or hand-crafted invalid data.
        """
        indices = [c.sequence_index for c in self.calls]
        if len(indices) != len(set(indices)):
            duplicates = sorted({i for i in indices if indices.count(i) > 1})
            raise ValueError(f"duplicate sequence_index in trace: {duplicates}")
        return self


__all__ = ["ToolCall", "ToolSpec", "ToolTrace"]
