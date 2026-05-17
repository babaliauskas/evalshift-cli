"""Pydantic models for the ``evalshift.yaml`` configuration file.

The schema is the contract between EvalShift and its users: any field added
here becomes part of the public API and must be documented in
``docs/configuration.md``. Models use ``extra='forbid'`` so unknown keys in a
user's YAML fail loudly rather than being silently ignored — typos in config
are a common source of frustrating bug reports, and we'd rather surface them
immediately.

The schema closely follows §5.1 of the EvalShift MVP build plan, with one
deliberate deviation: the top-level ``evaluators`` field is a typed
:class:`EvaluatorsConfig` rather than a loose ``dict``. The typed wrapper
gives us real validation and keeps ``mypy --strict`` happy.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Default judge model used across the config when none is specified.
# Centralised here so a single edit propagates everywhere. Gemini's
# flash-lite-preview tier is intentionally cheap; users who want a
# stronger judge should override via `evaluators.llm_judge[*].judge_model`.
DEFAULT_JUDGE_MODEL: str = "gemini-3.1-flash-lite-preview"


class _StrictModel(BaseModel):
    """Base for every config model: forbid extra keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PromptDefinition(_StrictModel):
    """A single prompt that EvalShift will evaluate.

    A prompt is sourced either inline from ``evalshift.yaml`` (``detection:
    manual``) or extracted from a Python source file by AST-walking for a
    module-level string assignment (``detection: python_string``).

    Attributes:
        id: Stable, human-readable identifier used in reports and CLI output.
        detection: How to load the prompt body — ``manual`` (use ``content``)
            or ``python_string`` (use ``path`` + ``variable``).
        content: Inline prompt body (required when ``detection='manual'``).
        path: Path to a ``.py`` file containing the prompt string (required
            when ``detection='python_string'``).
        variable: Name of the module-level variable holding the prompt string
            (required when ``detection='python_string'``).
        variables: Names of template variables (``{var}`` placeholders) the
            prompt expects to be filled in by suite examples.
    """

    id: str = Field(min_length=1, description="Stable identifier for the prompt.")
    detection: Literal["manual", "python_string"] = Field(
        description="How the prompt body is sourced.",
    )
    content: str | None = Field(default=None, description="Inline prompt body (manual only).")
    path: str | None = Field(
        default=None,
        description="Path to a .py file (python_string only).",
    )
    variable: str | None = Field(
        default=None,
        description="Module-level variable name (python_string only).",
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Template variables the prompt expects.",
    )
    # v0.2 — agent-style prompts have a ``tools_path`` pointing at a
    # yaml/json file with the tool specs. When set, the orchestrator
    # uses ``ModelClient.complete_with_tools`` and tool evaluators
    # apply.
    tools_path: str | None = Field(
        default=None,
        description="Path to a tools.yaml/json file. When set, the prompt is treated agent-style.",
    )

    @model_validator(mode="after")
    def _check_detection_fields(self) -> Self:
        """Enforce field combinations based on the chosen ``detection`` mode."""
        if self.detection == "manual":
            if not self.content:
                raise ValueError("'content' is required when detection='manual'")
            if self.path is not None or self.variable is not None:
                raise ValueError(
                    "'path' and 'variable' must not be set when detection='manual'",
                )
        else:  # python_string
            if not self.path or not self.variable:
                raise ValueError(
                    "'path' and 'variable' are required when detection='python_string'",
                )
            if self.content is not None:
                raise ValueError(
                    "'content' must not be set when detection='python_string'",
                )
        return self


class StructuralEvaluatorConfig(_StrictModel):
    """Configuration for a single structural evaluator.

    Three flavours are supported in the MVP:

    * ``json_schema`` — validate the output against a JSON Schema.
    * ``regex`` — check the output matches a regex.
    * ``length`` — score based on whether the output length is within bounds.

    Attributes:
        type: Which structural check to run.
        schema_path: Path to a JSON Schema file (``json_schema`` only).
        pattern: Regex pattern to match (``regex`` only).
        min_chars: Minimum acceptable length in characters (``length`` only).
        max_chars: Maximum acceptable length in characters (``length`` only).
        applies_to: Glob list of prompt IDs this evaluator applies to.
            Defaults to ``["*"]`` (every prompt).
    """

    type: Literal["json_schema", "regex", "length"]
    schema_path: str | None = None
    pattern: str | None = None
    min_chars: int | None = Field(default=None, ge=0)
    max_chars: int | None = Field(default=None, ge=0)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])

    @model_validator(mode="after")
    def _check_type_fields(self) -> Self:
        """Ensure the fields required by the chosen ``type`` are present."""
        if self.type == "json_schema":
            if not self.schema_path:
                raise ValueError("'schema_path' is required when type='json_schema'")
        elif self.type == "regex":
            if not self.pattern:
                raise ValueError("'pattern' is required when type='regex'")
        else:  # length
            if self.min_chars is None and self.max_chars is None:
                raise ValueError(
                    "at least one of 'min_chars' or 'max_chars' is required when type='length'",
                )
            if (
                self.min_chars is not None
                and self.max_chars is not None
                and self.min_chars > self.max_chars
            ):
                raise ValueError("'min_chars' must be <= 'max_chars'")
        return self


class SemanticEvaluatorConfig(_StrictModel):
    """Configuration for the cosine-similarity semantic evaluator.

    Attributes:
        embedding_model: LiteLLM-compatible embedding model identifier.
        applies_to: Glob list of prompt IDs this evaluator applies to.
    """

    embedding_model: str = "text-embedding-3-small"
    applies_to: list[str] = Field(default_factory=lambda: ["*"])


class LLMJudgeConfig(_StrictModel):
    """Configuration for a single pairwise LLM-as-judge evaluator.

    Attributes:
        criterion_name: Short, stable identifier surfaced in reports.
        criterion_prompt: Free-form criterion the judge will apply (e.g.
            "Which output preserves more factual detail?").
        judge_model: Model used as the judge. Defaults to a strong Anthropic
            model so users get good results out of the box.
        applies_to: Glob list of prompt IDs this evaluator applies to.
    """

    criterion_name: str = Field(min_length=1)
    criterion_prompt: str = Field(min_length=1)
    judge_model: str = DEFAULT_JUDGE_MODEL
    applies_to: list[str] = Field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# v0.2 — tool-call evaluator configs
# ---------------------------------------------------------------------------


class ToolSelectionEvaluatorConfig(_StrictModel):
    """Configuration for the tool-selection evaluator.

    Attributes:
        name: Stable identifier surfaced in reports.
        mode:
            * ``exact`` — target's tool-name sequence must equal source's.
            * ``set`` — target's tool-name set must equal source's
              (Jaccard-like).
            * ``first`` — only the first call is checked.
            * ``expected`` — match against ``example.expected_tools``
              (default; most useful for migrations with ground truth).
        applies_to: Glob list of prompt ids this evaluator applies to.
        severity_floor: When set, the analysis layer can floor severity
            at this level for any non-``none`` result on this evaluator
            (e.g. force ``high`` whenever the security agent loses a
            tool call).
    """

    name: str = Field(min_length=1)
    mode: Literal["exact", "set", "first", "expected"] = "expected"
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    severity_floor: Literal["low", "medium", "high", "critical"] | None = None


class ToolArgumentsEvaluatorConfig(_StrictModel):
    """Configuration for the tool-arguments evaluator.

    Attributes:
        name: Stable identifier surfaced in reports.
        applies_to: Glob list of prompt ids this evaluator applies to.
        strategies: Per-field strategy overrides. Field name → strategy.
            Unlisted fields default to ``exact``.
        numeric_tolerance: Relative-error tolerance for the ``numeric``
            strategy (0.05 = 5%).
        use_llm_judge_fallback: Reserved for v0.3; ignored in v0.2.
    """

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    strategies: dict[str, Literal["exact", "subset", "numeric", "semantic"]] = Field(
        default_factory=dict,
    )
    numeric_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    use_llm_judge_fallback: bool = False


class ToolTraceStructureEvaluatorConfig(_StrictModel):
    """Configuration for the trace-structure evaluator.

    Attributes:
        name: Stable identifier surfaced in reports.
        applies_to: Glob list of prompt ids this evaluator applies to.
        check_call_count: Score the number of calls.
        check_parallelism: Score whether parallel-call usage matches.
        check_refusals: Score whether refusal alignment matches; a
            refusal regression always forces ``severity_floor="high"``.
        call_count_tolerance: ``+/- N`` calls considered equivalent.
    """

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    check_call_count: bool = True
    check_parallelism: bool = True
    check_refusals: bool = True
    call_count_tolerance: int = Field(default=1, ge=0)


class EvaluatorsConfig(_StrictModel):
    """Container for all evaluator configurations attached to a run."""

    structural: list[StructuralEvaluatorConfig] = Field(default_factory=list)
    semantic: SemanticEvaluatorConfig | None = None
    llm_judge: list[LLMJudgeConfig] = Field(default_factory=list)
    # v0.2 — tool-call evaluators (all optional; v0.1 configs unaffected).
    tool_selection: list[ToolSelectionEvaluatorConfig] = Field(default_factory=list)
    tool_arguments: list[ToolArgumentsEvaluatorConfig] = Field(default_factory=list)
    tool_trace_structure: list[ToolTraceStructureEvaluatorConfig] = Field(
        default_factory=list,
    )


class SliceConfig(_StrictModel):
    """A named slice — a subset of suite examples to analyse separately.

    Attributes:
        name: Human-readable slice name surfaced in reports.
        filter: A Python expression evaluated against each example's inputs.
            Examples in the slice are those for which the expression is truthy.
            (Evaluation safety is the responsibility of the runtime; for the
            MVP we will document that ``evalshift.yaml`` is treated as
            trusted, like any project config file.)
        applies_to: Glob list of prompt IDs this slice applies to.
    """

    name: str = Field(min_length=1)
    filter: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])


class Defaults(_StrictModel):
    """Top-level defaults applied across the run.

    Attributes:
        source_model: Default ``--from`` model. Overridable on the CLI.
        target_model: Default ``--to`` model. Overridable on the CLI.
        judge_model: Default LLM-as-judge model.
        concurrency: Maximum in-flight LLM calls.
        cache: Whether to read/write the local SQLite cache.
        max_cost_usd: Hard ceiling for a single run; the orchestrator aborts
            the pre-flight if estimated cost exceeds this.
    """

    source_model: str | None = None
    target_model: str | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    concurrency: int = Field(default=10, gt=0, le=64)
    cache: bool = True
    max_cost_usd: float = Field(default=50.0, ge=0.0)


class EvalShiftConfig(_StrictModel):
    """Top-level configuration loaded from ``evalshift.yaml``."""

    version: Literal[1] = 1
    project: str | None = Field(default=None, pattern=r"^[a-z0-9-]+/[a-z0-9-]+$")
    thresholds: dict[str, Any] = Field(default_factory=dict)
    prompts: list[PromptDefinition] = Field(min_length=1)
    defaults: Defaults = Field(default_factory=Defaults)
    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)
    slices: list[SliceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_prompt_ids(self) -> Self:
        """Reject configs that declare the same prompt id twice."""
        ids = [p.id for p in self.prompts]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate prompt ids: {duplicates}")
        return self


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "Defaults",
    "EvalShiftConfig",
    "EvaluatorsConfig",
    "LLMJudgeConfig",
    "PromptDefinition",
    "SemanticEvaluatorConfig",
    "SliceConfig",
    "StructuralEvaluatorConfig",
    "ToolArgumentsEvaluatorConfig",
    "ToolSelectionEvaluatorConfig",
    "ToolTraceStructureEvaluatorConfig",
]
