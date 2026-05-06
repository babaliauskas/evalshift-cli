"""Pydantic models for the ``aimigrate.yaml`` configuration file.

The schema is the contract between AIMigrate and its users: any field added
here becomes part of the public API and must be documented in
``docs/configuration.md``. Models use ``extra='forbid'`` so unknown keys in a
user's YAML fail loudly rather than being silently ignored — typos in config
are a common source of frustrating bug reports, and we'd rather surface them
immediately.

The schema closely follows §5.1 of the AIMigrate MVP build plan, with one
deliberate deviation: the top-level ``evaluators`` field is a typed
:class:`EvaluatorsConfig` rather than a loose ``dict``. The typed wrapper
gives us real validation and keeps ``mypy --strict`` happy.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Default judge model used across the config when none is specified.
# Centralised here so a single edit propagates everywhere.
DEFAULT_JUDGE_MODEL: str = "claude-5-sonnet-20260101"


class _StrictModel(BaseModel):
    """Base for every config model: forbid extra keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PromptDefinition(_StrictModel):
    """A single prompt that AIMigrate will evaluate.

    A prompt is sourced either inline from ``aimigrate.yaml`` (``detection:
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


class EvaluatorsConfig(_StrictModel):
    """Container for all evaluator configurations attached to a run."""

    structural: list[StructuralEvaluatorConfig] = Field(default_factory=list)
    semantic: SemanticEvaluatorConfig | None = None
    llm_judge: list[LLMJudgeConfig] = Field(default_factory=list)


class SliceConfig(_StrictModel):
    """A named slice — a subset of suite examples to analyse separately.

    Attributes:
        name: Human-readable slice name surfaced in reports.
        filter: A Python expression evaluated against each example's inputs.
            Examples in the slice are those for which the expression is truthy.
            (Evaluation safety is the responsibility of the runtime; for the
            MVP we will document that ``aimigrate.yaml`` is treated as
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


class AIMigrateConfig(_StrictModel):
    """Top-level configuration loaded from ``aimigrate.yaml``."""

    version: Literal[1] = 1
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
    "AIMigrateConfig",
    "Defaults",
    "EvaluatorsConfig",
    "LLMJudgeConfig",
    "PromptDefinition",
    "SemanticEvaluatorConfig",
    "SliceConfig",
    "StructuralEvaluatorConfig",
]
