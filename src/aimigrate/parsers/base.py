"""Common types for prompt parsers.

A *prompt parser* turns a :class:`PromptDefinition` (a row from
``aimigrate.yaml``) into a :class:`PromptTemplate` — the runtime-ready
form the orchestrator hands to the templating engine and the LLM client.

We use a Protocol rather than ABC inheritance so each parser can be a
plain function-style class without ceremony, and so test doubles don't
need to subclass anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from aimigrate.config.models import PromptDefinition

PromptParseErrorKind = Literal[
    "missing_file",
    "not_a_file",
    "ast_syntax",
    "variable_not_found",
    "non_literal",
    "invalid_definition",
]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A loaded prompt ready for variable substitution and dispatch.

    Attributes:
        id: Stable identifier (mirrors :attr:`PromptDefinition.id`).
        content: The raw template string with ``{var}`` placeholders.
        declared_variables: Names the user declared in
            :attr:`PromptDefinition.variables`. The orchestrator validates
            these against suite example inputs before any LLM calls.
    """

    id: str
    content: str
    declared_variables: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PromptParseErrorDetail:
    """A single parse problem (location + message)."""

    location: str
    message: str


class PromptParseError(Exception):
    """Raised when a prompt cannot be parsed.

    Carries the same plain/rich rendering surface as :class:`ConfigError`
    and :class:`SuiteError` so CLI commands present a uniform error UX.
    """

    def __init__(
        self,
        prompt_id: str,
        kind: PromptParseErrorKind,
        summary: str,
        path: Path | None = None,
        details: list[PromptParseErrorDetail] | None = None,
    ) -> None:
        self.prompt_id = prompt_id
        self.kind: PromptParseErrorKind = kind
        self.summary = summary
        self.path = path
        self.details: list[PromptParseErrorDetail] = list(details or [])
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        head = f"Prompt parse error in prompt '{self.prompt_id}'"
        if self.path is not None:
            head += f" ({self.path})"
        lines = [f"{head}:", f"  {self.summary}"]
        if self.details:
            lines.append("")
            for d in self.details:
                lines.append(f"  • [{d.location}] {d.message}")
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        body: list[RenderableType] = [Text(self.summary, style="bold red")]
        if self.details:
            body.append(Text(""))
            for d in self.details:
                line = Text()
                line.append("• ", style="red")
                line.append(d.location, style="bold cyan")
                line.append(": ")
                line.append(d.message)
                body.append(line)
        title = f"[red]Invalid prompt[/red]: {self.prompt_id}"
        if self.path is not None:
            title += f" ({self.path})"
        return Panel(Group(*body), title=title, title_align="left", border_style="red")


@runtime_checkable
class PromptParser(Protocol):
    """Convert a :class:`PromptDefinition` into a :class:`PromptTemplate`.

    Implementations should raise :class:`PromptParseError` (never bubble
    up raw IO/syntax exceptions) so the CLI can render every failure with
    consistent formatting.
    """

    def parse(
        self,
        prompt: PromptDefinition,
        project_root: Path,
    ) -> PromptTemplate:
        """Parse a prompt definition rooted at ``project_root``.

        Args:
            prompt: The validated :class:`PromptDefinition` from config.
            project_root: Directory used to resolve relative paths.

        Returns:
            A ready-to-render :class:`PromptTemplate`.

        Raises:
            PromptParseError: If the prompt cannot be loaded for any reason.
        """
        ...


__all__ = [
    "PromptParseError",
    "PromptParseErrorDetail",
    "PromptParseErrorKind",
    "PromptParser",
    "PromptTemplate",
]
