"""Load a list of :class:`ToolSpec` from a yaml or json file.

Tools live in a separate file (not inside ``evalshift.yaml``) because:

* They tend to be authored alongside the agent code, not the eval
  config — keeping them adjacent to the prompt source is more natural.
* The same tool list is often shared across multiple prompts.
* Provider-native shapes (Anthropic's ``input_schema`` blocks) are
  already JSON-shaped; users can paste them straight in.

Two file formats are accepted: yaml and json. Either may contain a flat
list of tool specs or a top-level ``{"tools": [...]}`` mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift.evaluators.tool_models import ToolSpec

ToolLoaderErrorKind = Literal[
    "missing", "not_a_file", "parse", "wrong_shape", "empty", "invalid_tool"
]


@dataclass(frozen=True, slots=True)
class ToolLoaderErrorDetail:
    """A single human-readable issue from :func:`load_tools`."""

    location: str
    message: str


class ToolLoaderError(Exception):
    """Raised when a tools file is missing, unreadable, or malformed.

    Carries both a plain-text and a Rich-renderable representation so
    CLI commands can surface a uniform error UX (matches
    :class:`ConfigError` / :class:`SuiteError`).
    """

    def __init__(
        self,
        path: Path,
        kind: ToolLoaderErrorKind,
        summary: str,
        details: list[ToolLoaderErrorDetail] | None = None,
    ) -> None:
        self.path = path
        self.kind: ToolLoaderErrorKind = kind
        self.summary = summary
        self.details: list[ToolLoaderErrorDetail] = list(details or [])
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        lines = [f"Tools error in {self.path}:", f"  {self.summary}"]
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
        return Panel(
            Group(*body),
            title=f"[red]Invalid tools file[/red]: {self.path}",
            title_align="left",
            border_style="red",
        )


def load_tools(path: str | Path) -> list[ToolSpec]:
    """Load a list of :class:`ToolSpec` from a yaml or json file.

    Format accepted at the top level: either a list of tool dicts
    (Anthropic-shape or OpenAI-shape — :meth:`ToolSpec.from_dict`
    handles either), or a mapping with a ``"tools"`` key whose value is
    that list.

    Args:
        path: yaml or json file path.

    Returns:
        Non-empty list of :class:`ToolSpec`.

    Raises:
        ToolLoaderError: If the file is missing, unparseable, the
            top-level shape is wrong, the list is empty, or any single
            tool fails validation.
    """
    tools_path = Path(path)
    if not tools_path.exists():
        raise ToolLoaderError(
            path=tools_path,
            kind="missing",
            summary=f"file not found: {tools_path}",
        )
    if not tools_path.is_file():
        raise ToolLoaderError(
            path=tools_path,
            kind="not_a_file",
            summary=f"expected a file, got {tools_path}",
        )

    raw_text = tools_path.read_text(encoding="utf-8")
    suffix = tools_path.suffix.lower()
    try:
        data = json.loads(raw_text) if suffix == ".json" else yaml.safe_load(raw_text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ToolLoaderError(
            path=tools_path,
            kind="parse",
            summary=f"failed to parse tools file: {exc}",
        ) from exc

    if isinstance(data, dict):
        data = data.get("tools")
    if not isinstance(data, list):
        raise ToolLoaderError(
            path=tools_path,
            kind="wrong_shape",
            summary=(
                "expected a list of tools (or a mapping with a 'tools' key); "
                f"got {type(data).__name__}"
            ),
        )

    if not data:
        raise ToolLoaderError(
            path=tools_path,
            kind="empty",
            summary="tools list is empty",
        )

    out: list[ToolSpec] = []
    details: list[ToolLoaderErrorDetail] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            details.append(
                ToolLoaderErrorDetail(
                    location=f"tools[{idx}]",
                    message=f"expected a dict, got {type(entry).__name__}",
                ),
            )
            continue
        try:
            out.append(ToolSpec.from_dict(entry))
        except (ValueError, KeyError) as exc:
            details.append(
                ToolLoaderErrorDetail(
                    location=f"tools[{idx}]",
                    message=str(exc),
                ),
            )

    if details:
        n = len(details)
        raise ToolLoaderError(
            path=tools_path,
            kind="invalid_tool",
            summary=f"{n} tool {'entry' if n == 1 else 'entries'} failed validation",
            details=details,
        )

    return out


def _format_loc_suffix(loc: tuple[Any, ...]) -> str:  # pragma: no cover — kept for symmetry
    """Render a pydantic-style ``loc`` tuple as a ``: <path>`` suffix."""
    if not loc:
        return ""
    return ": " + ".".join(str(x) for x in loc)


__all__ = [
    "ToolLoaderError",
    "ToolLoaderErrorDetail",
    "ToolLoaderErrorKind",
    "load_tools",
]
