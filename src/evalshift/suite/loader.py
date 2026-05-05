"""Load and validate JSONL suite files.

This module mirrors :mod:`evalshift.config.loader` in shape so the error UX
across the project stays consistent: every failure raises a structured
exception with a ``kind``, a ``summary``, and a list of per-issue
``location``/``message`` pairs, with both plain-text and Rich renderers.

Suite files are JSON Lines: one JSON object per non-blank line. The loader
collects every error before raising, so users can fix several broken rows
in one pass instead of looping through one error at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift.suite.models import Suite, SuiteExample

SuiteErrorKind = Literal[
    "missing",
    "not_a_file",
    "empty",
    "json_parse",
    "schema",
    "duplicate_ids",
]


@dataclass(frozen=True, slots=True)
class SuiteErrorDetail:
    """A single human-readable problem in a suite file.

    Attributes:
        location: ``"line N"`` for parse errors, ``"line N (id=...)"`` or
            ``"line N: <field>"`` for schema errors.
        message: One-line explanation of what's wrong.
    """

    location: str
    message: str


class SuiteError(Exception):
    """Raised when a suite file is missing, unparseable, or invalid."""

    def __init__(
        self,
        path: Path,
        kind: SuiteErrorKind,
        summary: str,
        details: list[SuiteErrorDetail] | None = None,
    ) -> None:
        self.path = path
        self.kind: SuiteErrorKind = kind
        self.summary = summary
        self.details: list[SuiteErrorDetail] = list(details or [])
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        """Render this error as a multi-line plain-text string."""
        lines = [f"Suite error in {self.path}:", f"  {self.summary}"]
        if self.details:
            lines.append("")
            for d in self.details:
                lines.append(f"  • [{d.location}] {d.message}")
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel`."""
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
            title=f"[red]Invalid suite[/red]: {self.path}",
            title_align="left",
            border_style="red",
        )


def load_jsonl(path: str | Path) -> Suite:
    """Load and validate a JSONL suite file.

    Args:
        path: Path to a JSONL file (``str`` or :class:`Path`).

    Returns:
        A fully validated :class:`Suite` containing one
        :class:`SuiteExample` per non-blank input line.

    Raises:
        SuiteError: If the file is missing, contains invalid JSON, has rows
            that fail :class:`SuiteExample` validation, or contains
            duplicate example ids. All discoverable problems are collected
            before raising, not just the first.
    """
    suite_path = Path(path)

    if not suite_path.exists():
        raise SuiteError(
            path=suite_path,
            kind="missing",
            summary=f"file not found: {suite_path}",
        )
    if not suite_path.is_file():
        raise SuiteError(
            path=suite_path,
            kind="not_a_file",
            summary=f"expected a file, got {suite_path} (is it a directory?)",
        )

    raw_text = suite_path.read_text(encoding="utf-8")

    parse_details: list[SuiteErrorDetail] = []
    schema_details: list[SuiteErrorDetail] = []
    examples: list[SuiteExample] = []

    for line_idx, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_details.append(
                SuiteErrorDetail(
                    location=f"line {line_idx}",
                    message=f"invalid JSON: {exc.msg} (col {exc.colno})",
                ),
            )
            continue

        try:
            example = SuiteExample.model_validate(payload)
        except ValidationError as exc:
            row_id = payload.get("id") if isinstance(payload, dict) else None
            location = f"line {line_idx}" + (f" (id={row_id!r})" if row_id else "")
            for err in exc.errors():
                schema_details.append(
                    SuiteErrorDetail(
                        location=location + _format_loc_suffix(err["loc"]),
                        message=str(err.get("msg", "invalid value")),
                    ),
                )
            continue

        examples.append(example)

    if parse_details:
        n = len(parse_details)
        raise SuiteError(
            path=suite_path,
            kind="json_parse",
            summary=f"{n} JSON parse {'problem' if n == 1 else 'problems'} found",
            details=parse_details,
        )

    if schema_details:
        n = len(schema_details)
        raise SuiteError(
            path=suite_path,
            kind="schema",
            summary=f"{n} schema {'problem' if n == 1 else 'problems'} found",
            details=schema_details,
        )

    if not examples:
        raise SuiteError(
            path=suite_path,
            kind="empty",
            summary="suite contains no examples (every line was blank or missing)",
        )

    try:
        return Suite(examples=examples)
    except ValidationError as exc:
        # The only post-row validation we have today is duplicate-id
        # detection on the Suite model; surface it with a dedicated kind.
        first_msg = exc.errors()[0].get("msg", "validation failed")
        raise SuiteError(
            path=suite_path,
            kind="duplicate_ids",
            summary=str(first_msg),
        ) from exc


def _format_loc_suffix(loc: tuple[int | str, ...]) -> str:
    """Render a pydantic ``loc`` tuple as a ``: <path>`` suffix."""
    if not loc:
        return ""
    parts: list[str] = []
    for piece in loc:
        if isinstance(piece, int):
            parts.append(f"[{piece}]")
        elif parts:
            parts.append(f".{piece}")
        else:
            parts.append(str(piece))
    return ": " + "".join(parts)


__all__ = ["SuiteError", "SuiteErrorDetail", "SuiteErrorKind", "load_jsonl"]
