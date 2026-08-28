"""Template-variable extraction, rendering, and bulk compatibility checks.

A prompt template uses Python's ``str.format`` syntax: ``"Hello {name}"``
gets rendered against an example's ``inputs`` dict to produce the final
prompt sent to the LLM. This module is the only place that performs that
substitution, and it does so in a strict mode where:

* every ``{var}`` placeholder must be satisfied by ``inputs``;
* extra keys in ``inputs`` are tolerated (a suite can carry data beyond
  what any single prompt needs);
* escaped braces (``{{`` / ``}}``) are preserved as literal braces;
* attribute access in placeholders (``{foo.bar}``) reports a need for
  ``foo`` only — we don't try to navigate object trees at parse time.

The bulk pre-flight check (:func:`validate_suite_against_prompts`) walks
every (template, example) pair and collects all missing-variable
problems before raising, so users see one consolidated error message
instead of fixing-rerun-fixing in a loop.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift.parsers.base import PromptTemplate
from evalshift.suite.models import Suite


class MissingTemplateVariableError(KeyError):
    """Raised by :func:`render` when ``inputs`` is missing one or more vars.

    Subclasses :class:`KeyError` so callers that only know about the
    standard exception still get sensible behaviour, but exposes a
    structured ``missing`` set for downstream error reporting.
    """

    def __init__(self, missing: set[str]) -> None:
        self.missing = set(missing)
        super().__init__(
            f"missing template variable{'s' if len(missing) != 1 else ''}: "
            + ", ".join(sorted(missing))
        )


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """A single (prompt, example) pair that fails the compatibility check."""

    prompt_id: str
    example_id: str
    missing: frozenset[str]


class SuiteCompatibilityError(Exception):
    """Raised when one or more (prompt, example) pairs are incompatible."""

    def __init__(self, issues: list[CompatibilityIssue]) -> None:
        self.issues = list(issues)
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        n = len(self.issues)
        head = f"{n} suite/prompt compatibility {'problem' if n == 1 else 'problems'}:"
        lines = [head]
        for issue in self.issues:
            joined = ", ".join(sorted(issue.missing))
            lines.append(
                f"  • prompt '{issue.prompt_id}' x example '{issue.example_id}': "
                f"missing {{{joined}}}"
            )
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        body: list[RenderableType] = []
        for issue in self.issues:
            joined = ", ".join(sorted(issue.missing))
            line = Text()
            line.append("• ", style="red")
            line.append(f"prompt '{issue.prompt_id}'", style="bold cyan")
            line.append(" x ")
            line.append(f"example '{issue.example_id}'", style="bold cyan")
            line.append(": missing ")
            line.append(f"{{{joined}}}", style="bold yellow")
            body.append(line)
        n = len(self.issues)
        return Panel(
            Group(*body),
            title=f"[red]Suite incompatible with prompts[/red]: {n} issue(s)",
            title_align="left",
            border_style="red",
        )


def extract_variables(template: str) -> set[str]:
    """Return the set of top-level variable names in ``template``.

    Uses :class:`string.Formatter` so we get exactly the set of names
    Python's ``str.format`` would consume. Attribute and index access in a
    placeholder are stripped to the *root* name (``{user.name}`` →
    ``user``); escaped braces (``{{`` / ``}}``) are correctly skipped.

    Empty replacement fields (``{}``, ``{0}`` for positional substitution)
    are ignored — every EvalShift placeholder is named.
    """
    names: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(template):
        if field_name is None or field_name == "":
            continue
        # Auto-numbered fields like '0', '1' come through as digit strings;
        # we drop them since EvalShift templates always use named fields.
        if field_name.isdigit():
            continue
        # Strip attribute / index access: 'user.name' or 'items[0]' → 'user' / 'items'.
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root:
            names.add(root)
    return names


class _StrictFormatMapping(Mapping[str, Any]):
    """Mapping wrapper that records every missing key instead of raising once."""

    def __init__(self, inputs: Mapping[str, Any]) -> None:
        self._inputs = inputs
        self.missing: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        if key in self._inputs:
            return self._inputs[key]
        self.missing.add(key)
        # Return an empty string so format_map continues; we'll raise after.
        return ""

    def __iter__(self) -> Any:
        return iter(self._inputs)

    def __len__(self) -> int:
        return len(self._inputs)


def render(template: str, inputs: Mapping[str, Any]) -> str:
    """Render ``template`` using ``inputs`` for variable substitution.

    Args:
        template: The template string containing ``{var}`` placeholders.
        inputs: A mapping of variable name to value. Extra keys are
            ignored.

    Returns:
        The rendered string with every placeholder filled in.

    Raises:
        MissingTemplateVariableError: If any required placeholder isn't in
            ``inputs``. The exception's ``.missing`` attribute gives the
            full set of missing names so callers can report all of them
            at once instead of one per attempt.
    """
    proxy = _StrictFormatMapping(inputs)
    rendered = template.format_map(proxy)
    if proxy.missing:
        raise MissingTemplateVariableError(proxy.missing)
    return rendered


def validate_suite_against_prompts(
    suite: Suite,
    templates: list[PromptTemplate],
) -> None:
    """Verify every example provides every variable every template needs.

    For each ``(template, example)`` pair, compute the set of declared
    template variables that are absent from ``example.inputs``. If any
    pair is incomplete, collect every issue across the whole suite and
    raise a single :class:`SuiteCompatibilityError`.

    The orchestrator (Phase 4) calls this before any LLM calls so users
    don't burn money on a misaligned suite.
    """
    issues: list[CompatibilityIssue] = []
    for tmpl in templates:
        # Use the union of declared (config) and detected (template body)
        # so we catch both "you forgot to declare a {placeholder}" and
        # "you declared a variable that the example doesn't provide".
        required = set(tmpl.declared_variables) | extract_variables(tmpl.content)
        for example in suite.examples:
            missing = required - set(example.inputs.keys())
            if missing:
                issues.append(
                    CompatibilityIssue(
                        prompt_id=tmpl.id,
                        example_id=example.id,
                        missing=frozenset(missing),
                    ),
                )
    if issues:
        raise SuiteCompatibilityError(issues)


__all__ = [
    "CompatibilityIssue",
    "MissingTemplateVariableError",
    "SuiteCompatibilityError",
    "extract_variables",
    "render",
    "validate_suite_against_prompts",
]
