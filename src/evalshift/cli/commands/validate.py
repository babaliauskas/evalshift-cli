"""Implementation of ``evalshift validate`` (hidden dev command).

This command is the end-to-end smoke test for everything Phase 2 builds:

1. Loads ``evalshift.yaml`` via :func:`evalshift.config.loader.load_config`.
2. Loads the suite via :func:`evalshift.suite.loader.load_jsonl`.
3. Picks the right :class:`PromptParser` per prompt definition and produces
   a list of :class:`PromptTemplate` objects.
4. Cross-checks every (template, example) pair via
   :func:`evalshift.utils.templating.validate_suite_against_prompts`.

It exits 0 with a one-line summary on success and 1 on any failure, with
a Rich-rendered error pinned to the offending file/line/field.

The command is registered with ``hidden=True`` so it doesn't appear in
the user-facing ``evalshift --help``; it's a debugging aid that we'll
remove (or move under a hidden ``--debug`` group) in Phase 8.4 cleanup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift.cli.commands.doctor import CONFIG_FILENAME
from evalshift.cli.commands.init import SUITE_FILENAME
from evalshift.config.loader import ConfigError, load_config
from evalshift.config.models import EvalShiftConfig, PromptDefinition
from evalshift.parsers.base import PromptParseError, PromptParser, PromptTemplate
from evalshift.parsers.manual import ManualParser
from evalshift.parsers.python_string import PythonStringParser
from evalshift.suite.loader import SuiteError, load_jsonl
from evalshift.suite.models import Suite
from evalshift.utils.templating import (
    SuiteCompatibilityError,
    validate_suite_against_prompts,
)


def _select_parser(prompt: PromptDefinition) -> PromptParser:
    """Return the correct parser for ``prompt.detection``.

    Centralised here so the run orchestrator (Phase 4) can reuse the same
    dispatch. Adding a new detection mode means adding one branch here
    plus the parser implementation.
    """
    if prompt.detection == "manual":
        return ManualParser()
    return PythonStringParser()


def _parse_all(
    config: EvalShiftConfig,
    project_root: Path,
) -> list[PromptTemplate]:
    """Parse every prompt in ``config``; raise on the first failure.

    We do raise on first failure here (rather than collecting like the
    config/suite loaders) because once one prompt is broken, the others
    rarely give independent useful information — and the parser errors
    already carry full context.
    """
    return [_select_parser(p).parse(p, project_root) for p in config.prompts]


def validate(
    suite: Annotated[
        Path,
        typer.Option(
            "--suite",
            "-s",
            help=f"Path to the JSONL suite (default: ./{SUITE_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(SUITE_FILENAME),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help=f"Path to evalshift.yaml (default: ./{CONFIG_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(CONFIG_FILENAME),
) -> None:
    """Verify config + suite + prompts all load and are mutually compatible."""
    console = Console()
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    try:
        loaded_suite: Suite = load_jsonl(suite)
    except SuiteError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    project_root = config.resolve().parent
    try:
        templates = _parse_all(cfg, project_root)
    except PromptParseError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    try:
        validate_suite_against_prompts(loaded_suite, templates)
    except SuiteCompatibilityError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    n_prompts = len(templates)
    n_examples = len(loaded_suite)
    console.print(
        f"[green]✓[/green] Loaded {n_prompts} prompt"
        f"{'' if n_prompts == 1 else 's'}, "
        f"{n_examples} example{'' if n_examples == 1 else 's'}; "
        "every example is compatible with every prompt."
    )


__all__ = ["validate"]
