"""Implementation of ``evalshift doctor``.

The doctor command does a fast environmental sanity check so users discover
problems (missing API keys, an invalid ``evalshift.yaml``, the wrong Python
version, …) before they kick off a paid run.

Exit codes:
    * **0** — every check passed, or any failures were merely informational
      (e.g. an unset API key, or no config in this directory yet).
    * **1** — at least one **hard** failure was reported (currently: an
      ``evalshift.yaml`` exists in the cwd but doesn't validate).

Soft failures (missing API keys, no config yet) are surfaced visually with
a yellow ``✗`` so users see them, but they never fail the command — this
keeps ``doctor`` useful as a fresh-install smoke test before users have set
up their environment.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import typer
from rich.console import Console
from rich.table import Table

from evalshift import __version__
from evalshift.config.loader import ConfigError, load_config

CheckStatus = Literal["ok", "warn", "fail"]

CONFIG_FILENAME: Final = "evalshift.yaml"
PROVIDER_KEYS: Final[tuple[str, ...]] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
)

# Glyph + Rich style for each check status. The yellow ``✗`` for warnings is
# borrowed from the PDF spec's example output: missing API keys show up as
# ``✗`` but don't fail the command.
_GLYPHS: Final[dict[CheckStatus, tuple[str, str]]] = {
    "ok": ("✓", "green"),
    "warn": ("✗", "yellow"),
    "fail": ("✗", "red"),
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A single line in the doctor report.

    Attributes:
        name: Short label (e.g. ``"ANTHROPIC_API_KEY"`` or
            ``"evalshift.yaml"``).
        status: ``"ok"`` (passes), ``"warn"`` (informational, doesn't fail
            the command), or ``"fail"`` (hard failure, exits non-zero).
        detail: Free-form one-line explanation rendered next to the status.
    """

    name: str
    status: CheckStatus
    detail: str


def run_checks(cwd: Path, env: Mapping[str, str]) -> list[CheckResult]:
    """Run every doctor check and return the list of results.

    Pure of side effects beyond reading ``cwd`` and ``env``, which makes it
    trivial to test by monkeypatching either.

    Args:
        cwd: Directory to look in for ``evalshift.yaml``.
        env: Environment-variable mapping (typically ``os.environ``).

    Returns:
        One :class:`CheckResult` per row in the doctor table, in display order.
    """
    results = [_python_check()]
    results.extend(_api_key_check(env, key) for key in PROVIDER_KEYS)
    results.append(_config_check(cwd))
    results.extend(_tool_consistency_checks(cwd))
    return results


def _python_check() -> CheckResult:
    v = sys.version_info
    return CheckResult(
        name=f"Python {v.major}.{v.minor}.{v.micro}",
        status="ok",
        detail=f"EvalShift {__version__}",
    )


def _api_key_check(env: Mapping[str, str], key: str) -> CheckResult:
    if env.get(key):
        return CheckResult(name=key, status="ok", detail="set")
    return CheckResult(
        name=key,
        status="warn",
        detail="not set (calls to this provider will fail)",
    )


def _config_check(cwd: Path) -> CheckResult:
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return CheckResult(
            name=CONFIG_FILENAME,
            status="warn",
            detail=f"not found in {cwd} (run `evalshift init` to create one)",
        )
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        return CheckResult(
            name=CONFIG_FILENAME,
            status="fail",
            detail=exc.summary,
        )
    n = len(cfg.prompts)
    return CheckResult(
        name=CONFIG_FILENAME,
        status="ok",
        detail=f"valid ({n} prompt{'s' if n != 1 else ''})",
    )


def _tool_consistency_checks(cwd: Path) -> list[CheckResult]:
    """v0.2 — warn on common agent-config inconsistencies.

    Two checks (each optional, both ``warn``-level):

    1. A prompt has ``tools_path`` set but no ``tool_*`` evaluator is
       configured (likely a config mistake — agent prompt without
       evaluators silently scores nothing).
    2. A suite example carries ``expected_tools`` but no prompt is
       configured with ``tools_path`` (the ground truth will never be
       checked).

    Both are skipped silently when the config or suite can't be loaded
    — those errors already surface elsewhere in the doctor output.
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []

    out: list[CheckResult] = []

    has_tool_evaluators = bool(
        cfg.evaluators.tool_selection
        or cfg.evaluators.tool_arguments
        or cfg.evaluators.tool_trace_structure,
    )
    agent_prompts = [p for p in cfg.prompts if p.tools_path]
    if agent_prompts and not has_tool_evaluators:
        ids = ", ".join(p.id for p in agent_prompts)
        out.append(
            CheckResult(
                name="tools without evaluators",
                status="warn",
                detail=f"prompt(s) {ids} have tools_path but no tool_* evaluators are configured",
            ),
        )

    # Suite-side check: does any example carry expected_tools while no
    # prompt is configured agent-style?
    suite_path = cwd / "golden.jsonl"
    if not agent_prompts and suite_path.exists():
        try:
            from evalshift.suite.loader import load_jsonl

            suite = load_jsonl(suite_path)
        except Exception:
            return out
        with_expected = [ex for ex in suite.examples if ex.expected_tools]
        if with_expected:
            sample = ", ".join(ex.id for ex in with_expected[:3])
            more = " ..." if len(with_expected) > 3 else ""
            out.append(
                CheckResult(
                    name="expected_tools without tools_path",
                    status="warn",
                    detail=(
                        f"{len(with_expected)} example(s) carry expected_tools "
                        f"but no prompt sets tools_path (ids: {sample}{more})"
                    ),
                ),
            )

    return out


def render_results(results: list[CheckResult], console: Console) -> None:
    """Render the check results as a Rich table."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", no_wrap=True)
    table.add_column("name", style="bold")
    table.add_column("detail", overflow="fold")
    for r in results:
        glyph, style = _GLYPHS[r.status]
        table.add_row(f"[{style}]{glyph}[/{style}]", r.name, r.detail)
    console.print(table)


def doctor() -> None:
    """Check environment and configuration; exit 1 on hard failures."""
    cwd = Path.cwd()
    results = run_checks(cwd=cwd, env=os.environ)
    render_results(results, Console())
    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)


__all__ = [
    "CONFIG_FILENAME",
    "PROVIDER_KEYS",
    "CheckResult",
    "CheckStatus",
    "doctor",
    "render_results",
    "run_checks",
]
