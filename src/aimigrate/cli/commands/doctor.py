"""Implementation of ``aimigrate doctor``.

The doctor command does a fast environmental sanity check so users discover
problems (missing API keys, an invalid ``aimigrate.yaml``, the wrong Python
version, …) before they kick off a paid run.

Exit codes:
    * **0** — every check passed, or any failures were merely informational
      (e.g. an unset API key, or no config in this directory yet).
    * **1** — at least one **hard** failure was reported (currently: an
      ``aimigrate.yaml`` exists in the cwd but doesn't validate).

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

from aimigrate import __version__
from aimigrate.config.loader import ConfigError, load_config

CheckStatus = Literal["ok", "warn", "fail"]

CONFIG_FILENAME: Final = "aimigrate.yaml"
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
            ``"aimigrate.yaml"``).
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
        cwd: Directory to look in for ``aimigrate.yaml``.
        env: Environment-variable mapping (typically ``os.environ``).

    Returns:
        One :class:`CheckResult` per row in the doctor table, in display order.
    """
    results = [_python_check()]
    results.extend(_api_key_check(env, key) for key in PROVIDER_KEYS)
    results.append(_config_check(cwd))
    return results


def _python_check() -> CheckResult:
    v = sys.version_info
    return CheckResult(
        name=f"Python {v.major}.{v.minor}.{v.micro}",
        status="ok",
        detail=f"AIMigrate {__version__}",
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
            detail=f"not found in {cwd} (run `aimigrate init` to create one)",
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
