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

Not every misconfiguration is visible before the run. :func:`run_checks` is
the pre-flight half; :func:`source_conformance_check` is the post-scoring
half, reported in the same :class:`CheckResult` vocabulary because it is the
same kind of finding — *your setup is wrong*, not *your target model is* —
and ``evaluate`` renders it with this module's own :func:`render_results`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import typer
from rich.console import Console
from rich.table import Table

from evalshift import __version__
from evalshift.captures.toolset import EMPTY_TOOLSET_FINGERPRINT, fingerprint_tools
from evalshift.cli.commands._suites import SUITE_FILENAME
from evalshift.config.loader import ConfigError, load_config
from evalshift.config.models import EvalShiftConfig
from evalshift.evaluators import tool_selection
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.failures import BROKEN_HARNESS_CAUSES
from evalshift.models.registry import PROVIDER_ENV_VARS
from evalshift.suite.models import SuiteExample
from evalshift.utils.ci_pin import check_ci_pin, find_action_pins

CheckStatus = Literal["ok", "warn", "fail"]

CONFIG_FILENAME: Final = "evalshift.yaml"
# One entry per provider, primary env var first then accepted aliases. Sourced
# from the model registry so doctor and the client agree on what authenticates.
PROVIDER_KEYS: Final[tuple[tuple[str, ...], ...]] = tuple(PROVIDER_ENV_VARS.values())

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
    results.extend(_api_key_check(env, aliases) for aliases in PROVIDER_KEYS)
    results.append(_config_check(cwd))
    results.extend(_tool_consistency_checks(cwd))
    results.extend(_ci_pin_check(cwd))
    return results


def _python_check() -> CheckResult:
    v = sys.version_info
    return CheckResult(
        name=f"Python {v.major}.{v.minor}.{v.micro}",
        status="ok",
        detail=f"EvalShift {__version__}",
    )


def _api_key_check(env: Mapping[str, str], aliases: tuple[str, ...]) -> CheckResult:
    """Check one provider's API key, shown under its primary env-var name.

    ``aliases`` is the provider's env vars in preference order (primary first).
    The key counts as set if any alias is present; a non-primary hit is noted so
    the user knows which one carried them.
    """
    primary = aliases[0]
    for alias in aliases:
        if env.get(alias):
            detail = "set" if alias == primary else f"set via {alias}"
            return CheckResult(name=primary, status="ok", detail=detail)
    return CheckResult(
        name=primary,
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


def _named_suite_paths(cwd: Path, cfg: EvalShiftConfig) -> list[tuple[str, Path]]:
    """Return ``(name, path)`` for every golden-suite file the toolset check should inspect.

    Projects created by ``evalshift init`` + ``evalshift capture sync`` keep
    their suites under ``.evalshift/suites/<name>/golden.jsonl`` and wire them
    into the managed ``suites:`` block, so there is no ``./golden.jsonl`` to
    find. Configured suites therefore win, named after their ``suites:`` key;
    the flat ``./golden.jsonl`` layout is the fallback for configs that wire
    no suites at all, named after :data:`~evalshift.cli.commands._suites.SUITE_FILENAME`.

    Args:
        cwd: Project directory holding ``evalshift.yaml``; suite paths in the
            ``suites:`` block are relative to it.
        cfg: The loaded configuration.

    Returns:
        Existing suite files with their names, in config order (empty when
        none exist).
    """
    if cfg.suites:
        candidates = [(name, cwd / entry.path) for name, entry in cfg.suites.items()]
    else:
        candidates = [(SUITE_FILENAME, cwd / SUITE_FILENAME)]
    return [(name, path) for name, path in candidates if path.exists()]


def _example_toolset_fingerprint(example: SuiteExample) -> str:
    """Content-address one example's toolset to a ``sha256:`` string, inline or ref alike.

    A ``toolset_ref`` already *is* that fingerprint (verbatim, content-addressed
    at capture time), so it is used as-is with no sidecar I/O. Inline ``tools``
    are fingerprinted fresh via :func:`~evalshift.captures.toolset.fingerprint_tools`
    -- the same algorithm, so an inline example and a ``toolset_ref`` example
    naming the identical toolset always compare equal.
    """
    if example.toolset_ref is not None:
        return example.toolset_ref
    return fingerprint_tools([t.to_anthropic() for t in example.tools or []])


def _describe_toolset_fingerprint(fingerprint: str) -> str:
    """Human-readable label for one toolset fingerprint in a doctor report line."""
    if fingerprint == EMPTY_TOOLSET_FINGERPRINT:
        return "no tools"
    short = fingerprint.removeprefix("sha256:")[:12]
    return f"toolset sha256:{short}…"


def _suite_toolset_check(name: str, examples: list[SuiteExample]) -> CheckResult:
    """One report row for ``name``: the toolset its examples share, or a differing-toolset flag."""
    fingerprints = [_example_toolset_fingerprint(ex) for ex in examples]
    distinct = sorted(set(fingerprints))
    n = len(examples)
    plural = "s" if n != 1 else ""
    if len(distinct) == 1:
        return CheckResult(
            name=f"toolset: {name}",
            status="ok",
            detail=f"{n} example{plural} share one toolset ({_describe_toolset_fingerprint(distinct[0])})",
        )
    return CheckResult(
        name=f"toolset: {name}",
        status="warn",
        detail=(
            f"{n} example{plural} carry {len(distinct)} different toolsets — legal (the "
            "runner dispatches each example's own toolset, not one shared per prompt), but "
            "confirm the split is intentional."
        ),
    )


def _tool_consistency_checks(cwd: Path) -> list[CheckResult]:
    """v0.3 — report the toolset each configured suite carries.

    One ``CheckResult`` per suite (see :func:`_named_suite_paths`):

    * ``ok`` when every example in the suite shares one toolset fingerprint
      (:func:`_example_toolset_fingerprint`) — including "every example
      shares the empty toolset", the truthful signal for an agent that
      genuinely never sees a tool.
    * ``warn`` when a suite's examples carry more than one distinct
      fingerprint. Legal — different examples may legitimately dispatch
      different toolsets — but also the shape a wiring mistake takes, so it
      is surfaced rather than silently accepted.

    Skipped silently when the config or a suite file can't be loaded, and for
    a suite with zero examples — those cases already surface elsewhere in the
    doctor output (or have nothing to report).
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []

    from evalshift.suite.loader import load_jsonl

    out: list[CheckResult] = []
    for name, suite_path in _named_suite_paths(cwd, cfg):
        try:
            suite = load_jsonl(suite_path)
        except Exception:
            continue
        if not suite.examples:
            continue
        out.append(_suite_toolset_check(name, suite.examples))
    return out


def _ci_pin_check(cwd: Path) -> list[CheckResult]:
    """Report whether CI installs a CLI at least as new as this one.

    One ``ci pin`` row when a workflow under ``cwd/.github/workflows`` uses
    the EvalShift action: ``warn`` with the finding from
    :func:`~evalshift.utils.ci_pin.check_ci_pin`, else ``ok`` naming the pin.
    No row at all when no workflow uses the action.
    """
    pins = find_action_pins(cwd)
    if not pins:
        return []
    finding = check_ci_pin(cwd, __version__)
    if finding is not None:
        return [CheckResult(name="ci pin", status="warn", detail=finding.message)]
    versions = sorted({pin.version for pin in pins if pin.version is not None})
    detail = f"pinned to {', '.join(versions)}" if versions else "not pinned (version unknown)"
    return [CheckResult(name="ci pin", status="ok", detail=detail)]


#: The fraction of a suite's conformance rows the source model has to fail
#: before the run is called a broken harness rather than a migration finding.
#: Half is where the ground truth stops describing the source model at all:
#: the expectations were recorded *from* that model, so it is the one side
#: that should satisfy them, and a coin-flip rate is not sampling noise —
#: it is a different setup. Below half the rows stay a finding, reported as
#: ``TOOL_GROUND_TRUTH_MISS`` counts and excluded from the policy rates, but
#: not an accusation.
BROKEN_HARNESS_RATE: Final = 0.5

#: …and the smallest suite on which that accusation stands on its own data.
#: A 100% failure rate over three rows has a 95% Wilson lower bound of 0.44 —
#: under half — so a three-example smoke suite cannot support the sentence
#: this check prints. At four rows the bound is 0.51 and it can. Sub-threshold
#: suites stay silent rather than guessing.
BROKEN_HARNESS_MIN_ROWS: Final = 4


def source_conformance_check(records: Sequence[EvalRecord]) -> CheckResult | None:
    """Report a suite whose ground truth the *source* model cannot satisfy.

    The conformance axis grades each side absolutely against the expectations
    the suite recorded, and on a captured suite those expectations came from
    the source model. The source is therefore the one side that should always
    conform; when it does not, the comparison never reached the migration —
    it measured the harness, and every rate the run publishes describes that.

    Counted on the **source side alone**, over conformance rows the evaluator
    actually scored:

    * not ``analysis.policy.is_shared_ground_truth_miss``, which is the right
      selector for *excluding* a row from the rates but needs ``delta == 0``
      and a ``TOOL_GROUND_TRUTH_MISS`` tag that requires *both* sides to miss.
      A suite the source fails and the target happens to satisfy is invisible
      to it (``0.0 / 1.0`` is a positive delta and carries no tag) and is
      exactly as misconfigured;
    * errored rows are dropped: their neutral ``0.5`` is a measurement that
      broke, not an expectation that was missed.

    Args:
        records: Every row the scoring stage produced for the run.

    Returns:
        A ``fail``-status :class:`CheckResult` when the source failed at least
        :data:`BROKEN_HARNESS_RATE` of at least :data:`BROKEN_HARNESS_MIN_ROWS`
        conformance rows, else ``None``.
    """
    rows = [r for r in records if r.error is None and r.kind == tool_selection.KIND_CONFORMANCE]
    if len(rows) < BROKEN_HARNESS_MIN_ROWS:
        return None
    failed = sum(1 for r in rows if r.source_score < 1.0)
    rate = failed / len(rows)
    if rate < BROKEN_HARNESS_RATE:
        return None
    return CheckResult(
        name="broken eval harness",
        status="fail",
        detail=(
            f"the source model failed the recorded ground truth on {failed} of "
            f"{len(rows)} tool-selection conformance rows ({rate:.0%}). "
            f"{BROKEN_HARNESS_CAUSES} This run measured the eval harness, not "
            "the migration: no verdict it reports describes the target model "
            "until the suite is re-captured against the agent under test, or "
            "the evaluator's conformance axis is set to off."
        ),
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
    "BROKEN_HARNESS_MIN_ROWS",
    "BROKEN_HARNESS_RATE",
    "CONFIG_FILENAME",
    "PROVIDER_KEYS",
    "CheckResult",
    "CheckStatus",
    "doctor",
    "render_results",
    "run_checks",
    "source_conformance_check",
]
