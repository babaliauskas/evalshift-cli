"""Advisory check that CI installs a CLI at least as new as the local one.

``evalshift.yaml`` is written with ``extra="forbid"`` models, so a config
written by a newer CLI can carry keys an older CLI rejects outright. The
GitHub Action installs an exact CLI version (``evalshift-version``, or its own
default when the input is absent), which means the *reader* in CI must be at
least as new as the *writer* on the developer's machine. This module finds
every ``babaliauskas/evalshift-action`` step under ``.github/workflows/`` and
compares its pin with the running CLI.

Everything here is advisory: unreadable or invalid workflow files are skipped
silently, unparseable version strings are ignored, and no function raises on
odd input. Callers print the finding as a warning and never change exit codes
(decision 4 in the pin-drift plan: the CLI warns, never edits, a workflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from packaging.version import InvalidVersion, Version

#: ``uses:`` prefix that identifies an EvalShift action step.
ACTION_USES_PREFIX: Final = "babaliauskas/evalshift-action@"
#: The action input that pins the CLI version installed in CI.
VERSION_INPUT: Final = "evalshift-version"
#: Version reported by an editable install without package metadata.
UNKNOWN_VERSION: Final = "0.0.0+unknown"
#: Where GitHub reads workflows from, relative to the repository root.
WORKFLOWS_DIR: Final = Path(".github/workflows")

PinStatus = Literal["stale", "unpinned", "ahead"]


@dataclass(frozen=True, slots=True)
class ActionPin:
    """One ``babaliauskas/evalshift-action`` step and the CLI version it pins.

    Attributes:
        workflow: Workflow file path relative to the project root.
        job: Name of the job the step belongs to.
        version: Raw ``evalshift-version`` input, or ``None`` when the input
            is absent (the action's own default applies).
        literal: ``False`` when the value is a ``${{ }}`` expression that can
            only be resolved by GitHub at run time.
    """

    workflow: Path
    job: str
    version: str | None
    literal: bool


@dataclass(frozen=True, slots=True)
class CiPinFinding:
    """What :func:`check_ci_pin` has to say about the workflows it found.

    Attributes:
        status: ``"stale"`` (a literal pin is older than the CLI),
            ``"unpinned"`` (a step relies on the action default), or
            ``"ahead"`` (every literal pin is newer than the local CLI).
        pins: The pins that triggered the finding.
        message: Ready-to-print explanation ending in a ``Fix:`` line. Plain
            text; the first line is meant to follow a warning glyph and every
            later line is indented two spaces to hang under it.
    """

    status: PinStatus
    pins: tuple[ActionPin, ...]
    message: str


def find_action_pins(project_root: Path) -> list[ActionPin]:
    """Collect every EvalShift action step under ``project_root/.github/workflows``.

    Parses each ``*.yml`` / ``*.yaml`` file with :func:`yaml.safe_load`. Files
    that cannot be read or parsed, and documents that are not shaped like a
    workflow, are skipped silently — this is an advisory path, not a linter.

    Args:
        project_root: Directory holding ``evalshift.yaml`` (the repository
            root for the purposes of ``.github/``).

    Returns:
        Pins in file-name order, then job order, then step order. Empty when
        no workflow uses the action.
    """
    workflows_dir = project_root / WORKFLOWS_DIR
    if not workflows_dir.is_dir():
        return []
    pins: list[ActionPin] = []
    for path in sorted(workflows_dir.iterdir()):
        if path.suffix not in {".yml", ".yaml"} or not path.is_file():
            continue
        jobs = _load_jobs(path)
        if jobs is None:
            continue
        workflow = path.relative_to(project_root)
        for job_name, job in jobs.items():
            steps = job.get("steps") if isinstance(job, dict) else None
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                uses = step.get("uses")
                if not isinstance(uses, str) or not uses.startswith(ACTION_USES_PREFIX):
                    continue
                pins.append(_pin_from_step(workflow, str(job_name), step))
    return pins


def _load_jobs(path: Path) -> dict[Any, Any] | None:
    """Return the ``jobs:`` mapping of a workflow file, or ``None`` if unusable."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(doc, dict):
        return None
    jobs = doc.get("jobs")
    return jobs if isinstance(jobs, dict) else None


def _pin_from_step(workflow: Path, job: str, step: dict[Any, Any]) -> ActionPin:
    with_block = step.get("with")
    raw = with_block.get(VERSION_INPUT) if isinstance(with_block, dict) else None
    if raw is None:
        return ActionPin(workflow=workflow, job=job, version=None, literal=True)
    value = str(raw).strip()
    return ActionPin(workflow=workflow, job=job, version=value, literal="${{" not in value)


def _parse_version(text: str) -> Version | None:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def _pinned_version(pin: ActionPin) -> Version | None:
    """The comparable version of a literal pin, or ``None`` for anything else."""
    if pin.version is None or not pin.literal:
        return None
    return _parse_version(pin.version)


def check_ci_pin(project_root: Path, cli_version: str) -> CiPinFinding | None:
    """Compare every workflow's CLI pin with ``cli_version``.

    Rules, evaluated in order (the first that applies wins):

    * no action steps, or ``cli_version`` is :data:`UNKNOWN_VERSION` or
      unparseable → ``None``;
    * any literal pin older than ``cli_version`` → ``"stale"``;
    * any step without an ``evalshift-version`` input → ``"unpinned"``;
    * every parseable literal pin newer than ``cli_version`` → ``"ahead"``;
    * otherwise (equal, or only ``${{ }}`` / unparseable pins) → ``None``.

    Args:
        project_root: Directory holding ``evalshift.yaml``.
        cli_version: Version of the running CLI (``evalshift.__version__``).

    Returns:
        A :class:`CiPinFinding` to print as a warning, or ``None`` when there
        is nothing to say.
    """
    pins = find_action_pins(project_root)
    if not pins or cli_version == UNKNOWN_VERSION:
        return None
    current = _parse_version(cli_version)
    if current is None:
        return None

    parsed = [(pin, _pinned_version(pin)) for pin in pins]
    stale = [pin for pin, v in parsed if v is not None and v < current]
    if stale:
        return _stale_finding(stale, cli_version)
    unpinned = [pin for pin in pins if pin.version is None]
    if unpinned:
        return _unpinned_finding(unpinned, cli_version)
    literal = [(pin, v) for pin, v in parsed if v is not None]
    if literal and all(v > current for _, v in literal):
        return _ahead_finding([pin for pin, _ in literal], cli_version)
    return None


def _where(pins: list[ActionPin]) -> list[str]:
    """One ``(workflow, jobs...)`` clause per distinct workflow, in pin order."""
    jobs_by_workflow: dict[Path, list[str]] = {}
    for pin in pins:
        jobs = jobs_by_workflow.setdefault(pin.workflow, [])
        if pin.job not in jobs:
            jobs.append(pin.job)
    return [
        f"{workflow.as_posix()}, job{'s' if len(jobs) > 1 else ''} {', '.join(jobs)}"
        for workflow, jobs in jobs_by_workflow.items()
    ]


def _join(lines: list[str]) -> str:
    return "\n  ".join(lines)


def _stale_finding(pins: list[ActionPin], cli_version: str) -> CiPinFinding:
    by_version: dict[str, list[ActionPin]] = {}
    for pin in pins:
        by_version.setdefault(pin.version or "", []).append(pin)
    lines: list[str] = []
    for version, group in by_version.items():
        for where in _where(group):
            lines.append(
                f"CI installs evalshift {version} ({where}) but the local CLI is "
                f"{cli_version} — an older CLI rejects config keys a newer one writes."
            )
    lines.append(
        f'Fix: set `{VERSION_INPUT}: "{cli_version}"` on the {ACTION_USES_PREFIX.rstrip("@")} step.'
    )
    return CiPinFinding(status="stale", pins=tuple(pins), message=_join(lines))


def _unpinned_finding(pins: list[ActionPin], cli_version: str) -> CiPinFinding:
    lines = [
        f"CI installs the action's default evalshift version ({where}) but the local CLI "
        f"is {cli_version} — the default may lag behind, and an older CLI rejects config "
        "keys a newer one writes."
        for where in _where(pins)
    ]
    lines.append(
        f'Fix: add `{VERSION_INPUT}: "{cli_version}"` under `with:` on the '
        f"{ACTION_USES_PREFIX.rstrip('@')} step."
    )
    return CiPinFinding(status="unpinned", pins=tuple(pins), message=_join(lines))


def _ahead_finding(pins: list[ActionPin], cli_version: str) -> CiPinFinding:
    by_version: dict[str, list[ActionPin]] = {}
    for pin in pins:
        by_version.setdefault(pin.version or "", []).append(pin)
    lines: list[str] = []
    for version, group in by_version.items():
        for where in _where(group):
            lines.append(
                f"CI installs evalshift {version} ({where}) but the local CLI is "
                f"{cli_version} — local runs use an older CLI than CI."
            )
    lines.append("Fix: `pip install -U evalshift` so local runs and CI agree.")
    return CiPinFinding(status="ahead", pins=tuple(pins), message=_join(lines))


__all__ = [
    "ACTION_USES_PREFIX",
    "UNKNOWN_VERSION",
    "VERSION_INPUT",
    "WORKFLOWS_DIR",
    "ActionPin",
    "CiPinFinding",
    "PinStatus",
    "check_ci_pin",
    "find_action_pins",
]
