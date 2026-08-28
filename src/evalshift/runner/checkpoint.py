"""On-disk persistence for in-flight runs.

A run directory looks like:

::

    .evalshift/runs/r_20260601_abc123/
    ├── state.json       # RunState (PDF §5.4)
    ├── raw.jsonl        # one Call per non-blank line, append-only
    └── push_state.json  # PushCheckpoint, only while a hosted push is mid-flight

We avoid clever I/O: ``state.json`` is rewritten with an atomic
write-temp + rename so a crash mid-checkpoint can never corrupt it,
and ``raw.jsonl`` is opened in append mode each time so partial
writes from a crash leave a coherent prefix.

Resume contract:

1. Find the latest ``in_progress`` run dir for the project.
2. Validate ``state.config_hash`` matches the *current* config + suite.
   If not, abort — the user has changed something material and the
   pre-recorded calls would no longer reflect what the user actually
   wants tested.
3. Read the existing ``raw.jsonl`` to build a set of completed
   ``(prompt_id, example_id, role)`` tuples; the orchestrator skips
   those.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalshift.config.models import EvalShiftConfig
from evalshift.runner.models import Call, RunState

DEFAULT_RUNS_DIR: Path = Path(".evalshift") / "runs"
STATE_FILENAME: str = "state.json"
RAW_FILENAME: str = "raw.jsonl"


class CheckpointError(Exception):
    """Raised by checkpoint helpers when resume is impossible."""


# ---------------------------------------------------------------------------
# Run id + hashing
# ---------------------------------------------------------------------------


def slugify_run_component(text: str, *, max_length: int = 32) -> str:
    """Return a filesystem-safe slug for embedding in a run id.

    Lowercases, collapses any run of non-alphanumeric characters into a
    single underscore, trims leading/trailing underscores, and caps the
    length. Returns an empty string if nothing usable remains.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_length].strip("_")


def generate_run_id(now: datetime | None = None, suite_slug: str | None = None) -> str:
    """Generate a fresh run id.

    Without a suite slug: ``r_YYYYMMDD_<6hex>``. With one, the slug is
    inserted so the run is self-describing: ``r_YYYYMMDD_<slug>_<6hex>``.
    The date stays first so run directories still sort chronologically.

    Using ``secrets.token_hex(3)`` (24 bits of entropy) keeps the suffix
    short while still making collisions in a single day vanishingly
    unlikely (``2**24`` > 16 million).
    """
    when = now or datetime.now(UTC)
    slug = slugify_run_component(suite_slug) if suite_slug else ""
    middle = f"{slug}_" if slug else ""
    return f"r_{when:%Y%m%d}_{middle}{secrets.token_hex(3)}"


def compute_config_hash(config: EvalShiftConfig, suite_path: str) -> str:
    """Hash the config + suite path so resume can detect material changes.

    Uses pydantic's ``model_dump_json`` for canonicalisation so two
    semantically-equivalent configs always hash the same.
    """
    payload = json.dumps(
        {
            "config": json.loads(config.model_dump_json()),
            "suite_path": suite_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Run dir layout
# ---------------------------------------------------------------------------


def run_dir_for(run_id: str, base: Path | None = None) -> Path:
    """Return the directory for a given run id under ``base`` (or default)."""
    return (base or DEFAULT_RUNS_DIR) / run_id


def list_runs(base: Path | None = None) -> list[Path]:
    """Return run directories sorted oldest-first.

    Sort key is the directory name, which embeds the date and a random
    hex tail. Within a single day the order is arbitrary (tail bytes are
    random) but consistent.
    """
    root = base or DEFAULT_RUNS_DIR
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("r_")])


def find_latest_in_progress(base: Path | None = None) -> Path | None:
    """Return the most recently-created ``in_progress`` run dir, if any."""
    for path in reversed(list_runs(base)):
        try:
            state = read_state(path)
        except CheckpointError:
            continue
        if state.status == "in_progress":
            return path
    return None


# ---------------------------------------------------------------------------
# Retention — prune old run directories
# ---------------------------------------------------------------------------


MAX_RUNS_ENV = "EVALSHIFT_MAX_RUNS"


def resolve_max_runs(config_value: int) -> int:
    """Return the effective ``max_runs_per_suite``, applying the ``EVALSHIFT_MAX_RUNS`` override.

    ``0`` / ``none`` / ``unlimited`` (case-insensitive) disables count pruning; a malformed value
    falls back to the config value. The env var wins over ``evalshift.yaml`` but is itself beaten by
    an explicit CLI flag (which the caller resolves before invoking prune).
    """
    raw = os.environ.get(MAX_RUNS_ENV)
    if raw is None:
        return config_value
    raw = raw.strip().lower()
    if raw in {"0", "none", "unlimited", "off", ""}:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return config_value
    return max(value, 0)


def suite_of_run(run_dir_name: str) -> str:
    """Extract the suite slug embedded in a run id, or ``""`` if the id carries none.

    Run ids are ``r_<YYYYMMDD>_<slug>_<6hex>`` (slug present) or ``r_<YYYYMMDD>_<6hex>``
    (no slug). The slug itself may contain underscores, so it is everything between the
    date and the trailing hex tail. Names that don't match the shape group under ``""``.
    """
    parts = run_dir_name.split("_")
    # parts[0]="r", parts[1]=date, parts[-1]=hex tail; slug is whatever sits between.
    if len(parts) < 3 or parts[0] != "r":
        return ""
    return "_".join(parts[2:-1])


def _is_in_progress(run_dir: Path) -> bool:
    """True iff the run dir has a readable ``state.json`` marked ``in_progress``.

    Unreadable/missing state (a corrupt or partial dir) counts as *not* in progress, so
    junk dirs stay eligible for pruning — a live run always has an atomically-written state.
    """
    try:
        return read_state(run_dir).status == "in_progress"
    except CheckpointError:
        return False


def prune_runs(
    base: Path | None = None,
    *,
    max_runs_per_suite: int,
    run_ttl_days: int | None = None,
    keep_run_id: str | None = None,
    suite: str | None = None,
    dry_run: bool = False,
    now_ts: float | None = None,
) -> list[Path]:
    """Delete old run directories under ``base``, grouped per suite. Returns paths removed.

    Two independent rules combine (union), mirroring the SDK's capture GC:

    * **count** — within each suite, keep the newest ``max_runs_per_suite`` dirs by mtime,
      evict the rest. ``<= 0`` disables count pruning.
    * **TTL** — evict any dir whose age exceeds ``run_ttl_days``. ``None`` disables it.

    An ``in_progress`` run and the ``keep_run_id`` dir (the run that just finished) are never
    evicted. ``suite`` restricts pruning to a single suite group. ``dry_run`` returns the dirs
    that *would* be removed without deleting them. No-op when both rules are disabled. Never
    raises into the caller's run.
    """
    if (max_runs_per_suite <= 0) and (run_ttl_days is None):
        return []

    now = now_ts if now_ts is not None else time.time()
    ttl_seconds = run_ttl_days * 86400 if run_ttl_days is not None else None

    # Group run dirs by suite, each carrying its mtime for ordering.
    groups: dict[str, list[tuple[float, Path]]] = {}
    for path in list_runs(base):
        run_suite = suite_of_run(path.name)
        if suite is not None and run_suite != suite:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        groups.setdefault(run_suite, []).append((mtime, path))

    doomed: set[Path] = set()
    for entries in groups.values():
        entries.sort(key=lambda item: item[0])  # oldest first
        if ttl_seconds is not None:
            doomed.update(path for mtime, path in entries if now - mtime > ttl_seconds)
        if max_runs_per_suite > 0 and len(entries) > max_runs_per_suite:
            keep = {path for _, path in entries[len(entries) - max_runs_per_suite :]}
            doomed.update(path for _, path in entries if path not in keep)

    removed: list[Path] = []
    for path in sorted(doomed, key=lambda p: p.name):
        if keep_run_id is not None and path.name == keep_run_id:
            continue
        if _is_in_progress(path):
            continue
        if dry_run:
            removed.append(path)
            continue
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# state.json — atomic read/write
# ---------------------------------------------------------------------------


def read_state(run_dir: Path) -> RunState:
    """Load ``state.json`` from a run directory.

    Raises:
        CheckpointError: If the file is missing, unreadable, or fails
            schema validation.
    """
    state_path = run_dir / STATE_FILENAME
    if not state_path.exists():
        raise CheckpointError(f"state.json not found in {run_dir}")
    try:
        return RunState.model_validate_json(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(f"failed to load state from {state_path}: {exc}") from exc


def write_state(run_dir: Path, state: RunState) -> None:
    """Atomically write ``state.json`` (write-temp + rename).

    Atomicity matters because a crash between the temp write and the
    rename leaves the original file intact, so the resume logic always
    sees a coherent state document.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / STATE_FILENAME
    tmp = run_dir / f"{STATE_FILENAME}.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# push_state.json — hosted push resume
# ---------------------------------------------------------------------------


PUSH_STATE_FILENAME: str = "push_state.json"


@dataclass(frozen=True, slots=True)
class PushCheckpoint:
    """The ids and URLs a hosted push must survive a crash with.

    Written between the bundle upload and the finalize call — the one window
    where the server has already minted a run and taken the bytes but the run is
    not yet visible.

    Attributes:
        client_run_id: The CLI's ``r_YYYYMMDD_<suite>_<6hex>`` id, i.e. the
            bundle's ``manifest.run_id``. Per-project idempotency key only —
            never an address.
        server_run_id: The UUID the server minted at ``POST /runs``. The only id
            the hosted API answers to.
        project_slug: The bundle's ``org/project``, so a checkpoint can never
            finalize a run belonging to a different project.
        finalize_url: The host-relative finalize path the server handed back,
            e.g. ``/runs/{id}/finalize``. Stored verbatim because the CLI does
            not own run addressing and must not rebuild a path from a base plus
            an id.
        view_url: The absolute run URL the server handed back, used when the
            finalize response carries none.
        bundle_sha256: Hex SHA-256 of the bundle bytes that were uploaded. A
            resume compares it against the bundle now on disk, so a bundle
            rebuilt during the crash window can never be published as the stale
            bytes the server is holding.
    """

    client_run_id: str
    server_run_id: str
    project_slug: str
    finalize_url: str
    view_url: str
    bundle_sha256: str


_PUSH_CHECKPOINT_FIELDS = (
    "client_run_id",
    "server_run_id",
    "project_slug",
    "finalize_url",
    "view_url",
    "bundle_sha256",
)


def read_push_checkpoint(directory: Path) -> PushCheckpoint | None:
    """Load ``push_state.json`` from ``directory``, or ``None`` if nothing usable is there.

    Missing, unreadable, malformed, or incomplete files all count as absent —
    the same contract :func:`find_latest_in_progress` applies to ``state.json``.
    A push checkpoint is a crash-recovery hint and never a source of truth, so
    discarding one and starting a fresh push is always a correct outcome. The
    file carries no version field: nothing it says is worth migrating.
    """
    try:
        payload = json.loads((directory / PUSH_STATE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    values: dict[str, str] = {}
    for name in _PUSH_CHECKPOINT_FIELDS:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            return None
        values[name] = value
    return PushCheckpoint(
        client_run_id=values["client_run_id"],
        server_run_id=values["server_run_id"],
        project_slug=values["project_slug"],
        finalize_url=values["finalize_url"],
        view_url=values["view_url"],
        bundle_sha256=values["bundle_sha256"],
    )


def write_push_checkpoint(directory: Path, checkpoint: PushCheckpoint) -> None:
    """Atomically write ``push_state.json`` (write-temp + rename), as :func:`write_state` does."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / PUSH_STATE_FILENAME
    tmp = directory / f"{PUSH_STATE_FILENAME}.tmp"
    tmp.write_text(json.dumps(asdict(checkpoint), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def clear_push_checkpoint(directory: Path) -> None:
    """Delete ``push_state.json`` if it exists; a missing file is not an error.

    Undoes the ``mkdir`` in :func:`write_push_checkpoint` too, so a
    ``push --bundle <path>`` run outside a project leaves no empty
    ``.evalshift/runs/<run_id>/`` behind. ``rmdir`` is the deliberate mechanism:
    it refuses a directory that still holds anything, which is exactly the case
    when the push checkpointed into a real run directory alongside
    ``state.json``, ``raw.jsonl`` and the bundle.
    """
    (directory / PUSH_STATE_FILENAME).unlink(missing_ok=True)
    with suppress(OSError):
        directory.rmdir()


# ---------------------------------------------------------------------------
# raw.jsonl — append + iterate
# ---------------------------------------------------------------------------


def append_call(run_dir: Path, call: Call) -> None:
    """Append a single :class:`Call` to ``raw.jsonl``.

    Each call is its own line so partial writes from a crash leave a
    coherent prefix — the resume reader just skips the trailing blank
    or unparseable line.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / RAW_FILENAME
    with target.open("a", encoding="utf-8") as fh:
        fh.write(call.model_dump_json())
        fh.write("\n")


def iter_calls(run_dir: Path) -> Iterator[Call]:
    """Yield every :class:`Call` from ``raw.jsonl``, skipping malformed lines.

    A trailing partial line from a crash is silently dropped so resume
    just picks up where the last clean write finished.
    """
    target = run_dir / RAW_FILENAME
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                yield Call.model_validate_json(line)
            except Exception:
                # Crash mid-write can leave a truncated final line.
                # Skip it; the resume loop will redo the call.
                continue


def completed_call_keys(run_dir: Path) -> set[tuple[str, str, str]]:
    """Return ``(prompt_id, example_id, role)`` tuples already recorded.

    The orchestrator skips work items whose key is in this set. Errored
    calls count as "done" — we don't auto-retry across resumes because
    the most common cause of a single-call error is deterministic
    (auth, malformed input) and would just fail again.
    """
    return {(c.prompt_id, c.example_id, c.role) for c in iter_calls(run_dir)}


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------


def validate_resume(run_dir: Path, expected_hash: str) -> RunState:
    """Load ``state.json`` and ensure the config hasn't changed.

    Args:
        run_dir: The run directory to resume.
        expected_hash: The :func:`compute_config_hash` of the *current*
            config + suite.

    Raises:
        CheckpointError: If ``state.json`` is missing/corrupt, the run
            isn't in_progress, or the config hash has drifted.
    """
    state = read_state(run_dir)
    if state.status != "in_progress":
        raise CheckpointError(
            f"run {state.run_id} is {state.status!r}, not 'in_progress'",
        )
    if state.config_hash != expected_hash:
        raise CheckpointError(
            "config or suite has changed since this run was started; resume aborted "
            "(start a new run instead, or revert your changes)",
        )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def touch_checkpoint(state: RunState, completed: int) -> RunState:
    """Return a copy of ``state`` with updated checkpoint counters.

    Used by the orchestrator to bump ``completed_evaluations`` and
    ``last_checkpoint_at`` together without mutating the live model.
    """
    return state.model_copy(
        update={
            "completed_evaluations": completed,
            "last_checkpoint_at": _utcnow(),
        },
    )


def _read_json(path: Path) -> Any:
    """Convenience used by tests; ``json.loads`` of a Path's contents."""
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_RUNS_DIR",
    "PUSH_STATE_FILENAME",
    "RAW_FILENAME",
    "STATE_FILENAME",
    "CheckpointError",
    "PushCheckpoint",
    "append_call",
    "clear_push_checkpoint",
    "completed_call_keys",
    "compute_config_hash",
    "find_latest_in_progress",
    "generate_run_id",
    "iter_calls",
    "list_runs",
    "read_push_checkpoint",
    "read_state",
    "run_dir_for",
    "slugify_run_component",
    "touch_checkpoint",
    "validate_resume",
    "write_push_checkpoint",
    "write_state",
]
