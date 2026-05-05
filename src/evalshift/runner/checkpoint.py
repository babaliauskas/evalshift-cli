"""On-disk persistence for in-flight runs.

A run directory looks like:

::

    .evalshift/runs/r_20260601_abc123/
    ├── state.json    # RunState (PDF §5.4)
    └── raw.jsonl     # one Call per non-blank line, append-only

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
import secrets
from collections.abc import Iterator
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


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a fresh ``r_YYYYMMDD_<6hex>`` run id.

    Using ``secrets.token_hex(3)`` (24 bits of entropy) keeps the suffix
    short while still making collisions in a single day vanishingly
    unlikely (``2**24`` > 16 million).
    """
    when = now or datetime.now(UTC)
    return f"r_{when:%Y%m%d}_{secrets.token_hex(3)}"


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
    "RAW_FILENAME",
    "STATE_FILENAME",
    "CheckpointError",
    "append_call",
    "completed_call_keys",
    "compute_config_hash",
    "find_latest_in_progress",
    "generate_run_id",
    "iter_calls",
    "list_runs",
    "read_state",
    "run_dir_for",
    "touch_checkpoint",
    "validate_resume",
    "write_state",
]
