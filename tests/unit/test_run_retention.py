"""Unit tests for run-directory retention (`prune_runs` + `resolve_max_runs`)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalshift.runner.checkpoint import (
    prune_runs,
    resolve_max_runs,
    suite_of_run,
    write_state,
)
from evalshift.runner.models import RunModels, RunState

# A fixed "now" so TTL tests are deterministic (no wall-clock dependency).
NOW = 1_000_000_000.0
DAY = 86_400.0


def _make_run(base: Path, name: str, *, age_days: float = 0.0, status: str | None = None) -> Path:
    """Create a run dir under ``base`` with mtime ``age_days`` before NOW.

    When ``status`` is given, a valid ``state.json`` is written so `_is_in_progress` can read it;
    otherwise the dir has no state (still a valid prune candidate).
    """
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if status is not None:
        write_state(
            run_dir,
            RunState(
                run_id=name,
                status=status,  # type: ignore[arg-type]
                config_hash="deadbeef",
                started_at=datetime.fromtimestamp(NOW, tz=UTC),
                models=RunModels(source="gpt-a", target="gpt-b"),
                prompt_ids=["p1"],
                suite_path="golden.jsonl",
                total_evaluations=1,
            ),
        )
    mtime = NOW - age_days * DAY
    os.utime(run_dir, (mtime, mtime))
    return run_dir


@pytest.fixture
def runs_base(tmp_path: Path) -> Path:
    base = tmp_path / "runs"
    base.mkdir()
    return base


def _names(paths: list[Path]) -> set[str]:
    return {p.name for p in paths}


def test_suite_of_run_parsing() -> None:
    assert suite_of_run("r_20260708_main_chat_0600ef") == "main_chat"
    assert suite_of_run("r_20260708_briefing_ab12cd") == "briefing"
    assert suite_of_run("r_20260625_32a3ba") == ""  # no slug
    assert suite_of_run("not_a_run") == ""


def test_keep_newest_n_per_suite(runs_base: Path) -> None:
    for i in range(5):
        _make_run(runs_base, f"r_2026070{i}_main_chat_00000{i}", age_days=5 - i)
    removed = prune_runs(runs_base, max_runs_per_suite=2, now_ts=NOW)
    # Oldest 3 removed, newest 2 (age 1 and 0 days) survive.
    assert len(removed) == 3
    survivors = _names(list(runs_base.iterdir()))
    assert survivors == {"r_20260703_main_chat_000003", "r_20260704_main_chat_000004"}


def test_grouping_is_per_suite(runs_base: Path) -> None:
    # 3 of each suite; keep 1 per suite -> 2 removed per suite = 4 removed, 2 survive.
    for i in range(3):
        _make_run(runs_base, f"r_2026070{i}_alpha_a0000{i}", age_days=3 - i)
        _make_run(runs_base, f"r_2026070{i}_beta_b0000{i}", age_days=3 - i)
    removed = prune_runs(runs_base, max_runs_per_suite=1, now_ts=NOW)
    assert len(removed) == 4
    survivors = _names(list(runs_base.iterdir()))
    assert survivors == {"r_20260702_alpha_a00002", "r_20260702_beta_b00002"}


def test_ttl_prunes_old_runs(runs_base: Path) -> None:
    _make_run(runs_base, "r_20260101_x_aaaaaa", age_days=40)
    _make_run(runs_base, "r_20260601_x_bbbbbb", age_days=2)
    # count disabled, TTL = 30 days -> only the 40-day-old run goes.
    removed = prune_runs(runs_base, max_runs_per_suite=0, run_ttl_days=30, now_ts=NOW)
    assert _names(removed) == {"r_20260101_x_aaaaaa"}


def test_count_and_ttl_union(runs_base: Path) -> None:
    _make_run(runs_base, "r_a_x_000001", age_days=40)  # old -> TTL
    _make_run(runs_base, "r_b_x_000002", age_days=3)
    _make_run(runs_base, "r_c_x_000003", age_days=2)
    _make_run(runs_base, "r_d_x_000004", age_days=1)  # newest
    # keep newest 2 by count AND drop anything >30d. The 40d one is both.
    removed = prune_runs(runs_base, max_runs_per_suite=2, run_ttl_days=30, now_ts=NOW)
    assert _names(removed) == {"r_a_x_000001", "r_b_x_000002"}


def test_in_progress_never_pruned(runs_base: Path) -> None:
    _make_run(runs_base, "r_1_x_000001", age_days=3, status="in_progress")
    _make_run(runs_base, "r_2_x_000002", age_days=2, status="completed")
    _make_run(runs_base, "r_3_x_000003", age_days=1, status="completed")
    removed = prune_runs(runs_base, max_runs_per_suite=1, now_ts=NOW)
    # Only the completed #2 is evictable; the in_progress #1 is protected despite being oldest.
    assert _names(removed) == {"r_2_x_000002"}
    assert (runs_base / "r_1_x_000001").exists()


def test_keep_run_id_never_pruned(runs_base: Path) -> None:
    for i in range(3):
        _make_run(runs_base, f"r_{i}_x_00000{i}", age_days=3 - i)
    # keep the OLDEST as keep_run_id even though count would evict it.
    removed = prune_runs(runs_base, max_runs_per_suite=1, keep_run_id="r_0_x_000000", now_ts=NOW)
    assert "r_0_x_000000" not in _names(removed)
    assert (runs_base / "r_0_x_000000").exists()


def test_max_runs_zero_is_noop(runs_base: Path) -> None:
    for i in range(5):
        _make_run(runs_base, f"r_{i}_x_00000{i}")
    assert prune_runs(runs_base, max_runs_per_suite=0, now_ts=NOW) == []


def test_dry_run_deletes_nothing(runs_base: Path) -> None:
    for i in range(4):
        _make_run(runs_base, f"r_{i}_x_00000{i}", age_days=4 - i)
    candidates = prune_runs(runs_base, max_runs_per_suite=1, dry_run=True, now_ts=NOW)
    assert len(candidates) == 3
    # Everything still on disk.
    assert len(list(runs_base.iterdir())) == 4


def test_suite_filter_restricts_scope(runs_base: Path) -> None:
    for i in range(3):
        _make_run(runs_base, f"r_{i}_alpha_a0000{i}", age_days=3 - i)
        _make_run(runs_base, f"r_{i}_beta_b0000{i}", age_days=3 - i)
    removed = prune_runs(runs_base, max_runs_per_suite=1, suite="alpha", now_ts=NOW)
    assert all(suite_of_run(p.name) == "alpha" for p in removed)
    assert len(removed) == 2
    # All beta runs untouched.
    assert len([p for p in runs_base.iterdir() if suite_of_run(p.name) == "beta"]) == 3


def test_missing_base_is_noop(tmp_path: Path) -> None:
    assert prune_runs(tmp_path / "nope", max_runs_per_suite=2, now_ts=NOW) == []


def test_resolve_max_runs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALSHIFT_MAX_RUNS", raising=False)
    assert resolve_max_runs(20) == 20  # no env -> config value
    monkeypatch.setenv("EVALSHIFT_MAX_RUNS", "5")
    assert resolve_max_runs(20) == 5
    monkeypatch.setenv("EVALSHIFT_MAX_RUNS", "unlimited")
    assert resolve_max_runs(20) == 0
    monkeypatch.setenv("EVALSHIFT_MAX_RUNS", "0")
    assert resolve_max_runs(20) == 0
    monkeypatch.setenv("EVALSHIFT_MAX_RUNS", "garbage")
    assert resolve_max_runs(20) == 20  # malformed -> fall back to config value
