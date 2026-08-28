"""Phase B3.2: the CLI validates a prebuilt bundle, and warns before the server has to.

Two gaps this closes. ``push --bundle`` handed the file straight to the upload
without looking inside it, so a bundle built by an older CLI — or hand-edited,
or produced by another tool — spent a full upload before finalize rejected it.
And ``BUNDLE_SPEC.md`` has always named a 50 MB soft limit that nothing warned
about, leaving the 100 MB hard limit to arrive as an HTTP 413.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from evalshift.cli.main import app
from evalshift.hosted.bundle import build_bundle
from evalshift.hosted.push import (
    HARD_LIMIT_BYTES,
    SOFT_LIMIT_BYTES,
    PushError,
    _soft_limit_warning,
    push_bundle,
)
from tests.conftest import FakeHostedClient, write_completed_run, write_project_files

CLIENT_RUN_ID = "r_20260516_abcdef"
SERVER_RUN_ID = "0f1d2c3b-4a59-4c8e-9b1f-2d3e4f5a6b7c"
FINALIZE_URL = f"/runs/{SERVER_RUN_ID}/finalize"
VIEW_URL = f"https://app.evalshift.test/app/acme/model-migration/runs/{SERVER_RUN_ID}"


def _runs_base(tmp_path: Path) -> Path:
    return tmp_path / ".evalshift" / "runs"


def _bundle_for_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_project_files(tmp_path)
    write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    return build_bundle(
        CLIENT_RUN_ID,
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=_runs_base(tmp_path),
    ).path


def _rewrite(bundle_path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Edit a built bundle in place, the way a stale or hand-edited file would differ."""
    with gzip.open(bundle_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    mutate(payload)
    with gzip.open(bundle_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeHostedClient) -> None:
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)
    monkeypatch.setattr("evalshift.hosted.push._put_with_retries", lambda *a, **k: None)


def _fake() -> FakeHostedClient:
    return FakeHostedClient(
        responses=[
            {
                "id": SERVER_RUN_ID,
                "client_run_id": CLIENT_RUN_ID,
                "status": "pending_upload",
                "upload_url": "https://storage.test/upload",
                "finalize_url": FINALIZE_URL,
                "view_url": VIEW_URL,
            }
        ],
        finalize_response={"id": SERVER_RUN_ID, "view_url": VIEW_URL},
    )


def _push(bundle_path: Path, tmp_path: Path, **kwargs: Any) -> Any:
    return push_bundle(
        bundle_path,
        config_path=tmp_path / "evalshift.yaml",
        runs_base=_runs_base(tmp_path),
        **kwargs,
    )


# --- prebuilt bundle validation ---------------------------------------------


def _drop_budget_results(payload: dict[str, Any]) -> None:
    """What a bundle from a CLI predating the required-collections rule looks like."""
    payload["decision"].pop("budget_results")


def _non_utc_created_at(payload: dict[str, Any]) -> None:
    payload["manifest"]["created_at"] = "2026-05-15T12:34:56+02:00"


def _slice_named_overall(payload: dict[str, Any]) -> None:
    payload["decision"]["slices"]["overall"] = {
        "name": "overall",
        "verdict": "pass",
        "metrics": payload["decision"]["overall"],
        "budget_results": [],
    }


@pytest.mark.parametrize(
    "mutate",
    [_drop_budget_results, _non_utc_created_at, _slice_named_overall],
    ids=["missing-collection", "non-utc-timestamp", "overall-slice"],
)
def test_push_rejects_a_schema_invalid_bundle_before_any_http_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """The upload is the expensive half; the check that avoids it must come first."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _rewrite(bundle_path, mutate)
    fake = _fake()
    _install(monkeypatch, fake)

    with pytest.raises(PushError, match="schema validation"):
        _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 0


def test_push_accepts_a_freshly_built_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not reject what this CLI itself produces."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake()
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert result.run_id == SERVER_RUN_ID
    assert fake.initiate_calls == 1


def test_push_reports_an_unreadable_bundle_as_a_push_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated or non-gzip file is a message, not a traceback."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    bundle_path.write_bytes(b"not gzip at all")
    fake = _fake()
    _install(monkeypatch, fake)

    with pytest.raises(PushError):
        _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 0


# --- soft-limit warning ------------------------------------------------------


def test_limits_match_the_bundle_spec() -> None:
    """`BUNDLE_SPEC.md` §Size limits: 50 MB soft, 100 MB hard, both compressed."""
    assert SOFT_LIMIT_BYTES == 50 * 1024 * 1024
    assert HARD_LIMIT_BYTES == 100 * 1024 * 1024


def test_no_warning_one_byte_below_the_soft_limit() -> None:
    assert _soft_limit_warning(SOFT_LIMIT_BYTES - 1) is None


def test_warning_at_exactly_the_soft_limit_names_the_hard_limit() -> None:
    """The number that matters to the reader is the one that will reject the run."""
    warning = _soft_limit_warning(SOFT_LIMIT_BYTES)
    assert warning is not None
    assert "50 MB" in warning
    assert "100 MB" in warning


def test_push_prints_the_soft_limit_warning_and_still_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring: a warning, not a refusal — the server owns the hard limit."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake()
    _install(monkeypatch, fake)
    monkeypatch.setattr("evalshift.hosted.push.SOFT_LIMIT_BYTES", 1)
    console = Console(file=io.StringIO(), width=200)

    result = _push(bundle_path, tmp_path, console=console)

    assert "100 MB" in str(console.file.getvalue())
    assert result.uploaded is True


def test_push_is_silent_for_an_ordinary_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake()
    _install(monkeypatch, fake)
    console = Console(file=io.StringIO(), width=200)

    _push(bundle_path, tmp_path, console=console)

    assert "soft limit" not in str(console.file.getvalue())


def test_push_bundle_command_exits_nonzero_on_an_invalid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through Typer: the validation failure is an exit code, not a traceback.

    ``push`` already turns a ``PushError`` into exit 1; this pins that the new
    check reaches that handler rather than escaping as a ``BundleError``.
    """
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _rewrite(bundle_path, _drop_budget_results)
    fake = _fake()
    _install(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["push", "--bundle", str(bundle_path)])

    assert result.exit_code == 1, result.output
    assert "schema validation" in result.output
    assert fake.initiate_calls == 0
