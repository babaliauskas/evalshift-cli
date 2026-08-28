"""Phase B1: the CLI addresses hosted runs by the server-minted id.

``POST /runs`` mints the canonical run id (a UUID) and hands back the URLs that
carry it. The CLI's own ``r_YYYYMMDD_<suite>_<6hex>`` id survives only as the
per-project idempotency key inside ``manifest.run_id`` — it is never an address.
There is no compatibility path back to the old ``run_id`` response key.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalshift.hosted.bundle import BUNDLE_FILENAME, build_bundle
from evalshift.hosted.client import HostedClient, HostedHTTPError
from evalshift.hosted.push import PushError, PushResult, push_bundle
from evalshift.runner.checkpoint import PUSH_STATE_FILENAME
from tests.conftest import FakeHostedClient, write_completed_run, write_project_files

CLIENT_RUN_ID = "r_20260516_abcdef"
SERVER_RUN_ID = "0f1d2c3b-4a59-4c8e-9b1f-2d3e4f5a6b7c"
FINALIZE_URL = f"/runs/{SERVER_RUN_ID}/finalize"
"""The server returns a host-relative finalize path, never an absolute URL."""
VIEW_URL = f"https://app.evalshift.test/app/acme/model-migration/runs/{SERVER_RUN_ID}"
PROJECT_SLUG = "acme/model-migration"
_OTHER_SERVER_RUN_ID = "11111111-2222-3333-4444-555555555555"
"""Some run that is not this bundle's — a stale or tampered checkpoint's id."""


def _pending_upload_response() -> dict[str, Any]:
    """The server's ``RunUploadResponse`` for a freshly created run."""
    return {
        "id": SERVER_RUN_ID,
        "client_run_id": CLIENT_RUN_ID,
        "status": "pending_upload",
        "upload_url": "https://storage.test/upload",
        "finalize_url": FINALIZE_URL,
        "view_url": VIEW_URL,
        "canonical_thresholds": {"pass_rate_min": 0.9},
    }


def _fake_client(**kwargs: Any) -> FakeHostedClient:
    """The shared fake, wired to the server id and view URL this module scripts."""
    kwargs.setdefault("finalize_response", {"id": SERVER_RUN_ID, "view_url": VIEW_URL})
    return FakeHostedClient(**kwargs)


def _runs_base(tmp_path: Path) -> Path:
    return tmp_path / ".evalshift" / "runs"


def _bundle_for_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_project_files(tmp_path)
    write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    result = build_bundle(
        CLIENT_RUN_ID,
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=_runs_base(tmp_path),
    )
    return result.path


def _push(bundle_path: Path, tmp_path: Path, **kwargs: Any) -> PushResult:
    """``push_bundle`` against this project's config and run directory.

    ``runs_base`` is always passed: it is where ``push_state.json`` goes, and
    defaulting it would put the checkpoint under the *current* directory rather
    than the temporary project.
    """
    return push_bundle(
        bundle_path,
        config_path=tmp_path / "evalshift.yaml",
        runs_base=_runs_base(tmp_path),
        **kwargs,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeHostedClient,
) -> None:
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)
    monkeypatch.setattr("evalshift.hosted.push._put_with_retries", lambda *a, **k: None)


def _write_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / PUSH_STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint_payload(bundle_path: Path, **overrides: str) -> dict[str, str]:
    """A complete ``push_state.json`` for ``bundle_path``, exactly as a push writes it.

    ``bundle_sha256`` is the digest of the bytes that were uploaded, so a payload
    built from the bundle currently on disk describes a resumable push.
    """
    payload = {
        "client_run_id": CLIENT_RUN_ID,
        "server_run_id": SERVER_RUN_ID,
        "project_slug": PROJECT_SLUG,
        "finalize_url": FINALIZE_URL,
        "view_url": VIEW_URL,
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }
    payload.update(overrides)
    return payload


def test_push_finalizes_at_the_server_supplied_url_and_returns_the_server_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every address in the push path comes from the server id, not the client id."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.finalize_urls == [FINALIZE_URL]
    assert result.run_id == SERVER_RUN_ID
    assert SERVER_RUN_ID in result.view_url
    assert CLIENT_RUN_ID not in result.view_url


def test_push_rejects_a_create_response_carrying_only_the_legacy_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No compat shim: an ``id``-less response is an error, never a fallback."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    response = _pending_upload_response()
    del response["id"]
    response["run_id"] = CLIENT_RUN_ID
    fake = _fake_client(responses=[response])
    _install(monkeypatch, fake)

    with pytest.raises(PushError, match="did not return a run id"):
        _push(bundle_path, tmp_path)

    assert fake.finalize_urls == []


def test_push_rejects_a_create_response_without_a_finalize_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI no longer knows how to build the finalize path itself."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    response = _pending_upload_response()
    del response["finalize_url"]
    fake = _fake_client(responses=[response])
    _install(monkeypatch, fake)

    with pytest.raises(PushError, match="did not return a finalize_url"):
        _push(bundle_path, tmp_path)


def test_push_checkpoint_records_both_ids_before_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The crash window is between upload and finalize, so the ids land before it."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}

    def snapshot() -> None:
        seen.update(json.loads((bundle_path.parent / PUSH_STATE_FILENAME).read_text()))

    fake = _fake_client(responses=[_pending_upload_response()], on_finalize=snapshot)
    _install(monkeypatch, fake)

    _push(bundle_path, tmp_path)

    assert seen["client_run_id"] == CLIENT_RUN_ID
    assert seen["server_run_id"] == SERVER_RUN_ID
    assert seen["finalize_url"] == FINALIZE_URL
    # The digest of the bytes actually uploaded, so a resume can tell whether the
    # local bundle is still the one the server holds.
    assert seen["bundle_sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()


def test_push_checkpoint_is_cleared_once_the_run_is_finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    _push(bundle_path, tmp_path)

    assert not (bundle_path.parent / PUSH_STATE_FILENAME).exists()


def test_push_of_an_out_of_tree_bundle_leaves_no_empty_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``push --bundle <path>`` must not litter a run directory it never ran.

    The checkpoint directory is minted by the write and is the only thing in it,
    so once the resume hint is dropped there is nothing left to keep — and the
    user never asked for a run directory in the first place.
    """
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    detached = tmp_path / "elsewhere"
    detached.mkdir()
    detached_bundle = detached / BUNDLE_FILENAME
    detached_bundle.write_bytes(bundle_path.read_bytes())
    runs_base = tmp_path / "fresh" / "runs"
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = push_bundle(
        detached_bundle,
        config_path=tmp_path / "evalshift.yaml",
        runs_base=runs_base,
    )

    assert result.run_id == SERVER_RUN_ID
    assert not (runs_base / CLIENT_RUN_ID).exists()


def test_resumed_push_reuses_the_stored_server_id_without_creating_a_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A push interrupted after upload finalizes the run it already created."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _write_checkpoint(bundle_path.parent, _checkpoint_payload(bundle_path))
    fake = _fake_client(responses=[])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 0
    assert fake.finalize_urls == [FINALIZE_URL]
    assert result.run_id == SERVER_RUN_ID
    assert not (bundle_path.parent / PUSH_STATE_FILENAME).exists()


def test_checkpoint_from_a_different_client_run_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale checkpoint must never finalize the run the bundle does not describe."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _write_checkpoint(
        bundle_path.parent,
        _checkpoint_payload(
            bundle_path,
            client_run_id="r_20260101_stale",
            server_run_id=_OTHER_SERVER_RUN_ID,
            finalize_url=f"/runs/{_OTHER_SERVER_RUN_ID}/finalize",
            view_url="https://app.evalshift.test/stale",
        ),
    )
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 1
    assert fake.finalize_urls == [FINALIZE_URL]
    assert result.run_id == SERVER_RUN_ID


def test_resume_starts_over_when_the_local_bundle_is_no_longer_the_uploaded_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuilt bundle must never be published as the bytes the server already holds.

    Crash between upload and finalize, re-run ``analyze``/``report`` (or ``bundle``
    after a config edit), then push again: finalizing the checkpoint would report
    ``uploaded=True`` and print a URL for content that is no longer on disk. The
    server cannot catch it — ``manifest.run_id`` is unchanged.
    """
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _write_checkpoint(bundle_path.parent, _checkpoint_payload(bundle_path))
    uploaded_bytes = bundle_path.read_bytes()
    # The bundle is rebuilt after the crash: same client run id, different bytes.
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/rebuilt-after-the-crash")
    build_bundle(
        CLIENT_RUN_ID,
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=_runs_base(tmp_path),
    )
    assert bundle_path.read_bytes() != uploaded_bytes

    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 1
    assert fake.finalize_urls == [FINALIZE_URL]
    assert result.run_id == SERVER_RUN_ID


def test_checkpoint_whose_finalize_url_does_not_carry_its_server_id_is_discarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The id reported and the run finalized are the same run, or the file is junk.

    ``push_state.json`` is plain JSON on disk; a hand-edited one whose id and
    finalize path disagree would otherwise print one run's id while publishing
    another's bytes.
    """
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    _write_checkpoint(
        bundle_path.parent,
        _checkpoint_payload(bundle_path, server_run_id=_OTHER_SERVER_RUN_ID),
    )
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 1
    assert fake.finalize_urls == [FINALIZE_URL]
    assert result.run_id == SERVER_RUN_ID


def test_checkpoint_stays_under_evalshift_when_the_bundle_lives_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``push --bundle`` may point anywhere; the resume hint still lands in the run dir.

    ``.evalshift/`` is the only directory the CLI gitignores, so a checkpoint
    written beside an arbitrary ``--bundle`` path — a CI artifacts directory, a
    working tree — is generated state escaping into the user's project.
    """
    built = _bundle_for_push(tmp_path, monkeypatch)
    artifacts = tmp_path / "ci-artifacts"
    artifacts.mkdir()
    stray = artifacts / BUNDLE_FILENAME
    stray.write_bytes(built.read_bytes())
    seen: dict[str, bool] = {}

    def snapshot() -> None:
        seen["beside_bundle"] = (artifacts / PUSH_STATE_FILENAME).exists()
        seen["run_dir"] = (_runs_base(tmp_path) / CLIENT_RUN_ID / PUSH_STATE_FILENAME).exists()

    fake = _fake_client(responses=[_pending_upload_response()], on_finalize=snapshot)
    _install(monkeypatch, fake)

    _push(stray, tmp_path)

    assert seen == {"beside_bundle": False, "run_dir": True}


def test_unreadable_checkpoint_is_discarded_and_the_push_starts_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract as ``read_state``'s callers: junk on disk counts as absent."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    (bundle_path.parent / PUSH_STATE_FILENAME).write_text("{ not json", encoding="utf-8")
    fake = _fake_client(responses=[_pending_upload_response()])
    _install(monkeypatch, fake)

    result = _push(bundle_path, tmp_path)

    assert fake.initiate_calls == 1
    assert result.run_id == SERVER_RUN_ID


def test_failed_finalize_drops_the_checkpoint_so_the_next_push_is_a_full_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged finalize must not strand every later push on the same dead URL."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[_pending_upload_response()],
        finalize_error=HostedHTTPError(500, "boom"),
    )
    _install(monkeypatch, fake)

    with pytest.raises(PushError, match="finalize failed"):
        _push(bundle_path, tmp_path)

    assert not (bundle_path.parent / PUSH_STATE_FILENAME).exists()


def test_finalize_run_posts_the_path_the_server_supplied_to_the_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client no longer joins a base URL to a run id it does not own.

    The server owns the path; the CLI owns the origin. The request is the two
    halves joined, never a path the CLI rebuilt from an id.
    """
    seen: dict[str, Any] = {}

    class _FakeHTTPClient:
        def __init__(self, **_: Any) -> None:
            pass

        def __enter__(self) -> _FakeHTTPClient:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            seen["method"] = method
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            return httpx.Response(200, json={"id": SERVER_RUN_ID, "view_url": VIEW_URL})

    monkeypatch.setattr(httpx, "Client", _FakeHTTPClient)
    client = HostedClient(host="https://api.evalshift.test", token="es_secret")

    finalized = client.finalize_run(FINALIZE_URL)

    assert seen["method"] == "POST"
    assert seen["url"] == f"https://api.evalshift.test{FINALIZE_URL}"
    assert finalized["id"] == SERVER_RUN_ID


def test_finalize_run_rejects_a_url_that_is_not_a_host_relative_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bearer token never leaves over a scheme or origin the CLI did not choose."""
    client = HostedClient(host="https://api.evalshift.test", token="es_secret")

    with pytest.raises(HostedHTTPError, match="finalize URL"):
        client.finalize_run("file:///etc/passwd")


def test_push_finalizes_through_the_real_client_at_the_configured_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end over a mocked transport, with no fake client in the way.

    Every other push test substitutes ``FakeHostedClient``, so a ``finalize_url``
    fixture that does not match what the server actually sends stays invisible.
    This one drives the real :class:`HostedClient`: the server's host-relative
    path must resolve against the configured host.
    """
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PUT":
            return httpx.Response(200)
        if request.url.path == "/runs":
            return httpx.Response(200, json=_pending_upload_response())
        return httpx.Response(200, json={"id": SERVER_RUN_ID, "view_url": VIEW_URL})

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")

    result = _push(bundle_path, tmp_path)

    finalize_requests = [
        request for request in seen if request.method == "POST" and request.url.path != "/runs"
    ]
    assert len(finalize_requests) == 1
    assert str(finalize_requests[0].url) == f"https://api.evalshift.test{FINALIZE_URL}"
    assert finalize_requests[0].headers["Authorization"] == "Bearer es_secret"
    assert result.run_id == SERVER_RUN_ID


def test_finalize_url_from_a_foreign_origin_is_refused_before_any_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response cannot redirect the push's bearer token to another host."""
    bundle_path = _bundle_for_push(tmp_path, monkeypatch)
    response = _pending_upload_response()
    response["finalize_url"] = f"https://evil.example/runs/{SERVER_RUN_ID}/finalize"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "PUT":
            return httpx.Response(200)
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")

    with pytest.raises(PushError, match="unusable finalize URL"):
        _push(bundle_path, tmp_path)

    assert not any(request.url.host == "evil.example" for request in seen)
