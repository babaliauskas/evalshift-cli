"""Hosted push flow for EvalShift run bundles."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from evalshift.config.loader import ConfigError, load_config
from evalshift.hosted.bundle import (
    BUNDLE_FILENAME,
    BundleError,
    build_bundle,
    canonical_json,
    load_bundle,
    validate_bundle,
)
from evalshift.hosted.client import HostedClient, HostedError, HostedHTTPError, HostedNetworkError
from evalshift.hosted.credentials import CredentialsError, resolve_credentials
from evalshift.runner.checkpoint import (
    PushCheckpoint,
    clear_push_checkpoint,
    read_push_checkpoint,
    run_dir_for,
    write_push_checkpoint,
)

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
"""Upload statuses worth another attempt. 402 must never appear here: a payment error is a
decision, not a hiccup, and retrying it only burns quota the org does not have."""


_PAYMENT_REQUIRED = 402

_MB = 1024 * 1024

SOFT_LIMIT_BYTES = 50 * _MB
"""Compressed size at which the CLI warns. `BUNDLE_SPEC.md` §Size limits."""

HARD_LIMIT_BYTES = 100 * _MB
"""The server's default rejection threshold, quoted in the warning and enforced there."""


def _soft_limit_warning(size_bytes: int) -> str | None:
    """The sentence to print when a bundle crosses the soft limit, else ``None``.

    A warning and never a refusal. The hard limit is configurable server-side,
    so a CLI that enforced its own copy would start rejecting runs the hosted
    plan actually accepts the moment the two numbers drifted apart.
    """
    if size_bytes < SOFT_LIMIT_BYTES:
        return None
    return (
        f"bundle is {size_bytes / _MB:.1f} MB compressed, over the "
        f"{SOFT_LIMIT_BYTES / _MB:.0f} MB soft limit; the server's hard limit is "
        f"{HARD_LIMIT_BYTES / _MB:.0f} MB and it rejects anything larger."
    )


class PushError(Exception):
    """Raised when a bundle cannot be pushed to hosted EvalShift."""


def _upgrade_prompt(exc: HostedHTTPError) -> PushError:
    """Turn the server's 402 into the whole upgrade prompt.

    The CLI never inspects entitlements and never decides whether a plan covers a run — it
    repeats the server's sentence and the link the server built, and exits non-zero.
    """
    lines = ["EvalShift: this run needs a paid plan.", f"  {exc}"]
    details = exc.details
    if isinstance(details, dict):
        upgrade_url = details.get("upgrade_url")
        if isinstance(upgrade_url, str) and upgrade_url:
            lines.append(f"  Upgrade: {upgrade_url}")
    return PushError("\n".join(lines))


@dataclass(frozen=True, slots=True)
class PushResult:
    """Outcome of a hosted push.

    Attributes:
        run_id: The server-minted run id (a UUID). This is the run's only
            address; the CLI's ``r_...`` id lives on solely as the bundle's
            ``manifest.run_id`` idempotency key.
        view_url: The run URL the server built, never one assembled here.
        uploaded: True when this push moved the run from pending to available,
            whether it uploaded the bytes itself or finished an interrupted push
            that had already uploaded them. False when the run was already
            available and there was nothing to do.
        project_created: True when the push auto-created the project first.
    """

    run_id: str
    view_url: str
    uploaded: bool
    project_created: bool = False


def push_local_run(
    *,
    run_id: str,
    config_path: Path,
    suite_path: Path | None = None,
    suite_name: str | None = None,
    runs_base: Path,
    project: str | None = None,
    host: str | None = None,
    token: str | None = None,
    create_project: bool = True,
    console: Console | None = None,
) -> PushResult:
    """Build a missing local bundle if needed, then push it."""
    run_dir = run_dir_for(run_id, runs_base)
    bundle_path = run_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        build_bundle(
            run_id,
            config_path=config_path,
            suite_path=suite_path,
            suite_name=suite_name,
            runs_base=runs_base,
            project=project,
        )
    return push_bundle(
        bundle_path,
        config_path=config_path,
        project=project,
        host=host,
        token=token,
        create_project=create_project,
        console=console,
        runs_base=runs_base,
    )


def push_bundle(
    bundle_path: Path,
    *,
    config_path: Path | None = None,
    project: str | None = None,
    host: str | None = None,
    token: str | None = None,
    thresholds: dict[str, Any] | None = None,
    create_project: bool = True,
    console: Console | None = None,
    runs_base: Path | None = None,
) -> PushResult:
    """Push a prebuilt bundle through signed upload and finalize.

    ``runs_base`` locates the run directory the resume checkpoint is written to.
    It is deliberately not derived from ``bundle_path``: ``--bundle`` may point
    anywhere, and generated state belongs under ``.evalshift/``.
    """
    try:
        credentials = resolve_credentials(host=host, token=token)
    except CredentialsError as exc:
        raise PushError(str(exc)) from exc
    client = HostedClient(host=credentials.host, token=credentials.token)
    # Read and validate before anything reaches the network. ``--bundle`` may
    # point at a file this CLI never wrote — an older build, another tool, a
    # hand-edit — and finalize would reject it only after the whole upload.
    try:
        bundle = load_bundle(bundle_path)
        validate_bundle(bundle)
    except BundleError as exc:
        raise PushError(str(exc)) from exc
    manifest = _manifest(bundle)
    # The size of the file actually uploaded, measured on disk rather than
    # carried inside the bundle it describes.
    upload_size_bytes = bundle_path.stat().st_size
    _warn_soft_limit(console, upload_size_bytes)
    if project is not None and project != manifest["project_slug"]:
        raise PushError(
            f"--project {project!r} does not match bundle project {manifest['project_slug']!r}; "
            "rebuild the bundle with the desired project",
        )
    client_run_id = str(manifest["run_id"])
    project_slug = str(manifest["project_slug"])
    checkpoint_dir = run_dir_for(client_run_id, runs_base)
    bundle_sha256 = _digest(bundle_path)
    resumed = _resume_push(
        client,
        checkpoint_dir,
        client_run_id=client_run_id,
        project_slug=project_slug,
        bundle_sha256=bundle_sha256,
    )
    if resumed is not None:
        return resumed
    resolved_thresholds = (
        thresholds if thresholds is not None else _thresholds_from_config(config_path)
    )
    try:
        response = client.initiate_run(
            manifest,
            size_bytes=upload_size_bytes,
            thresholds=_non_empty(resolved_thresholds),
        )
        project_created = False
    except HostedHTTPError as exc:
        if exc.status_code == _PAYMENT_REQUIRED:
            raise _upgrade_prompt(exc) from exc
        if exc.status_code != 404 or not create_project:
            if exc.status_code == 404:
                raise PushError(
                    "project was not found and auto-creation is disabled or not permitted",
                ) from exc
            raise PushError(str(exc)) from exc
        _auto_create_project(
            client,
            project_slug=str(manifest["project_slug"]),
            thresholds=_non_empty(resolved_thresholds),
        )
        project_created = True
        response = client.initiate_run(
            manifest,
            size_bytes=upload_size_bytes,
            thresholds=_non_empty(resolved_thresholds),
        )
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc

    _warn_threshold_drift(console, resolved_thresholds, response)
    server_run_id = _require_str_field(response, "id", "hosted API did not return a run id")
    view_url = str(response.get("view_url") or "")
    upload_url = response.get("upload_url")
    if upload_url is None:
        if not view_url:
            raise PushError("hosted API did not return a view_url for the existing run")
        return PushResult(
            run_id=server_run_id, view_url=view_url, uploaded=False, project_created=project_created
        )
    if not isinstance(upload_url, str) or not upload_url:
        raise PushError("hosted API returned an invalid upload_url")
    finalize_url = _require_str_field(
        response,
        "finalize_url",
        "hosted API did not return a finalize_url",
    )

    data = bundle_path.read_bytes()
    _put_with_retries(upload_url, data)
    # The bytes are in storage but the run is still invisible: from here until
    # finalize answers, a crash would otherwise lose the server's id and cost a
    # duplicate upload on the next attempt.
    _remember_push(
        checkpoint_dir,
        PushCheckpoint(
            client_run_id=client_run_id,
            server_run_id=server_run_id,
            project_slug=project_slug,
            finalize_url=finalize_url,
            view_url=view_url,
            bundle_sha256=bundle_sha256,
        ),
    )
    final_view_url = _finalize(
        client,
        finalize_url,
        fallback_view_url=view_url,
        checkpoint_dir=checkpoint_dir,
    )
    return PushResult(
        run_id=server_run_id,
        view_url=final_view_url,
        uploaded=True,
        project_created=project_created,
    )


def _resume_push(
    client: HostedClient,
    checkpoint_dir: Path,
    *,
    client_run_id: str,
    project_slug: str,
    bundle_sha256: str,
) -> PushResult | None:
    """Finish a push whose bundle was uploaded but never finalized, else ``None``.

    The checkpoint only exists between the upload and the finalize response, so
    finding one means the server already minted an id for this bundle and
    already holds its bytes. Re-POSTing would just ask for that same id back.
    """
    checkpoint = read_push_checkpoint(checkpoint_dir)
    if checkpoint is None:
        return None
    if not _describes_this_push(
        checkpoint,
        client_run_id=client_run_id,
        project_slug=project_slug,
        bundle_sha256=bundle_sha256,
    ):
        _forget_push(checkpoint_dir)
        return None
    view_url = _finalize(
        client,
        checkpoint.finalize_url,
        fallback_view_url=checkpoint.view_url,
        checkpoint_dir=checkpoint_dir,
    )
    return PushResult(run_id=checkpoint.server_run_id, view_url=view_url, uploaded=True)


def _finalize(
    client: HostedClient,
    finalize_url: str,
    *,
    fallback_view_url: str,
    checkpoint_dir: Path,
) -> str:
    """POST the server's finalize URL and return the run's view URL.

    The checkpoint is dropped either way. A finalize that answered has nothing
    left to resume, and one that failed must not strand every later push on the
    same dead URL — the next push starts over from ``POST /runs``, which is
    idempotent on the bundle's ``client_run_id``.
    """
    try:
        finalized = client.finalize_run(finalize_url)
    except HostedHTTPError as exc:
        _forget_push(checkpoint_dir)
        if exc.status_code == _PAYMENT_REQUIRED:
            raise _upgrade_prompt(exc) from exc
        raise PushError(f"finalize failed: {exc}") from exc
    except HostedError as exc:
        _forget_push(checkpoint_dir)
        raise PushError(f"finalize failed: {exc}") from exc
    _forget_push(checkpoint_dir)
    view_url = str(finalized.get("view_url") or fallback_view_url)
    if not view_url:
        raise PushError("hosted API did not return a view_url after finalize")
    return view_url


def _describes_this_push(
    checkpoint: PushCheckpoint,
    *,
    client_run_id: str,
    project_slug: str,
    bundle_sha256: str,
) -> bool:
    """Whether the checkpoint still describes the bundle about to be pushed.

    Three ways it can stop describing it, all of which mean *start the push
    over* rather than *finalize what the server already holds*:

    * another bundle left it behind — finalizing it would target a run this push
      knows nothing about;
    * the local bundle was rebuilt during the crash window (``analyze``/``report``
      re-run, or ``bundle`` after a config edit) — finalizing would publish the
      *old* bytes and print a URL for content no longer on disk. The server
      cannot catch this: ``manifest.run_id`` is unchanged;
    * the stored id and the stored finalize path disagree — ``push_state.json``
      is plain JSON on disk, and a hand-edited one would otherwise report one
      run's id while publishing another run's bytes.
    """
    return (
        checkpoint.client_run_id == client_run_id
        and checkpoint.project_slug == project_slug
        and checkpoint.bundle_sha256 == bundle_sha256
        and checkpoint.server_run_id in checkpoint.finalize_url
    )


def _digest(bundle_path: Path) -> str:
    """Hex SHA-256 of the bundle file's bytes, streamed rather than buffered."""
    digest = hashlib.sha256()
    with bundle_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_str_field(response: dict[str, Any], key: str, message: str) -> str:
    """Read a required string off the create response — no fallback, no shim."""
    value = response.get(key)
    if not isinstance(value, str) or not value:
        raise PushError(message)
    return value


def _remember_push(checkpoint_dir: Path, checkpoint: PushCheckpoint) -> None:
    """Persist the resume hint. A read-only directory must not fail a live push."""
    try:
        write_push_checkpoint(checkpoint_dir, checkpoint)
    except OSError:
        return


def _forget_push(checkpoint_dir: Path) -> None:
    """Drop the resume hint, best effort."""
    try:
        clear_push_checkpoint(checkpoint_dir)
    except OSError:
        return


def _put_with_retries(
    upload_url: str,
    data: bytes,
    *,
    put: Callable[..., httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 3,
) -> None:
    """PUT bundle bytes, retrying only transient upload failures."""
    putter = put or _httpx_put
    delay = 0.25
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = putter(
                upload_url,
                content=data,
                headers={"Content-Type": "application/gzip"},
            )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise PushError(f"upload failed after {attempt} attempts: {exc}") from exc
            sleep(delay)
            delay *= 2
            continue
        if 200 <= response.status_code < 300:
            return
        if response.status_code in _TRANSIENT_STATUSES and attempt < max_attempts:
            sleep(delay)
            delay *= 2
            continue
        raise PushError(f"upload failed with HTTP {response.status_code}")
    if last_error is not None:
        raise PushError(f"upload failed: {last_error}") from last_error


def _httpx_put(upload_url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
    with httpx.Client(timeout=60.0) as client:
        return client.put(upload_url, content=content, headers=headers)


def _manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        raise PushError("bundle is missing a manifest object")
    return manifest


def _thresholds_from_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise PushError(str(exc)) from exc
    return dict(cfg.thresholds)


def _non_empty(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return value if value else None


def _auto_create_project(
    client: HostedClient,
    *,
    project_slug: str,
    thresholds: dict[str, Any] | None,
) -> None:
    try:
        org_slug, project = project_slug.split("/", 1)
    except ValueError as exc:
        raise PushError(f"invalid project slug {project_slug!r}; expected org/project") from exc
    try:
        projects = client.list_projects(org_slug)
    except HostedHTTPError as exc:
        raise PushError(
            f"cannot auto-create {project_slug!r} at {client.host}: "
            f"{_server_said(exc)}. The org is inaccessible or this token lacks "
            f"permission — check `evalshift whoami` reports this host and an org "
            f"named {org_slug!r}",
        ) from exc
    if any(item.get("slug") == project for item in projects):
        return
    try:
        client.create_project(
            org_slug,
            slug=project,
            name=_name_from_slug(project),
            thresholds=thresholds,
        )
    except HostedHTTPError as exc:
        raise PushError(
            f"cannot auto-create {project_slug!r} at {client.host}: "
            f"{_server_said(exc)}. Creating a project needs owner access to "
            f"{org_slug!r}; a scoped service-account key cannot do it — create the "
            f"project by hand, or push with an owner token",
        ) from exc


def _server_said(exc: HostedHTTPError) -> str:
    """Render a hosted error as ``HTTP <status>: <what the server wrote>``.

    Both auto-create failures used to discard ``exc`` entirely and assert a
    cause ("the org is inaccessible or this token lacks permission"). That is a
    guess: the CLI cannot know, and the one fact that settles it -- the status
    and sentence the server returned -- was the part being thrown away. Pair it
    with the host, since host resolution has four sources (flag, env,
    credentials file, built-in default) and the wrong one produces this same
    failure against a server the user never meant to talk to.
    """
    return f"HTTP {exc.status_code}: {exc}"


def _name_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-") if part) or slug


def _warn_soft_limit(console: Console | None, size_bytes: int) -> None:
    """Print the soft-limit warning, if this bundle earns one."""
    warning = _soft_limit_warning(size_bytes)
    if console is None or warning is None:
        return
    console.print(f"[yellow]![/yellow] {warning}")


def _warn_threshold_drift(
    console: Console | None,
    local: dict[str, Any] | None,
    response: dict[str, Any],
) -> None:
    """Warn when the local thresholds differ from the server's canonical view."""
    if console is None or not local:
        return
    canonical = response.get("canonical_thresholds")
    if not isinstance(canonical, dict):
        return
    if canonical_json(canonical) == canonical_json(local):
        return
    deltas: list[str] = []
    missing = object()
    for key in sorted(set(local) | set(canonical)):
        local_v = local.get(key, missing)
        remote_v = canonical.get(key, missing)
        if local_v != remote_v:
            deltas.append(
                f"  {key}: local={'<unset>' if local_v is missing else repr(local_v)} "
                f"canonical={'<unset>' if remote_v is missing else repr(remote_v)}"
            )
    body = "\n".join(deltas) or "(no per-key differences detected)"
    console.print(
        "[yellow]![/yellow] thresholds in evalshift.yaml differ from project "
        "canonical thresholds:\n" + body
    )


__all__ = [
    "HARD_LIMIT_BYTES",
    "SOFT_LIMIT_BYTES",
    "HostedHTTPError",
    "PushError",
    "PushResult",
    "_put_with_retries",
    "push_bundle",
    "push_local_run",
]
