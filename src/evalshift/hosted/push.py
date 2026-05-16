"""Hosted push flow for EvalShift run bundles."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from evalshift.config.loader import ConfigError, load_config
from evalshift.hosted.bundle import BUNDLE_FILENAME, build_bundle, canonical_json, load_bundle
from evalshift.hosted.client import HostedClient, HostedError, HostedHTTPError, HostedNetworkError
from evalshift.hosted.credentials import CredentialsError, resolve_credentials
from evalshift.runner.checkpoint import run_dir_for

_TRANSIENT_STATUSES = {500, 502, 503, 504}


class PushError(Exception):
    """Raised when a bundle cannot be pushed to hosted EvalShift."""


@dataclass(frozen=True, slots=True)
class PushResult:
    run_id: str
    view_url: str
    uploaded: bool
    project_created: bool = False


def push_local_run(
    *,
    run_id: str,
    config_path: Path,
    suite_path: Path,
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
) -> PushResult:
    """Push a prebuilt bundle through signed upload and finalize."""
    try:
        credentials = resolve_credentials(host=host, token=token)
    except CredentialsError as exc:
        raise PushError(str(exc)) from exc
    client = HostedClient(host=credentials.host, token=credentials.token)
    bundle = load_bundle(bundle_path)
    manifest = _manifest(bundle)
    if project is not None and project != manifest["project_slug"]:
        raise PushError(
            f"--project {project!r} does not match bundle project {manifest['project_slug']!r}; "
            "rebuild the bundle with the desired project",
        )
    resolved_thresholds = (
        thresholds if thresholds is not None else _thresholds_from_config(config_path)
    )
    try:
        response = client.initiate_run(manifest, _non_empty(resolved_thresholds))
        project_created = False
    except HostedHTTPError as exc:
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
        response = client.initiate_run(manifest, _non_empty(resolved_thresholds))
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc

    _warn_threshold_drift(console, resolved_thresholds, response)
    run_id = str(response.get("run_id") or manifest["run_id"])
    view_url = str(response.get("view_url") or "")
    upload_url = response.get("upload_url")
    if upload_url is None:
        if not view_url:
            raise PushError("hosted API did not return a view_url for the existing run")
        return PushResult(
            run_id=run_id, view_url=view_url, uploaded=False, project_created=project_created
        )
    if not isinstance(upload_url, str) or not upload_url:
        raise PushError("hosted API returned an invalid upload_url")

    data = bundle_path.read_bytes()
    _put_with_retries(upload_url, data)
    try:
        finalized = client.finalize_run(run_id)
    except (HostedError, HostedHTTPError) as exc:
        raise PushError(f"finalize failed: {exc}") from exc
    final_view_url = str(finalized.get("view_url") or view_url)
    if not final_view_url:
        raise PushError("hosted API did not return a view_url after finalize")
    return PushResult(
        run_id=run_id,
        view_url=final_view_url,
        uploaded=True,
        project_created=project_created,
    )


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
            "cannot auto-create project: the org is inaccessible or this token lacks permission",
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
            "cannot auto-create project: this token must have owner access to the org",
        ) from exc


def _name_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-") if part) or slug


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
    "HostedHTTPError",
    "PushError",
    "PushResult",
    "_put_with_retries",
    "push_bundle",
    "push_local_run",
]
