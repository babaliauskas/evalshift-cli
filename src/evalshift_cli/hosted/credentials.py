"""Credential storage and resolution for hosted EvalShift."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "https://api.evalshift.dev"
_HOST_ENV = "EVALSHIFT_HOST"
_TOKEN_ENV = "EVALSHIFT_TOKEN"
_CREDENTIALS_PATH_ENV = "EVALSHIFT_CREDENTIALS_PATH"


class CredentialsError(Exception):
    """Raised when hosted credentials cannot be resolved or loaded."""


@dataclass(frozen=True, slots=True)
class HostedCredentials:
    host: str
    token: str


def normalize_host(host: str) -> str:
    """Return a host URL without trailing slashes."""
    normalized = host.strip().rstrip("/")
    if not normalized:
        raise CredentialsError("host cannot be empty")
    return normalized


def is_insecure_host(host: str) -> bool:
    """Return True when ``host`` uses plain http against a non-local destination.

    Pushing a bearer token over plain http to a remote host transits the token
    in cleartext; callers should warn users before persisting such a host.
    """
    from urllib.parse import urlparse

    parsed = urlparse(host)
    if parsed.scheme != "http":
        return False
    hostname = (parsed.hostname or "").lower()
    return hostname not in {"localhost", "127.0.0.1", "::1", ""}


def resolve_host(host: str | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve the hosted API base URL using flag, then ``EVALSHIFT_HOST``, then default.

    Mirrors the host-resolution precedence of :func:`resolve_credentials` for callers
    (such as ``evalshift login``) that need a host before any token exists.
    """
    source_env = env or os.environ
    return normalize_host(host or source_env.get(_HOST_ENV) or DEFAULT_HOST)


def credentials_path(path: Path | None = None) -> Path:
    """Return the credential file path, honoring the test override env var."""
    if path is not None:
        return path
    override = os.environ.get(_CREDENTIALS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".evalshift" / "credentials"


def save_credentials(host: str, token: str, *, path: Path | None = None) -> Path:
    """Write hosted credentials with owner-only file permissions."""
    if not token.strip():
        raise CredentialsError("token cannot be empty")
    target = credentials_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        target.parent.chmod(0o700)
    payload = {"host": normalize_host(host), "token": token.strip()}
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, target)
    target.chmod(0o600)
    return target


def load_credentials(*, path: Path | None = None) -> HostedCredentials | None:
    """Load hosted credentials from disk, returning ``None`` when absent."""
    target = credentials_path(path)
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialsError(f"failed to read hosted credentials at {target}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("host"), str):
        raise CredentialsError(f"invalid hosted credentials at {target}")
    if not isinstance(raw.get("token"), str) or not raw["token"].strip():
        raise CredentialsError(f"invalid hosted credentials at {target}")
    return HostedCredentials(host=normalize_host(raw["host"]), token=raw["token"].strip())


def delete_credentials(*, path: Path | None = None) -> bool:
    """Remove stored hosted credentials, returning True when a file was deleted."""
    target = credentials_path(path)
    if not target.exists():
        return False
    try:
        target.unlink()
    except OSError as exc:
        raise CredentialsError(f"failed to remove hosted credentials at {target}: {exc}") from exc
    return True


def resolve_credentials(
    *,
    host: str | None = None,
    token: str | None = None,
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> HostedCredentials:
    """Resolve credentials using flags, env vars, then the credentials file."""
    source_env = env or os.environ
    stored = load_credentials(path=path)
    resolved_host = host or source_env.get(_HOST_ENV) or (stored.host if stored else DEFAULT_HOST)
    resolved_token = token or source_env.get(_TOKEN_ENV) or (stored.token if stored else None)
    if not resolved_token:
        raise CredentialsError(
            "missing hosted token; run `evalshift login`, "
            "`evalshift login --token es_...`, or set EVALSHIFT_TOKEN",
        )
    return HostedCredentials(host=normalize_host(resolved_host), token=resolved_token)


__all__ = [
    "DEFAULT_HOST",
    "CredentialsError",
    "HostedCredentials",
    "credentials_path",
    "delete_credentials",
    "is_insecure_host",
    "load_credentials",
    "normalize_host",
    "resolve_credentials",
    "resolve_host",
    "save_credentials",
]
