"""Small synchronous HTTP client for the hosted backend."""

from __future__ import annotations

from typing import Any

import httpx

from evalshift.hosted.credentials import normalize_host


class HostedError(Exception):
    """Base error for hosted API calls."""


class HostedNetworkError(HostedError):
    """Raised when the hosted API cannot be reached."""


class HostedHTTPError(HostedError):
    """Raised when the hosted API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, *, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class HostedClient:
    """Typed-ish wrapper around the hosted API endpoints used by the CLI."""

    def __init__(self, *, host: str, token: str, timeout: float = 20.0) -> None:
        self.host = normalize_host(host)
        self.token = token
        self.timeout = timeout

    def me(self) -> dict[str, Any]:
        data = self._request("GET", "/me")
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected /me response")
        return data

    def orgs(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/orgs")
        if not isinstance(data, list):
            raise HostedHTTPError(502, "unexpected /orgs response")
        return [item for item in data if isinstance(item, dict)]

    def list_projects(self, org_slug: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/orgs/{org_slug}/projects")
        if not isinstance(data, list):
            raise HostedHTTPError(502, "unexpected project list response")
        return [item for item in data if isinstance(item, dict)]

    def create_project(
        self,
        org_slug: str,
        *,
        slug: str,
        name: str,
        thresholds: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "slug": slug}
        if thresholds:
            payload["thresholds"] = thresholds
        data = self._request("POST", f"/orgs/{org_slug}/projects", json=payload)
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected project create response")
        return data

    def initiate_run(
        self,
        manifest: dict[str, Any],
        thresholds: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"manifest": manifest}
        if thresholds:
            payload["thresholds"] = thresholds
        data = self._request("POST", "/runs", json=payload)
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected run create response")
        return data

    def finalize_run(self, run_id: str) -> dict[str, Any]:
        data = self._request("POST", f"/runs/{run_id}/finalize")
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected finalize response")
        return data

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.token}"
        headers.setdefault("Accept", "application/json")
        url = f"{self.host}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise HostedNetworkError(
                f"failed to reach hosted EvalShift at {self.host}: {exc}"
            ) from exc
        if response.status_code >= 400:
            code, message = _extract_error(response)
            raise HostedHTTPError(response.status_code, message, code=code)
        if not response.content:
            return None
        return response.json()


def _extract_error(response: httpx.Response) -> tuple[str | None, str]:
    try:
        payload = response.json()
    except ValueError:
        return None, response.text or f"hosted API returned HTTP {response.status_code}"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else None
            message = error.get("message") if isinstance(error.get("message"), str) else None
            return code, message or f"hosted API returned HTTP {response.status_code}"
        detail = payload.get("detail")
        if isinstance(detail, str):
            return None, detail
    return None, f"hosted API returned HTTP {response.status_code}"


__all__ = [
    "HostedClient",
    "HostedError",
    "HostedHTTPError",
    "HostedNetworkError",
]
