"""Phase 5 hosted CLI tests."""

from __future__ import annotations

import gzip
import io
import json
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from evalshift.cli.main import app
from evalshift.evaluators.base import EvalRecord
from evalshift.hosted.bundle import BUNDLE_FILENAME, BundleError, build_bundle
from evalshift.hosted.credentials import (
    CredentialsError,
    load_credentials,
    resolve_credentials,
    save_credentials,
)
from evalshift.hosted.push import HostedHTTPError, PushError, _put_with_retries, push_bundle
from evalshift.runner.checkpoint import append_call, write_state
from evalshift.runner.models import Call, RunModels, RunState

runner = CliRunner()


def _write_project_files(root: Path) -> None:
    (root / "evalshift.yaml").write_text(
        """
        version: 1
        project: acme/model-migration
        thresholds:
          pass_rate_min: 0.9
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
        """,
        encoding="utf-8",
    )
    (root / "golden.jsonl").write_text(
        json.dumps({"id": "ex1", "inputs": {"name": "Ada"}, "tags": ["security"]}) + "\n",
        encoding="utf-8",
    )


def _write_completed_run(root: Path, run_id: str = "r_20260516_abcdef") -> Path:
    runs_base = root / ".evalshift" / "runs"
    run_dir = runs_base / run_id
    state = RunState(
        run_id=run_id,
        status="completed",
        config_hash="local-config-hash",
        started_at=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        last_checkpoint_at=datetime(2026, 5, 16, 12, 1, 14, tzinfo=UTC),
        models=RunModels(
            source="gemini/gemini-2.5-flash", target="gemini/gemini-3.1-flash-lite-preview"
        ),
        prompt_ids=["greet"],
        suite_path=str(root / "golden.jsonl"),
        total_evaluations=2,
        completed_evaluations=2,
    )
    write_state(run_dir, state)
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            model_id="gemini/gemini-2.5-flash",
            role="source",
            text="hello ada",
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.01,
            latency_ms=100,
        ),
    )
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            model_id="gemini/gemini-3.1-flash-lite-preview",
            role="target",
            text="hello ada!",
            input_tokens=10,
            output_tokens=3,
            cost_usd=0.02,
            latency_ms=110,
        ),
    )
    (run_dir / "scores.jsonl").write_text(
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            evaluator_name="length",
            source_score=1.0,
            target_score=1.0,
            delta=0.0,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "analysis.json").write_text('{"comparisons": []}', encoding="utf-8")
    (run_dir / "report.html").write_text(
        "<!doctype html><title>EvalShift</title>", encoding="utf-8"
    )
    return run_dir


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
    return data


def test_credentials_file_uses_0600_and_resolution_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "credentials"

    save_credentials("https://api.evalshift.test/", "es_secret", path=path)

    mode = stat.S_IMODE(path.stat().st_mode)
    stored = load_credentials(path=path)
    assert stored is not None
    assert mode == 0o600
    assert stored.host == "https://api.evalshift.test"
    assert stored.token == "es_secret"
    monkeypatch.setenv("EVALSHIFT_HOST", "https://env.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_env")
    assert resolve_credentials(path=path).token == "es_env"
    assert resolve_credentials(host="https://flag.test", token="es_flag", path=path).host == (
        "https://flag.test"
    )


def test_resolve_credentials_requires_token_when_none_available(tmp_path: Path) -> None:
    with pytest.raises(CredentialsError, match="evalshift login"):
        resolve_credentials(path=tmp_path / "missing")


def test_login_writes_credentials_without_printing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    monkeypatch.setattr("evalshift.cli.commands.login.HostedClient.me", fake_me)

    result = runner.invoke(
        app,
        ["login", "--token", "es_plaintext", "--host", "https://api.evalshift.test"],
    )

    assert result.exit_code == 0, result.output
    assert "dev@example.com" in result.output
    assert "es_plaintext" not in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_plaintext"


def test_whoami_prints_user_and_visible_org_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    def fake_orgs(self: Any) -> list[dict[str, Any]]:
        return [{"slug": "acme", "name": "Acme", "role": "owner"}]

    monkeypatch.setattr("evalshift.cli.commands.whoami.HostedClient.me", fake_me)
    monkeypatch.setattr("evalshift.cli.commands.whoami.HostedClient.orgs", fake_orgs)

    result = runner.invoke(app, ["whoami"])

    assert result.exit_code == 0, result.output
    assert "https://api.evalshift.test" in result.output
    assert "dev@example.com" in result.output
    assert "acme" in result.output
    assert "owner" in result.output


def test_bundle_builder_writes_valid_gzip_bundle_with_deterministic_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")

    first = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    second = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
        output=tmp_path / "second_bundle.json.gz",
    )

    assert first.path.name == BUNDLE_FILENAME
    data = _read_bundle(first.path)
    again = _read_bundle(second.path)
    assert data["manifest"]["project_slug"] == "acme/model-migration"
    assert data["manifest"]["git_sha"] == "a" * 40
    assert data["manifest"]["branch"] == "feature/model-swap"
    assert data["manifest"]["size_bytes"] == first.path.stat().st_size
    assert data["manifest"]["eval_config_hash"] == again["manifest"]["eval_config_hash"]
    assert data["manifest"]["dataset_hash"] == again["manifest"]["dataset_hash"]
    assert data["examples"][0]["passed"] is True
    assert data["aggregate"]["total"] == 1


def test_bundle_builder_reports_missing_artifacts(tmp_path: Path) -> None:
    _write_project_files(tmp_path)
    run_dir = _write_completed_run(tmp_path)
    (run_dir / "report.html").unlink()

    with pytest.raises(BundleError, match=r"report\.html"):
        build_bundle(
            "r_20260516_abcdef",
            config_path=tmp_path / "evalshift.yaml",
            suite_path=tmp_path / "golden.jsonl",
            runs_base=tmp_path / ".evalshift" / "runs",
        )


@dataclass
class _FakeHostedClient:
    responses: list[dict[str, Any]]
    projects: list[dict[str, Any]] | None = None
    created_project: dict[str, Any] | None = None

    def initiate_run(
        self, manifest: dict[str, Any], thresholds: dict[str, Any] | None
    ) -> dict[str, Any]:
        response = self.responses.pop(0)
        if response.get("raise_404"):
            raise HostedHTTPError(404, "project not found", code="not_found")
        return response

    def finalize_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "view_url": "https://app.test/runs/" + run_id}

    def list_projects(self, org_slug: str) -> list[dict[str, Any]]:
        return list(self.projects or [])

    def create_project(
        self,
        org_slug: str,
        *,
        slug: str,
        name: str,
        thresholds: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.created_project = {
            "org_slug": org_slug,
            "slug": slug,
            "name": name,
            "thresholds": thresholds,
        }
        return self.created_project


def _build_bundle_for_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")
    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    return result.path


def test_push_treats_available_run_as_idempotent_without_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _FakeHostedClient(
        responses=[
            {
                "run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "view_url": "https://app.test/runs/r_20260516_abcdef",
                "canonical_thresholds": {"pass_rate_min": 0.9},
            },
        ],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)

    result = push_bundle(
        bundle_path,
        config_path=tmp_path / "evalshift.yaml",
        create_project=False,
    )

    assert result.uploaded is False
    assert result.view_url == "https://app.test/runs/r_20260516_abcdef"


def test_put_retries_transient_failures_only() -> None:
    attempts = 0

    def fake_put(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    _put_with_retries(
        "https://storage.test/upload",
        b"bundle",
        put=fake_put,
        sleep=lambda _: None,
    )

    assert attempts == 3


def test_push_auto_creates_missing_project_when_org_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _FakeHostedClient(
        responses=[
            {"raise_404": True},
            {
                "run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "view_url": "https://app.test/runs/r_20260516_abcdef",
                "canonical_thresholds": {"pass_rate_min": 0.9},
            },
        ],
        projects=[],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)

    result = push_bundle(bundle_path, config_path=tmp_path / "evalshift.yaml", create_project=True)

    assert result.project_created is True
    assert fake.created_project == {
        "org_slug": "acme",
        "slug": "model-migration",
        "name": "Model Migration",
        "thresholds": {"pass_rate_min": 0.9},
    }


def test_push_does_not_auto_create_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _FakeHostedClient(responses=[{"raise_404": True}], projects=[])
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError, match="project was not found"):
        push_bundle(bundle_path, config_path=tmp_path / "evalshift.yaml", create_project=False)


def test_push_warns_when_canonical_thresholds_differ_from_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.console import Console

    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _FakeHostedClient(
        responses=[
            {
                "run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "view_url": "https://app.test/runs/r_20260516_abcdef",
                "canonical_thresholds": {"pass_rate_min": 0.95, "regression_max": 0.0},
            },
        ],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift.hosted.push.HostedClient", lambda **_: fake)

    buffer = io.StringIO()
    push_bundle(
        bundle_path,
        config_path=tmp_path / "evalshift.yaml",
        create_project=False,
        console=Console(file=buffer, force_terminal=False, width=120),
    )

    output = buffer.getvalue()
    assert "differ from project canonical thresholds" in output
    assert "pass_rate_min" in output
    assert "regression_max" in output


def test_put_retries_connection_errors_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fake_put(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection refused")

    with pytest.raises(PushError, match="upload failed after 3 attempts"):
        _put_with_retries(
            "https://storage.test/upload",
            b"bundle",
            put=fake_put,
            sleep=lambda _: None,
        )
    assert attempts == 3


def test_bundle_builder_rejects_invalid_bundle_via_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a manifest field to a value the schema rejects and assert BundleError."""
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")  # not a 40-char hex sha
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")

    with pytest.raises(BundleError):
        build_bundle(
            "r_20260516_abcdef",
            config_path=tmp_path / "evalshift.yaml",
            suite_path=tmp_path / "golden.jsonl",
            runs_base=tmp_path / ".evalshift" / "runs",
        )


def test_login_warns_on_plain_http_to_non_local_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(tmp_path / "credentials"))

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    monkeypatch.setattr("evalshift.cli.commands.login.HostedClient.me", fake_me)

    result = runner.invoke(
        app,
        ["login", "--token", "es_plaintext", "--host", "http://prod.evalshift.example.com"],
    )

    assert result.exit_code == 0, result.output
    assert "cleartext" in result.output
    assert "es_plaintext" not in result.output
