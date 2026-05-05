"""Tests for ``evalshift doctor`` (:mod:`evalshift.cli.commands.doctor`).

Two layers of testing:

* Unit-level: :func:`run_checks` is a pure function of ``cwd`` and ``env``,
  so most behaviour is asserted there with direct calls — no CLI plumbing.
* CLI-level: a couple of :class:`CliRunner` invocations confirm the exit
  codes and that the rendered table reaches stdout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.commands.doctor import (
    CONFIG_FILENAME,
    PROVIDER_KEYS,
    CheckResult,
    run_checks,
)
from evalshift.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_name(results: list[CheckResult], name: str) -> CheckResult:
    matching = [r for r in results if r.name == name]
    assert len(matching) == 1, f"expected one row named {name!r}, got {matching}"
    return matching[0]


def _empty_env() -> Mapping[str, str]:
    return {}


# ---------------------------------------------------------------------------
# run_checks — pure function tests
# ---------------------------------------------------------------------------


class TestRunChecksPython:
    def test_python_row_is_always_ok(self, tmp_path: Path) -> None:
        results = run_checks(cwd=tmp_path, env=_empty_env())
        py = results[0]
        assert py.status == "ok"
        assert py.name.startswith("Python ")


class TestRunChecksAPIKeys:
    def test_missing_keys_warn(self, tmp_path: Path) -> None:
        results = run_checks(cwd=tmp_path, env=_empty_env())
        for key in PROVIDER_KEYS:
            row = _by_name(results, key)
            assert row.status == "warn"
            assert "not set" in row.detail

    def test_set_keys_ok(self, tmp_path: Path) -> None:
        env = dict.fromkeys(PROVIDER_KEYS, "sk-test")
        results = run_checks(cwd=tmp_path, env=env)
        for key in PROVIDER_KEYS:
            assert _by_name(results, key).status == "ok"

    def test_partial_keys(self, tmp_path: Path) -> None:
        env = {"ANTHROPIC_API_KEY": "x"}
        results = run_checks(cwd=tmp_path, env=env)
        assert _by_name(results, "ANTHROPIC_API_KEY").status == "ok"
        assert _by_name(results, "OPENAI_API_KEY").status == "warn"
        assert _by_name(results, "GOOGLE_API_KEY").status == "warn"


class TestRunChecksConfig:
    def test_missing_config_warns_not_fails(self, tmp_path: Path) -> None:
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "warn"
        assert "not found" in row.detail
        assert "evalshift init" in row.detail

    def test_valid_config_ok(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "ok"
        assert "1 prompt" in row.detail and "prompts" not in row.detail.replace("prompt", "")

    def test_valid_config_pluralizes(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            """
            prompts:
              - {id: a, detection: manual, content: hi}
              - {id: b, detection: manual, content: hello}
            """,
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert "2 prompts" in row.detail

    def test_invalid_config_fails(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts: []\n",  # empty prompts list — schema rejects
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "fail"

    def test_unparseable_yaml_fails(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - id: a\n  detection: manual\n   content: bad-indent\n",
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "fail"


# ---------------------------------------------------------------------------
# CLI-level
# ---------------------------------------------------------------------------


class TestDoctorCLI:
    def _isolate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Force doctor to run inside ``tmp_path`` with no env vars set."""
        monkeypatch.chdir(tmp_path)
        for key in PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_doctor_with_no_config_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Both glyphs should appear: ✓ for Python, ✗ for missing keys/config.
        assert "✓" in result.stdout
        assert "✗" in result.stdout

    def test_doctor_with_valid_config_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "evalshift.yaml" in result.stdout
        assert "1 prompt" in result.stdout

    def test_doctor_with_invalid_config_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / CONFIG_FILENAME).write_text("prompts: []\n", encoding="utf-8")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1

    def test_doctor_with_set_keys_renders_them_ok(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "ANTHROPIC_API_KEY" in result.stdout
        assert "set" in result.stdout
