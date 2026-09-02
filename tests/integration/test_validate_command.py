"""Integration tests for ``evalshift validate``.

Each test runs the command inside a fixture project directory under
``tests/integration/fixtures/`` and asserts the exit code + key fragments
of the rendered output. No LLM calls happen here — Phase 2 is purely
local validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.cli.main import app

runner = CliRunner()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _invoke_in_fixture(
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
) -> tuple[int, str]:
    """Run ``evalshift validate`` inside the named fixture directory."""
    monkeypatch.chdir(FIXTURES_DIR / fixture_name)
    result = runner.invoke(app, ["validate"])
    return result.exit_code, result.stdout


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidateHappy:
    def test_ok_fixture_exits_zero_with_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_ok")
        assert code == 0, output
        assert "Loaded 1 prompt" in output
        assert "2 examples" in output
        assert "compatible" in output

    def test_validate_is_hidden_from_help(self) -> None:
        # `validate` is registered with `hidden=True`; it should run, but
        # not appear in the top-level --help text.
        result = runner.invoke(app, ["--help"])
        assert "validate" not in result.stdout


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestValidateFailures:
    def test_missing_template_variable_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_missing_var")
        assert code == 1, output
        # Error must point at the offending prompt + example + variable.
        assert "greet" in output
        assert "ex2" in output
        assert "tone" in output

    def test_non_literal_prompt_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_non_literal")
        assert code == 1, output
        # Error should mention the offending value form.
        assert "f-string" in output or "non_literal" in output.lower()

    def test_missing_config_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)  # empty dir — no evalshift.yaml
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "Invalid config" in result.stdout

    def test_missing_suite_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Valid config but no suite file in the cwd.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "Invalid suite" in result.stdout


# ---------------------------------------------------------------------------
# CI pin drift (advisory — never changes the exit code)
# ---------------------------------------------------------------------------


class TestValidateCiPin:
    def test_warns_after_the_success_line_when_ci_pins_an_older_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import shutil

        shutil.copytree(FIXTURES_DIR / "validate_ok", tmp_path, dirs_exist_ok=True)
        workflow = tmp_path / ".github" / "workflows" / "evalshift.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(
            "on: push\njobs:\n  evalshift:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: babaliauskas/evalshift-action@v0\n"
            '        with:\n          evalshift-version: "0.0.1"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("evalshift.cli.commands.validate.__version__", "1.2.3")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, result.stdout
        assert result.stdout.index("compatible") < result.stdout.index(
            "CI installs evalshift 0.0.1"
        )
