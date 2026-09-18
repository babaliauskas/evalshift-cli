"""Tests for the ``compare`` rename and its permanent ``all`` alias.

``all`` read as "run all my suites" when the command has always meant "all
pipeline stages, one suite". The command is ``compare`` now; ``all`` stays
registered forever because ``evalshift init`` writes ``EVALSHIFT.md`` into
user repos telling agents to run it, and that file never regenerates.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from evalshift_cli.cli.main import app

runner = CliRunner()


class TestCommandSurface:
    def test_compare_is_listed_in_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout

    def test_all_is_hidden_from_top_level_help(self) -> None:
        # Hidden, not gone: the alias still runs, it just stops being taught.
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Match the command column, not the prose (compare's own summary says
        # "Run all pipeline stages").
        rows = re.findall(r"^\s*\u2502 (\S+)", result.stdout, re.MULTILINE)
        assert "compare" in rows
        assert "all" not in rows

    def test_epilog_teaches_compare_rather_than_all(self) -> None:
        result = runner.invoke(app, ["--help"])
        flat = " ".join(result.stdout.split())  # undo terminal wrapping
        assert "'evalshift compare' runs it end to end" in flat
        assert "evalshift all" not in flat


class TestBothNamesResolve:
    def test_compare_help_describes_one_suite(self) -> None:
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        flat = " ".join(result.stdout.split())
        assert "Compare two models on one suite" in flat

    def test_compare_help_records_the_former_name(self) -> None:
        flat = " ".join(runner.invoke(app, ["compare", "--help"]).stdout.split())
        assert "Formerly named 'all'" in flat

    def test_all_help_still_works(self) -> None:
        result = runner.invoke(app, ["all", "--help"])
        assert result.exit_code == 0

    def test_both_names_expose_the_same_options(self) -> None:
        compare = runner.invoke(app, ["compare", "--help"]).stdout
        legacy = runner.invoke(app, ["all", "--help"]).stdout
        for flag in ("--suite-name", "--policy-gate", "--push", "--gate"):
            assert flag in compare
            assert flag in legacy


class TestDeprecationNotice:
    """The notice goes to stderr so CI parsing stdout is unaffected."""

    def _invoke(self, name: str) -> tuple[str, str]:
        # A missing config fails the command early; the notice is printed first.
        result = CliRunner().invoke(app, [name, "--config", "does-not-exist.yaml"])
        return result.stdout, result.stderr

    def test_all_points_at_its_replacement(self) -> None:
        _, err = self._invoke("all")
        assert "evalshift compare" in " ".join(err.split())

    def test_all_says_the_old_name_still_works(self) -> None:
        _, err = self._invoke("all")
        assert "keeps working" in " ".join(err.split())

    def test_compare_is_silent(self) -> None:
        _, err = self._invoke("compare")
        assert "is now" not in err

    def test_notice_stays_off_stdout(self) -> None:
        out, _ = self._invoke("all")
        assert "is now" not in out

    def test_click_adds_no_second_deprecation_line(self) -> None:
        # deprecated=True would print its own vaguer warning above ours.
        _, err = self._invoke("all")
        assert "DeprecationWarning" not in err
