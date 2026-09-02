"""Tests for :mod:`evalshift.utils.ci_pin` — the CI pin-drift check.

The check is advisory and reads ``.github/workflows/*.yml`` next to the
project's ``evalshift.yaml``; it must never raise on odd input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift.utils.ci_pin import (
    ActionPin,
    CiPinFinding,
    check_ci_pin,
    find_action_pins,
)

WORKFLOWS = Path(".github/workflows")


def _workflow(root: Path, name: str, body: str) -> Path:
    path = root / WORKFLOWS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _action_job(job: str, *, with_lines: str = "") -> str:
    return (
        f"  {job}:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n"
        "      - uses: babaliauskas/evalshift-action@v0\n"
        "        with:\n"
        '          token: "${{ secrets.EVALSHIFT_TOKEN }}"\n' + with_lines
    )


def _pinned(version: str) -> str:
    return f'          evalshift-version: "{version}"\n'


def _finding(root: Path, cli_version: str = "0.13.1") -> CiPinFinding:
    finding = check_ci_pin(root, cli_version)
    assert finding is not None
    return finding


# ---------------------------------------------------------------------------
# find_action_pins
# ---------------------------------------------------------------------------


class TestFindActionPins:
    def test_no_workflows_dir(self, tmp_path: Path) -> None:
        assert find_action_pins(tmp_path) == []

    def test_workflow_without_the_action(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "ci.yml",
            "on: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - uses: actions/checkout@v7\n      - run: pytest\n",
        )
        assert find_action_pins(tmp_path) == []

    def test_literal_pin(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.12.1")),
        )
        assert find_action_pins(tmp_path) == [
            ActionPin(
                workflow=WORKFLOWS / "evalshift.yml",
                job="evalshift",
                version="0.12.1",
                literal=True,
            )
        ]

    def test_absent_input_is_none(self, tmp_path: Path) -> None:
        _workflow(tmp_path, "evalshift.yml", "on: push\njobs:\n" + _action_job("evalshift"))
        [pin] = find_action_pins(tmp_path)
        assert pin.version is None
        assert pin.literal is True

    def test_expression_is_not_literal(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n"
            + _action_job(
                "evalshift",
                with_lines='          evalshift-version: "${{ vars.EVALSHIFT_VERSION }}"\n',
            ),
        )
        [pin] = find_action_pins(tmp_path)
        assert pin.literal is False
        assert pin.version == "${{ vars.EVALSHIFT_VERSION }}"

    def test_two_jobs_with_different_pins_in_two_workflows(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "a.yml",
            "on: push\njobs:\n" + _action_job("one", with_lines=_pinned("0.12.1")),
        )
        _workflow(
            tmp_path,
            "b.yaml",
            "on: push\njobs:\n" + _action_job("two", with_lines=_pinned("0.13.1")),
        )
        pins = find_action_pins(tmp_path)
        assert [(str(p.workflow), p.job, p.version) for p in pins] == [
            (".github/workflows/a.yml", "one", "0.12.1"),
            (".github/workflows/b.yaml", "two", "0.13.1"),
        ]

    def test_invalid_yaml_is_skipped_silently(self, tmp_path: Path) -> None:
        _workflow(tmp_path, "broken.yml", "jobs: [unclosed\n")
        _workflow(tmp_path, "scalar.yml", "just a string\n")
        _workflow(
            tmp_path,
            "ok.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.13.1")),
        )
        assert [p.job for p in find_action_pins(tmp_path)] == ["evalshift"]

    def test_non_yaml_files_are_ignored(self, tmp_path: Path) -> None:
        _workflow(tmp_path, "README.md", "uses: babaliauskas/evalshift-action@v0\n")
        assert find_action_pins(tmp_path) == []


# ---------------------------------------------------------------------------
# check_ci_pin
# ---------------------------------------------------------------------------


class TestCheckCiPin:
    def test_no_action_steps_is_silent(self, tmp_path: Path) -> None:
        assert check_ci_pin(tmp_path, "0.13.1") is None

    def test_stale_pin(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.12.1")),
        )
        finding = _finding(tmp_path)
        assert finding.status == "stale"
        assert "CI installs evalshift 0.12.1" in finding.message
        assert ".github/workflows/evalshift.yml, job evalshift" in finding.message
        assert "0.13.1" in finding.message
        assert 'Fix: set `evalshift-version: "0.13.1"`' in finding.message

    def test_unpinned(self, tmp_path: Path) -> None:
        _workflow(tmp_path, "evalshift.yml", "on: push\njobs:\n" + _action_job("evalshift"))
        finding = _finding(tmp_path)
        assert finding.status == "unpinned"
        assert "default" in finding.message
        assert 'Fix: add `evalshift-version: "0.13.1"`' in finding.message

    def test_ahead(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.14.0")),
        )
        finding = _finding(tmp_path)
        assert finding.status == "ahead"
        assert "pip install -U evalshift" in finding.message

    def test_equal_is_silent(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.13.1")),
        )
        assert check_ci_pin(tmp_path, "0.13.1") is None

    def test_expression_only_is_silent(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n"
            + _action_job(
                "evalshift",
                with_lines='          evalshift-version: "${{ vars.V }}"\n',
            ),
        )
        assert check_ci_pin(tmp_path, "0.13.1") is None

    def test_dev_version_is_silent(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.12.1")),
        )
        assert check_ci_pin(tmp_path, "0.0.0+unknown") is None

    def test_stale_wins_over_unpinned_and_lists_each_pin(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "a.yml",
            "on: push\njobs:\n" + _action_job("old", with_lines=_pinned("0.12.1")),
        )
        _workflow(tmp_path, "b.yml", "on: push\njobs:\n" + _action_job("bare"))
        finding = _finding(tmp_path)
        assert finding.status == "stale"
        assert [p.job for p in finding.pins] == ["old"]

    def test_unpinned_wins_over_ahead(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "a.yml",
            "on: push\njobs:\n" + _action_job("newer", with_lines=_pinned("0.14.0")),
        )
        _workflow(tmp_path, "b.yml", "on: push\njobs:\n" + _action_job("bare"))
        assert _finding(tmp_path).status == "unpinned"

    def test_mixed_equal_and_ahead_is_silent(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "a.yml",
            "on: push\njobs:\n" + _action_job("same", with_lines=_pinned("0.13.1")),
        )
        _workflow(
            tmp_path,
            "b.yml",
            "on: push\njobs:\n" + _action_job("newer", with_lines=_pinned("0.14.0")),
        )
        assert check_ci_pin(tmp_path, "0.13.1") is None

    def test_one_message_line_per_distinct_workflow_and_pin(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "a.yml",
            "on: push\njobs:\n"
            + _action_job("one", with_lines=_pinned("0.12.1"))
            + _action_job("two", with_lines=_pinned("0.12.1"))
            + _action_job("three", with_lines=_pinned("0.11.0")),
        )
        finding = _finding(tmp_path)
        assert finding.status == "stale"
        assert finding.message.count("CI installs evalshift") == 2
        assert "jobs one, two" in finding.message
        assert "job three" in finding.message

    @pytest.mark.parametrize("bad", ["latest", "not-a-version"])
    def test_unparseable_literal_never_raises(self, tmp_path: Path, bad: str) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned(bad)),
        )
        assert check_ci_pin(tmp_path, "0.13.1") is None

    def test_unparseable_cli_version_never_raises(self, tmp_path: Path) -> None:
        _workflow(
            tmp_path,
            "evalshift.yml",
            "on: push\njobs:\n" + _action_job("evalshift", with_lines=_pinned("0.12.1")),
        )
        assert check_ci_pin(tmp_path, "garbage") is None
