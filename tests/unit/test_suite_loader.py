"""Tests for :mod:`evalshift.suite.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from evalshift.suite.loader import (
    SuiteError,
    SuiteErrorDetail,
    _format_loc_suffix,
    load_jsonl,
)
from evalshift.suite.models import Suite


def _write(tmp_path: Path, body: str, name: str = "golden.jsonl") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadJsonlHappy:
    def test_minimal_well_formed_file(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "a", "inputs": {"n": 1}, "tags": ["t"], "tools": []}\n'
            '{"id": "b", "inputs": {"n": 2}, "tools": []}\n',
        )
        suite = load_jsonl(path)
        assert isinstance(suite, Suite)
        assert len(suite) == 2
        assert suite.ids() == {"a", "b"}

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = _write(tmp_path, '{"id": "a", "tools": []}\n')
        suite = load_jsonl(str(path))
        assert suite.ids() == {"a"}

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "\n"
            '{"id": "a", "tools": []}\n'
            "   \n"  # whitespace-only line
            '{"id": "b", "tools": []}\n'
            "\n",
        )
        suite = load_jsonl(path)
        assert suite.ids() == {"a", "b"}

    def test_round_trip_preserves_inputs_and_tags(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "a", "inputs": {"k": [1, 2]}, "tags": ["x", "y"], "tools": []}\n',
        )
        suite = load_jsonl(path)
        assert suite.examples[0].inputs == {"k": [1, 2]}
        assert suite.examples[0].tags == ["x", "y"]


# ---------------------------------------------------------------------------
# File-system failures
# ---------------------------------------------------------------------------


class TestLoadJsonlFileSystem:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError) as info:
            load_jsonl(tmp_path / "does-not-exist.jsonl")
        assert info.value.kind == "missing"

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SuiteError) as info:
            load_jsonl(tmp_path)
        assert info.value.kind == "not_a_file"

    def test_truly_empty_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "")
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "empty"

    def test_only_blank_lines(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "\n   \n\t\n")
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "empty"


# ---------------------------------------------------------------------------
# JSON parse failures
# ---------------------------------------------------------------------------


class TestLoadJsonlParseErrors:
    def test_malformed_line_reports_line_number(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "ok"}\nnot json\n{"id": "still_ok"}\n',
        )
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "json_parse"
        assert any("line 2" in d.location for d in info.value.details)

    def test_multiple_parse_errors_collected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{ broken\n{"id": "a"}\n} also broken\n',
        )
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "json_parse"
        assert len(info.value.details) == 2


# ---------------------------------------------------------------------------
# Schema failures
# ---------------------------------------------------------------------------


class TestLoadJsonlSchemaErrors:
    def test_missing_id_reports_field(self, tmp_path: Path) -> None:
        path = _write(tmp_path, '{"inputs": {"n": 1}}\n')
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "schema"
        assert any(": id" in d.location for d in info.value.details)

    def test_extra_keys_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "a", "rogue": true}\n',
        )
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "schema"
        assert any("rogue" in d.location for d in info.value.details)

    def test_empty_id_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, '{"id": ""}\n')
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "schema"

    def test_schema_error_locations_include_id_when_known(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "good_id_but_bad_field", "tags": "should_be_a_list"}\n',
        )
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        # The detail location should mention the row id for fast triage.
        assert any("good_id_but_bad_field" in d.location for d in info.value.details)


# ---------------------------------------------------------------------------
# Duplicate ids
# ---------------------------------------------------------------------------


class TestLoadJsonlDuplicateIds:
    def test_duplicate_ids_dedicated_kind(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            '{"id": "a", "tools": []}\n{"id": "b", "tools": []}\n{"id": "a", "tools": []}\n',
        )
        with pytest.raises(SuiteError) as info:
            load_jsonl(path)
        assert info.value.kind == "duplicate_ids"


# ---------------------------------------------------------------------------
# SuiteError formatting
# ---------------------------------------------------------------------------


class TestSuiteErrorFormatting:
    def test_format_plain_includes_path_and_details(self, tmp_path: Path) -> None:
        err = SuiteError(
            path=tmp_path / "golden.jsonl",
            kind="schema",
            summary="2 schema problems found",
            details=[
                SuiteErrorDetail("line 5: id", "field required"),
                SuiteErrorDetail("line 6", "unexpected key 'rogue'"),
            ],
        )
        text = err.format_plain()
        assert "golden.jsonl" in text
        assert "line 5" in text
        assert "line 6" in text

    def test_format_rich_renders(self) -> None:
        # Use a short path so the Rich panel title doesn't truncate; we
        # care that the filename + content reach the rendered output, not
        # whether the title fits an arbitrary terminal width.
        err = SuiteError(
            path=Path("golden.jsonl"),
            kind="schema",
            summary="1 schema problem found",
            details=[SuiteErrorDetail("line 3", "Field required")],
        )
        console = Console(record=True, width=120)
        console.print(err.format_rich())
        rendered = console.export_text()
        assert "Invalid suite" in rendered
        assert "golden.jsonl" in rendered
        assert "line 3" in rendered

    def test_str_uses_plain_format(self, tmp_path: Path) -> None:
        err = SuiteError(
            path=tmp_path / "x.jsonl",
            kind="missing",
            summary="file not found: x.jsonl",
        )
        assert "file not found" in str(err)


# ---------------------------------------------------------------------------
# _format_loc_suffix
# ---------------------------------------------------------------------------


class TestFormatLocSuffix:
    @pytest.mark.parametrize(
        ("loc", "expected"),
        [
            ((), ""),
            (("id",), ": id"),
            (("inputs", "n"), ": inputs.n"),
            (("tags", 0), ": tags[0]"),
        ],
    )
    def test_format(self, loc: tuple[int | str, ...], expected: str) -> None:
        assert _format_loc_suffix(loc) == expected


# ---------------------------------------------------------------------------
# Checked-in example suites
# ---------------------------------------------------------------------------

# Repo root: tests/unit/test_suite_loader.py -> tests/unit -> tests -> <root>.
# Deliberately a direct path to <root>/examples, never a broad rglob from the
# repo root, so this can't wander into .claude/worktrees/ (a stale full copy
# of the repo that must never be treated as source of truth).
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


class TestLoadJsonlCheckedInExamples:
    """Every ``examples/**/golden.jsonl`` shipped in this repo must load.

    These files are documented, user-facing content (linked from README.md,
    docs/agents.md, and DOCS.md), not exercised by any other test. A
    structural change to ``SuiteExample`` (e.g. the toolset_ref/tools
    exactly-one-of requirement) can silently break every one of them while
    the rest of CI stays green -- this test exists so that class of breakage
    cannot be silent again.
    """

    def test_every_checked_in_golden_jsonl_loads(self) -> None:
        golden_files = sorted(_EXAMPLES_DIR.glob("**/golden.jsonl"))
        # Guard the guard: if this ever finds nothing, the glob/path is
        # broken, not the examples -- fail loudly instead of vacuously
        # passing.
        assert len(golden_files) >= 3, (
            f"expected at least 3 example golden.jsonl files under {_EXAMPLES_DIR}, "
            f"found {len(golden_files)}: {golden_files}"
        )

        failures: list[str] = []
        for path in golden_files:
            try:
                suite = load_jsonl(path)
            except SuiteError as exc:
                failures.append(f"{path.relative_to(_EXAMPLES_DIR.parent)}: {exc.summary}")
            else:
                if len(suite) == 0:
                    failures.append(f"{path.relative_to(_EXAMPLES_DIR.parent)}: loaded 0 examples")

        assert not failures, "example suite(s) failed to load:\n" + "\n".join(failures)
