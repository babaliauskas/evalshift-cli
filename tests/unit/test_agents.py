"""Tests for agent-instruction wiring (:mod:`evalshift.cli.commands._agents`).

``init`` writes a standalone ``EVALSHIFT.md`` guide and appends an idempotent,
marker-delimited pointer to it into whatever agent-context files a project
already has — creating ``AGENTS.md`` when none exist. The invariants: the guide
is written, human-authored content is preserved, the reference path is
project-relative (never absolute), and re-running never duplicates the block.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from evalshift.cli.commands._agents import (
    AGENT_CONTEXT_FILES,
    AGENT_INSTRUCTIONS,
    AGENT_INSTRUCTIONS_FILENAME,
    CI_DOCS_URL,
    CLI_DOCS_URL,
    DEFAULT_AGENT_CONTEXT_FILE,
    POINTER_MARKER_BEGIN,
    POINTER_MARKER_END,
    SDK_DOCS_URL,
    wire_agent_instructions,
)

_QUIET = Console(quiet=True)


def _wire(target: Path) -> None:
    wire_agent_instructions(target=target, console=_QUIET)


class TestWritesGuide:
    def test_writes_standalone_guide(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        guide = tmp_path / AGENT_INSTRUCTIONS_FILENAME
        assert guide.is_file()
        body = guide.read_text(encoding="utf-8")
        assert "EvalShift" in body
        assert "evalshift --help" in body
        # Safety guidance must survive into the guide.
        assert "Costs money" in body

    def test_guide_steers_capture_report_push_happy_path(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        body = (tmp_path / AGENT_INSTRUCTIONS_FILENAME).read_text(encoding="utf-8")
        # The backfill + report + push recipe uses commands, not manual edits.
        assert "evalshift capture sync" in body
        assert "evalshift all" in body
        assert "evalshift push" in body

    def test_guide_forbids_editing_generated_state(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        body = (tmp_path / AGENT_INSTRUCTIONS_FILENAME).read_text(encoding="utf-8")
        assert ".evalshift/" in body
        assert "Never" in body  # the hands-off rule for generated state


class TestPythonFloor:
    def test_states_current_floor_not_stale_one(self) -> None:
        # The guide is written verbatim to EVALSHIFT.md, so a stale floor here
        # ships straight to users. Assert on the floor value, not incidental
        # wording, so the next floor change can't silently leave this stale.
        assert "Requires Python 3.11+" in AGENT_INSTRUCTIONS
        assert "3.14" not in AGENT_INSTRUCTIONS


class TestDocsLinks:
    def test_pointer_block_links_llms_docs(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        body = (tmp_path / DEFAULT_AGENT_CONTEXT_FILE).read_text(encoding="utf-8")
        assert CLI_DOCS_URL in body
        assert SDK_DOCS_URL in body
        assert CI_DOCS_URL in body
        # The docs must be scoped: agents fetch them only for EvalShift work.
        assert "Only fetch" in body
        assert "unrelated" in body

    def test_guide_links_llms_docs(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        body = (tmp_path / AGENT_INSTRUCTIONS_FILENAME).read_text(encoding="utf-8")
        assert CLI_DOCS_URL in body
        assert SDK_DOCS_URL in body
        assert CI_DOCS_URL in body


class TestFallbackHost:
    def test_creates_agents_md_when_none_exist(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        host = tmp_path / DEFAULT_AGENT_CONTEXT_FILE
        assert host.is_file()
        body = host.read_text(encoding="utf-8")
        assert POINTER_MARKER_BEGIN in body
        assert POINTER_MARKER_END in body
        assert AGENT_INSTRUCTIONS_FILENAME in body

    def test_no_other_context_files_created(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        for name in AGENT_CONTEXT_FILES:
            if name == DEFAULT_AGENT_CONTEXT_FILE:
                continue
            assert not (tmp_path / name).exists(), f"unexpectedly created {name}"


class TestPreservesExisting:
    def test_appends_to_existing_file_keeping_content(self, tmp_path: Path) -> None:
        agents = tmp_path / "AGENTS.md"
        agents.write_text("# My project rules\n\nUse tabs.\n", encoding="utf-8")
        _wire(tmp_path)
        body = agents.read_text(encoding="utf-8")
        assert "# My project rules" in body
        assert "Use tabs." in body
        assert POINTER_MARKER_BEGIN in body

    def test_claude_md_uses_import_syntax(self, tmp_path: Path) -> None:
        claude = tmp_path / "CLAUDE.md"
        claude.write_text("# Claude rules\n", encoding="utf-8")
        _wire(tmp_path)
        body = claude.read_text(encoding="utf-8")
        assert f"@./{AGENT_INSTRUCTIONS_FILENAME}" in body
        # AGENTS.md must NOT be created — an existing context file was found.
        assert not (tmp_path / "AGENTS.md").exists()

    def test_reference_is_relative_not_absolute(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        body = (tmp_path / DEFAULT_AGENT_CONTEXT_FILE).read_text(encoding="utf-8")
        assert f"./{AGENT_INSTRUCTIONS_FILENAME}" in body
        assert str(tmp_path) not in body


class TestWiresAllExisting:
    def test_all_existing_context_files_get_wired(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("a\n", encoding="utf-8")
        (tmp_path / "GEMINI.md").write_text("g\n", encoding="utf-8")
        copilot = tmp_path / ".github" / "copilot-instructions.md"
        copilot.parent.mkdir(parents=True)
        copilot.write_text("c\n", encoding="utf-8")
        _wire(tmp_path)
        for path in (tmp_path / "AGENTS.md", tmp_path / "GEMINI.md", copilot):
            assert POINTER_MARKER_BEGIN in path.read_text(encoding="utf-8")


class TestIdempotent:
    def test_second_wire_does_not_duplicate_block(self, tmp_path: Path) -> None:
        _wire(tmp_path)
        _wire(tmp_path)
        body = (tmp_path / DEFAULT_AGENT_CONTEXT_FILE).read_text(encoding="utf-8")
        assert body.count(POINTER_MARKER_BEGIN) == 1
        assert body.count(POINTER_MARKER_END) == 1

    def test_rewire_preserves_surrounding_content(self, tmp_path: Path) -> None:
        agents = tmp_path / "AGENTS.md"
        agents.write_text("# Rules\n\nline one\n", encoding="utf-8")
        _wire(tmp_path)
        _wire(tmp_path)
        body = agents.read_text(encoding="utf-8")
        assert "# Rules" in body
        assert "line one" in body
        assert body.count(POINTER_MARKER_BEGIN) == 1
