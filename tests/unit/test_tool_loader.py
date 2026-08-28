"""Tests for :mod:`evalshift.evaluators.tool_loader`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalshift.evaluators.tool_loader import ToolLoaderError, load_tools


def _write(tmp_path: Path, body: str, name: str = "tools.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


class TestLoadToolsHappyPath:
    def test_yaml_list(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            - name: search_db
              description: Search the customer DB.
              input_schema:
                type: object
                properties:
                  query: {type: string}
            """,
        )
        tools = load_tools(path)
        assert len(tools) == 1
        assert tools[0].name == "search_db"

    def test_yaml_with_top_level_tools_key(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            tools:
              - name: a
                description: A
              - name: b
                description: B
            """,
        )
        assert [t.name for t in load_tools(path)] == ["a", "b"]

    def test_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "tools.json"
        path.write_text(
            json.dumps(
                [
                    {"name": "a", "description": "A", "input_schema": {}},
                ],
            ),
        )
        assert load_tools(path)[0].name == "a"

    def test_openai_shape_accepted(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            - type: function
              function:
                name: a
                description: A
                parameters: {type: object}
            """,
        )
        tools = load_tools(path)
        assert tools[0].name == "a"


class TestLoadToolsErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ToolLoaderError) as info:
            load_tools(tmp_path / "nope.yaml")
        assert info.value.kind == "missing"

    def test_directory_path(self, tmp_path: Path) -> None:
        with pytest.raises(ToolLoaderError) as info:
            load_tools(tmp_path)
        assert info.value.kind == "not_a_file"

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "not: [a, b\n")
        with pytest.raises(ToolLoaderError) as info:
            load_tools(path)
        assert info.value.kind == "parse"

    def test_top_level_not_list_or_mapping(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "scalar value")
        with pytest.raises(ToolLoaderError) as info:
            load_tools(path)
        assert info.value.kind == "wrong_shape"

    def test_empty_list(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "[]")
        with pytest.raises(ToolLoaderError) as info:
            load_tools(path)
        assert info.value.kind == "empty"

    def test_invalid_entry_is_collected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            - name: ok
              description: yes
            - foo: bar
            """,
        )
        with pytest.raises(ToolLoaderError) as info:
            load_tools(path)
        assert info.value.kind == "invalid_tool"
        assert any("tools[1]" in d.location for d in info.value.details)


class TestErrorFormatting:
    def test_format_plain_includes_path_and_details(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "[]")
        try:
            load_tools(path)
        except ToolLoaderError as err:
            text = err.format_plain()
            assert str(path) in text
            assert "empty" in text
        else:
            pytest.fail("expected ToolLoaderError")

    def test_format_rich_renders(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "[]")
        try:
            load_tools(path)
        except ToolLoaderError as err:
            rendered = err.format_rich()
            assert rendered is not None
        else:
            pytest.fail("expected ToolLoaderError")
