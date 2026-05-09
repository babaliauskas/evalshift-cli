"""Tests for :mod:`evalshift.parsers`.

The most important invariant: :class:`PythonStringParser` *never* runs
user code. Every value form that isn't ``ast.Constant(str)`` must be
rejected with a clear, actionable error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift.config.models import PromptDefinition
from evalshift.parsers.base import (
    PromptParseError,
    PromptParser,
    PromptTemplate,
)
from evalshift.parsers.manual import ManualParser
from evalshift.parsers.python_string import PythonStringParser


def _write_py(tmp_path: Path, body: str, name: str = "prompts.py") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_manual_parser_is_a_prompt_parser(self) -> None:
        assert isinstance(ManualParser(), PromptParser)

    def test_python_string_parser_is_a_prompt_parser(self) -> None:
        assert isinstance(PythonStringParser(), PromptParser)


# ---------------------------------------------------------------------------
# ManualParser
# ---------------------------------------------------------------------------


class TestManualParser:
    def test_returns_inline_content(self, tmp_path: Path) -> None:
        prompt = PromptDefinition(
            id="cs",
            detection="manual",
            content="Hello {name}",
            variables=["name"],
        )
        result = ManualParser().parse(prompt, project_root=tmp_path)
        assert isinstance(result, PromptTemplate)
        assert result.id == "cs"
        assert result.content == "Hello {name}"
        assert result.declared_variables == ["name"]

    def test_rejects_python_string_definition(self, tmp_path: Path) -> None:
        prompt = PromptDefinition(
            id="cs",
            detection="python_string",
            path="prompts.py",
            variable="X",
        )
        with pytest.raises(PromptParseError) as info:
            ManualParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "invalid_definition"


# ---------------------------------------------------------------------------
# PythonStringParser — happy paths
# ---------------------------------------------------------------------------


class TestPythonStringHappy:
    def test_simple_single_line_string(self, tmp_path: Path) -> None:
        _write_py(tmp_path, 'GREET = "Hello {name}"\n')
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
            variables=["name"],
        )
        result = PythonStringParser().parse(prompt, project_root=tmp_path)
        assert result.content == "Hello {name}"
        assert result.declared_variables == ["name"]

    def test_triple_quoted_multi_line(self, tmp_path: Path) -> None:
        _write_py(
            tmp_path,
            'GREET = """\nYou are a helper.\nGreet {name}.\n"""\n',
        )
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
            variables=["name"],
        )
        result = PythonStringParser().parse(prompt, project_root=tmp_path)
        assert "Greet {name}" in result.content
        assert result.content.startswith("\n")  # triple-quoted preserves newline

    def test_multiple_assignments_takes_last(self, tmp_path: Path) -> None:
        _write_py(
            tmp_path,
            'GREET = "first"\nGREET = "second"\n',
        )
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
        )
        result = PythonStringParser().parse(prompt, project_root=tmp_path)
        assert result.content == "second"

    def test_ignores_other_module_level_names(self, tmp_path: Path) -> None:
        _write_py(
            tmp_path,
            'OTHER = "ignore me"\nGREET = "the chosen one"\nYET_ANOTHER = "also ignored"\n',
        )
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
        )
        result = PythonStringParser().parse(prompt, project_root=tmp_path)
        assert result.content == "the chosen one"

    def test_absolute_path_used_as_is(self, tmp_path: Path) -> None:
        py_file = _write_py(tmp_path, 'GREET = "hi"\n')
        # project_root is an unrelated directory; absolute path should still load.
        unrelated_root = tmp_path / "unrelated"
        unrelated_root.mkdir()
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path=str(py_file.resolve()),
            variable="GREET",
        )
        result = PythonStringParser().parse(prompt, project_root=unrelated_root)
        assert result.content == "hi"

    def test_relative_path_resolved_against_project_root(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        _write_py(sub, 'GREET = "hi"\n', name="prompts.py")
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="src/prompts.py",
            variable="GREET",
        )
        result = PythonStringParser().parse(prompt, project_root=tmp_path)
        assert result.content == "hi"


# ---------------------------------------------------------------------------
# PythonStringParser — non-literal rejections (the safety surface)
# ---------------------------------------------------------------------------


class TestPythonStringNonLiteral:
    def _try(
        self,
        tmp_path: Path,
        body: str,
        variable: str = "GREET",
    ) -> PromptParseError:
        _write_py(tmp_path, body)
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable=variable,
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        return info.value

    def test_rejects_fstring(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, 'NAME = "world"\nGREET = f"hi {NAME}"\n')
        assert err.kind == "non_literal"
        assert any("f-string" in d.message for d in err.details)

    def test_rejects_concatenation(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, 'GREET = "hello " + "world"\n')
        assert err.kind == "non_literal"
        assert any("concatenation" in d.message for d in err.details)

    def test_rejects_format_method_call(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, 'GREET = "hi {x}".format(x="world")\n')
        assert err.kind == "non_literal"
        assert any(".format()" in d.message for d in err.details)

    def test_rejects_function_call(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, 'GREET = open("foo").read()\n')
        assert err.kind == "non_literal"

    def test_rejects_name_reference(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, 'OTHER = "x"\nGREET = OTHER\n')
        assert err.kind == "non_literal"
        assert any("name reference" in d.message for d in err.details)

    def test_rejects_non_string_constant(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, "GREET = 42\n")
        assert err.kind == "non_literal"
        assert any("non-string constant" in d.message for d in err.details)

    def test_rejects_attribute_access(self, tmp_path: Path) -> None:
        err = self._try(tmp_path, "import os\nGREET = os.linesep\n")
        assert err.kind == "non_literal"


# ---------------------------------------------------------------------------
# PythonStringParser — file/source errors
# ---------------------------------------------------------------------------


class TestPythonStringFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="does_not_exist.py",
            variable="GREET",
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "missing_file"

    def test_directory_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="subdir",
            variable="GREET",
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "not_a_file"

    def test_invalid_python_syntax(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "GREET = (\n")
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "ast_syntax"

    def test_variable_not_found_lists_available_names(self, tmp_path: Path) -> None:
        _write_py(tmp_path, 'OTHER = "x"\nALSO = "y"\n')
        prompt = PromptDefinition(
            id="g",
            detection="python_string",
            path="prompts.py",
            variable="GREET",
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "variable_not_found"
        # Available names should appear in the error so users can fix typos.
        joined = " ".join(d.message for d in info.value.details)
        assert "OTHER" in joined
        assert "ALSO" in joined

    def test_rejects_manual_definition(self, tmp_path: Path) -> None:
        prompt = PromptDefinition(
            id="g",
            detection="manual",
            content="hi",
        )
        with pytest.raises(PromptParseError) as info:
            PythonStringParser().parse(prompt, project_root=tmp_path)
        assert info.value.kind == "invalid_definition"


# ---------------------------------------------------------------------------
# Error formatting smoke
# ---------------------------------------------------------------------------


class TestPromptParseErrorFormatting:
    def test_format_plain(self) -> None:
        err = PromptParseError(
            prompt_id="g",
            kind="non_literal",
            summary="rejected",
            path=Path("prompts.py"),
        )
        assert "g" in err.format_plain()
        assert "prompts.py" in err.format_plain()
