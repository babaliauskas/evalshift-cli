"""Parser for ``detection: python_string`` prompts.

Reads a ``.py`` file, AST-walks the module body, and extracts a single
string literal assigned to the configured variable name. **Never executes
the module** — string literals only. F-strings, concatenations, ``.format()``
calls, function results, attribute access, and any other dynamic value
form are explicitly rejected.

Why so strict? Two reasons:

1. **Safety** — running arbitrary user code at parse time is a foot-gun;
   we'd be loading prompts from untrusted-ish project files (the user
   wrote them, but lots of OSS forks would import unfamiliar repos).
2. **Reliability** — even if the user wrote a benign computed prompt, the
   computed value at parse time may differ from the value seen at runtime
   in the user's actual app, which would silently invalidate the whole
   evaluation.

If users genuinely need a computed prompt, the documented workaround is
to use ``detection: manual`` with the rendered string inlined into
``evalshift.yaml``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from evalshift.config.models import PromptDefinition
from evalshift.parsers.base import (
    PromptParseError,
    PromptParseErrorDetail,
    PromptTemplate,
)


class PythonStringParser:
    """Extract a string-literal prompt from a ``.py`` source file via AST."""

    def parse(
        self,
        prompt: PromptDefinition,
        project_root: Path,
    ) -> PromptTemplate:
        if prompt.detection != "python_string":
            raise PromptParseError(
                prompt_id=prompt.id,
                kind="invalid_definition",
                summary=f"PythonStringParser can't handle detection={prompt.detection!r}",
            )
        if not prompt.path or not prompt.variable:
            raise PromptParseError(
                prompt_id=prompt.id,
                kind="invalid_definition",
                summary="python_string prompt is missing 'path' or 'variable' (this is a bug)",
            )

        resolved = _resolve_path(prompt.path, project_root)
        source = _read_source(prompt.id, resolved)
        tree = _parse_source(prompt.id, resolved, source)

        candidates = _module_level_assigns(tree, prompt.variable)
        if not candidates:
            raise PromptParseError(
                prompt_id=prompt.id,
                kind="variable_not_found",
                summary=(f"variable {prompt.variable!r} not found at module level in {resolved}"),
                path=resolved,
                details=[
                    PromptParseErrorDetail(
                        location="available names",
                        message=", ".join(sorted(_module_level_names(tree))) or "<none>",
                    ),
                ],
            )

        # Python evaluation order: a later assignment shadows earlier ones.
        chosen = candidates[-1]
        content = _extract_string_literal(prompt.id, resolved, chosen)
        return PromptTemplate(
            id=prompt.id,
            content=content,
            declared_variables=list(prompt.variables),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: str, project_root: Path) -> Path:
    """Resolve ``path`` against ``project_root`` if relative."""
    p = Path(path)
    return p if p.is_absolute() else (project_root / p)


def _read_source(prompt_id: str, resolved: Path) -> str:
    if not resolved.exists():
        raise PromptParseError(
            prompt_id=prompt_id,
            kind="missing_file",
            summary=f"file not found: {resolved}",
            path=resolved,
        )
    if not resolved.is_file():
        raise PromptParseError(
            prompt_id=prompt_id,
            kind="not_a_file",
            summary=f"expected a file, got {resolved}",
            path=resolved,
        )
    return resolved.read_text(encoding="utf-8")


def _parse_source(prompt_id: str, resolved: Path, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename=str(resolved))
    except SyntaxError as exc:
        raise PromptParseError(
            prompt_id=prompt_id,
            kind="ast_syntax",
            summary=f"failed to parse Python source: {exc.msg}",
            path=resolved,
            details=[
                PromptParseErrorDetail(
                    location=f"line {exc.lineno or 0}",
                    message=str(exc.msg),
                ),
            ],
        ) from exc


def _module_level_assigns(tree: ast.Module, variable: str) -> list[ast.Assign]:
    """Return every module-level ``ast.Assign`` whose first target is ``variable``."""
    out: list[ast.Assign] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == variable:
                out.append(node)
    return out


def _module_level_names(tree: ast.Module) -> set[str]:
    """Return every module-level assigned name (for error suggestions)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _extract_string_literal(prompt_id: str, resolved: Path, node: ast.Assign) -> str:
    """Pull a plain string out of an Assign value, or refuse with a clear error."""
    value = node.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value

    kind_label = _value_kind_label(value)
    raise PromptParseError(
        prompt_id=prompt_id,
        kind="non_literal",
        summary=(
            "expected a plain string literal; EvalShift refuses to evaluate Python "
            "at parse time for safety. Inline the computed value into "
            "evalshift.yaml with detection: manual instead."
        ),
        path=resolved,
        details=[
            PromptParseErrorDetail(
                location=f"line {node.lineno}",
                message=f"got {kind_label}",
            ),
        ],
    )


def _value_kind_label(value: ast.expr) -> str:
    """Label the AST node kind for a friendlier non-literal error."""
    match value:
        case ast.JoinedStr():
            return "f-string"
        case ast.BinOp(op=ast.Add()):
            return "string concatenation (BinOp +)"
        case ast.Call(func=ast.Attribute(attr="format")):
            return "method call: .format()"
        case ast.Call():
            return "function call"
        case ast.Attribute():
            return "attribute access"
        case ast.Name():
            return "name reference"
        case ast.Constant(value=v):
            return f"non-string constant: {type(v).__name__}"
        case _:
            return type(value).__name__


__all__ = ["PythonStringParser"]
