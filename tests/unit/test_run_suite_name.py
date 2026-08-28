"""Tests for ``run`` suite-path resolution (the ``--suite-name`` bridge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalshift.cli.commands.run import UnknownSuiteNameError, _resolve_suite_path
from evalshift.config.models import EvalShiftConfig


def _cfg(suites: dict[str, Any] | None = None) -> EvalShiftConfig:
    return EvalShiftConfig.model_validate(
        {
            "version": 1,
            "prompts": [{"id": "p", "detection": "manual", "content": "Hi {query}"}],
            "suites": suites or {},
        },
    )


def test_explicit_suite_path_wins() -> None:
    resolved = _resolve_suite_path(
        suite_path=Path("custom.jsonl"),
        suite_name="promoted",
        cfg=_cfg({"promoted": {"source": "captured", "path": "x.jsonl"}}),
        config_path=Path("evalshift.yaml"),
    )
    assert resolved == Path("custom.jsonl")


def test_suite_name_resolves_relative_to_config_dir() -> None:
    resolved = _resolve_suite_path(
        suite_path=None,
        suite_name="promoted",
        cfg=_cfg({"promoted": {"source": "captured", "path": ".evalshift/suites/s/golden.jsonl"}}),
        config_path=Path("/proj/evalshift.yaml"),
    )
    assert resolved == Path("/proj/.evalshift/suites/s/golden.jsonl")


def test_unknown_suite_name_raises_with_known_names() -> None:
    with pytest.raises(UnknownSuiteNameError) as exc:
        _resolve_suite_path(
            suite_path=None,
            suite_name="missing",
            cfg=_cfg({"promoted": {"source": "captured", "path": "x.jsonl"}}),
            config_path=Path("evalshift.yaml"),
        )
    assert "missing" in str(exc.value)
    assert "promoted" in str(exc.value)  # lists known names to help the user


def test_default_when_neither_given() -> None:
    resolved = _resolve_suite_path(
        suite_path=None,
        suite_name=None,
        cfg=_cfg(),
        config_path=Path("evalshift.yaml"),
    )
    assert resolved == Path("golden.jsonl")
