"""Smoke test that the package imports and exposes ``__version__``."""

from __future__ import annotations

import re

import aimigrate


def test_version_is_semver_like() -> None:
    assert isinstance(aimigrate.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+(?:[-+.].+)?$", aimigrate.__version__)
