"""The docs must advertise the current command name, not a hidden alias.

`evalshift all` became `evalshift compare` in 1.0.0. The old name stays
registered forever -- scaffolded EVALSHIFT.md files in user repos reference it,
and removing it would itself be breaking -- but it is `hidden=True` and prints a
rename notice. Nine doc sites still told readers to type it, so the docs taught
a name that `evalshift --help` does not list.

Prose that *describes* the alias ("formerly `all`", "`all` -> `compare` in
1.0.0") is correct and deliberately not matched here: the assertion is on the
exact string `all --push`, which only ever appeared as advertised usage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file that documents the CLI in prose, relative to the repo root.
PROSE_FILES: tuple[str, ...] = (
    "README.md",
    "DOCS.md",
    "llms-full.txt",
    "docs/faq.md",
    "docs/hosted.md",
    "docs/configuration.md",
    "docs/index.md",
)


@pytest.mark.parametrize("name", PROSE_FILES)
def test_prose_advertises_compare_not_the_hidden_alias(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "all --push" not in text, (
        f"{name} advertises the hidden `all` alias; write `compare --push`"
    )
