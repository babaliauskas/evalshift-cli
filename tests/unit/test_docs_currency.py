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


#: Files that describe *which providers work*, as opposed to naming three as
#: examples. Each must name LiteLLM, because LiteLLM is the actual boundary:
#: `models/registry.py` says so in its module docstring, and `docs/faq.md`
#: already answers "which models?" with "Anything LiteLLM supports."
PROVIDER_SCOPE_FILES: tuple[str, ...] = (
    "README.md",
    "docs/index.md",
    "docs/faq.md",
    "docs/hosted.md",
)


@pytest.mark.parametrize("name", PROVIDER_SCOPE_FILES)
def test_provider_scope_is_not_capped_at_three(name: str) -> None:
    """The curated registry has three entries; the CLI calls far more than three.

    `Provider` is a Literal of three names plus "other" because those three have
    pricing tables and env-var mappings worth curating. Every call still goes
    through `litellm.acompletion`, and `resolve_model` never raises -- an
    unregistered id is dispatched with a prefix-inferred provider. Prose that
    lists the three without naming LiteLLM reads as a compatibility list and
    undersells the tool.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "LiteLLM" in text, f"{name} scopes providers without naming LiteLLM"
