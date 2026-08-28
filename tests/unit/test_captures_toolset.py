"""Tests for toolset fingerprinting (``evalshift.captures.toolset``).

``THREE_TOOL_FINGERPRINT`` and ``EMPTY_TOOLSET_FINGERPRINT`` below are the shared vector defined
by ``evalshift-sdk``'s ``tests/test_toolset.py`` (Task 1 of the per-call-toolset-capture plan).
This file asserts the identical two values, copied verbatim, so the SDK's and the CLI's
independent hashing implementations cannot silently drift apart. ``THREE_TOOL_NORMALIZED`` is
likewise copied verbatim from that file's shared-vector input, already in normalised form: this
repo has no ``normalize_tools`` (see ``captures/toolset.py``'s module docstring for why), so the
vector here starts from the normalised shape directly rather than from the three raw
Anthropic/OpenAI/Gemini provider shapes the SDK's vector is built from.
"""

from __future__ import annotations

from typing import Any

from evalshift.captures.toolset import fingerprint_tools

# --- The shared vector, copied verbatim from evalshift-sdk's tests/test_toolset.py -----------

THREE_TOOL_NORMALIZED: list[dict[str, Any]] = [
    {
        "name": "get_schedule",
        "description": "Look up a user's schedule for a given date.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
    },
    {
        "name": "add_task",
        "description": "Add a task to the user's to-do list.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    },
]

# Pinned by running the SDK's real implementation once and copying its output -- a content hash
# cannot be derived by inspection. If a future edit to either implementation changes either
# hash, one of the two repos goes red.
THREE_TOOL_FINGERPRINT = "sha256:8128183b3b2871d3887b7a21b3ac2e1928d939180e9ea12ec583e93bb1ccebde"
EMPTY_TOOLSET_FINGERPRINT = (
    "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)


# --- fingerprint_tools: determinism, and independence from key/list order --------------------


def test_fingerprint_is_deterministic() -> None:
    assert fingerprint_tools(THREE_TOOL_NORMALIZED) == fingerprint_tools(THREE_TOOL_NORMALIZED)


def test_fingerprint_independent_of_list_order() -> None:
    forward = THREE_TOOL_NORMALIZED
    backward = list(reversed(THREE_TOOL_NORMALIZED))
    assert fingerprint_tools(forward) == fingerprint_tools(backward)


def test_fingerprint_independent_of_dict_key_order() -> None:
    forward = [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]
    reordered = [{"input_schema": {"type": "object"}, "description": "d", "name": "x"}]
    assert fingerprint_tools(forward) == fingerprint_tools(reordered)


def test_fingerprint_differs_for_different_toolsets() -> None:
    one = [THREE_TOOL_NORMALIZED[0]]
    two = [THREE_TOOL_NORMALIZED[1]]
    assert fingerprint_tools(one) != fingerprint_tools(two)


def test_fingerprint_has_sha256_prefix_and_64_char_hex_digest() -> None:
    fp = fingerprint_tools([])
    assert fp.startswith("sha256:")
    hex_part = fp.removeprefix("sha256:")
    assert len(hex_part) == 64
    int(hex_part, 16)  # raises ValueError if this is not valid hex


# --- The shared vector: asserted identical to evalshift-sdk's tests/test_toolset.py ----------


def test_three_tool_vector_matches_pinned_fingerprint() -> None:
    assert fingerprint_tools(THREE_TOOL_NORMALIZED) == THREE_TOOL_FINGERPRINT


def test_empty_toolset_vector_matches_pinned_fingerprint() -> None:
    assert fingerprint_tools([]) == EMPTY_TOOLSET_FINGERPRINT
