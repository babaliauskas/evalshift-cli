"""Content-address an already-normalised toolset by fingerprinting it.

Every model call records the list of tools it was offered, content-addressed by a SHA-256
fingerprint so a toolset (potentially tens of KB of JSON Schema) is written once as a sidecar
(``<base>/toolsets/<hex>.json`` -- see :func:`evalshift.captures.reader.toolset_path`) and
referenced by every capture that used it via ``toolset_ref``, instead of inlining the same bytes
into every capture.

:func:`fingerprint_tools` ports ``evalshift-sdk``'s ``evalshift.capture.toolset.fingerprint_tools``
(the side that writes sidecars) step-for-step, so the two independently-maintained
implementations of one hashing rule cannot silently drift apart:
``tests/unit/test_captures_toolset.py`` asserts the identical two constants the SDK's test suite
pins, copied verbatim.

This module deliberately does **not** port the SDK's ``normalize_tools`` -- the piece that
reduces a live Anthropic/OpenAI/Gemini SDK object to ``{name, description, input_schema}``. The
CLI never receives a live provider SDK object: it only ever fingerprints tools already in that
canonical shape -- a toolset sidecar is pre-normalised JSON written by the SDK, and a
hand-authored suite's inline tools already validate against
:class:`evalshift.evaluators.tool_models.ToolSpec`, whose wire format is the same three fields.
Should a future caller here need provider-shape normalisation, port it then, from the same SDK
source, rather than speculatively now (YAGNI).

Stdlib only, deliberately -- so this implementation stays directly, line-for-line comparable to
the SDK's.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final


def fingerprint_tools(normalized: list[dict[str, Any]]) -> str:
    """Content-address an already-normalised tool list.

    Reuses ``evalshift-server/BUNDLE_SPEC.md``'s Hashing section verbatim, so one hashing rule
    holds product-wide (identical to ``evalshift-sdk``'s
    ``evalshift.capture.toolset.fingerprint_tools``):

    1. (done by the caller) normalise each tool to ``{name, description, input_schema}``.
    2. Sort the tool list by ``name``.
    3. ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``.
    4. SHA-256 the UTF-8 bytes, hex-encode, prefix ``sha256:``.

    Sorting the list (step 2) is required *in addition to* ``sort_keys=True`` -- ``sort_keys``
    only orders each dict's own keys, never list element order, so two logically-identical
    toolsets recorded in a different call order would otherwise fingerprint differently.

    Args:
        normalized: A list of ``{name, description, input_schema}`` dicts -- e.g. a parsed
            toolset sidecar, or ``[spec.to_anthropic() for spec in tool_specs]`` for a
            hand-authored suite's inline tools. This function's precondition is that
            normalisation already happened; it does not check the shape.

    Returns:
        ``"sha256:" + hex digest`` of the tool list's canonical JSON form.
    """
    ordered = sorted(normalized, key=lambda tool: tool["name"])
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


#: :func:`fingerprint_tools` of the empty toolset ("no tools offered"). A
#: property of the hashing algorithm itself -- there is exactly one possible
#: fingerprint for zero tools -- not of any one caller, which is why it lives
#: here beside the function that produces it rather than beside any one
#: consumer. Two independent consumers rely on it: ``cli.commands.doctor``
#: (render "no tools" instead of an opaque hash in a report row) and
#: ``suite.models.SuiteExample`` (reject a ``toolset_ref`` naming the empty
#: toolset paired with non-empty tool-call ground truth -- the ``toolset_ref``
#: mirror of rejecting inline ``tools == []`` paired with the same, since both
#: spellings assert the identical thing and neither can ever satisfy a
#: ground-truth tool call).
EMPTY_TOOLSET_FINGERPRINT: Final = fingerprint_tools([])


__all__ = ["EMPTY_TOOLSET_FINGERPRINT", "fingerprint_tools"]
