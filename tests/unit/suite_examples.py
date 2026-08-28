"""Shared helper for constructing ``SuiteExample`` rows in tests that don't care about tools.

``SuiteExample.toolset_ref`` / ``.tools`` are exactly-one-of and required (see
``evalshift.suite.models.SuiteExample._check_exactly_one_toolset_field`` and
``PER_CALL_TOOLSET_CAPTURE_PLAN.md`` V7): every model call records the toolset it was
offered, so a suite example -- promoted or hand-authored -- must carry it too.

Most tests in this repo are about something else entirely (templating, orchestrator
dispatch, report rendering, slicing, ...) and have no opinion on what the toolset was, so
``suite_example`` fills in ``tools=[]`` -- "this example's agent had no tools available", a
real, valid assertion -- whenever the caller specifies neither field.

``tools=[]`` is incompatible with tool-call ground truth (``SuiteExample`` rejects that
combination -- see ``suite/models.py``'s ``_check_tool_expectations_consistent``, I2 of the
final per-call-toolset-capture review): a test that passes ``expected_tools`` /
``expected_tool_count`` / ``expected_tool_rounds`` without naming a toolset gets a neutral
placeholder ``toolset_ref`` instead of the empty-tools default, so evaluator-scoring tests
(which exercise scoring directly against ``Call``/``ToolTrace`` objects, never
``resolve_example_tools``, so no real sidecar on disk is ever needed) keep working without
every such call site having to know about this interaction. Tests that genuinely ARE about
the toolset fields themselves pass ``toolset_ref=`` or ``tools=`` explicitly and this defers
to them unchanged.
"""

from __future__ import annotations

from typing import Any

from evalshift.suite.models import SuiteExample

#: Neutral, never-resolved placeholder -- see the module docstring's second paragraph.
_PLACEHOLDER_TOOLSET_REF = "sha256:" + "00" * 32


def _asserts_tool_ground_truth(kwargs: dict[str, Any]) -> bool:
    return (
        bool(kwargs.get("expected_tools"))
        or bool(kwargs.get("expected_tool_rounds"))
        or (kwargs.get("expected_tool_count") or 0) > 0
    )


def suite_example(**kwargs: Any) -> SuiteExample:
    """Build a ``SuiteExample``, defaulting a toolset when neither field is given.

    Defaults to ``tools=[]`` unless ``kwargs`` asserts tool-call ground truth
    (see :func:`_asserts_tool_ground_truth`), in which case it defaults to
    :data:`_PLACEHOLDER_TOOLSET_REF` instead -- ``tools=[]`` paired with
    ground truth is rejected at construction (I2).
    """
    if "tools" not in kwargs and "toolset_ref" not in kwargs:
        if _asserts_tool_ground_truth(kwargs):
            kwargs["toolset_ref"] = _PLACEHOLDER_TOOLSET_REF
        else:
            kwargs["tools"] = []
    return SuiteExample(**kwargs)


__all__ = ["suite_example"]
