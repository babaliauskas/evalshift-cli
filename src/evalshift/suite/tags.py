"""Tags EvalShift itself writes onto suite examples.

Most tags on a :class:`~evalshift.suite.models.SuiteExample` come from the
user and mean whatever they want them to mean. A few are written by the CLI
to record *where an example came from* rather than to name a subset worth
analysing separately — those live here so the writer
(:mod:`evalshift.captures.promote`) and the readers agree on the literal.

Provenance tags are still ordinary tags: they build slices like any other,
and a suite that mixes promoted and hand-written cases gets a genuinely
useful ``captured`` slice out of them. They are only special when a slice
is redundant and something has to decide which name to keep — see
:func:`evalshift.analysis.slicing.dedupe_slices`.
"""

from __future__ import annotations

CAPTURED_TAG: str = "captured"
"""Written by ``evalshift capture promote`` onto every promoted example."""

PROVENANCE_TAGS: frozenset[str] = frozenset({CAPTURED_TAG})
"""Tags recording an example's origin rather than naming an analysis subset."""

RESERVED_SLICE_NAME: str = "overall"
"""A slice name nothing may claim, because the run-level scope already has it.

`decision.overall` is the whole-run summary and `BudgetResult.scope` defaults to
`"overall"`, so a slice by that name shadows the run's own numbers wherever the two
render side by side. `BUNDLE_SPEC.md` has always said so; the server enforces it at
finalize. A slice name reaches a bundle from a suite tag, from `SliceConfig.name`, or
from a `migration_policy.slices` key, and all three refuse it — the literal lives in
this module because it is the only one all three can import without a cycle.
"""

__all__ = ["CAPTURED_TAG", "PROVENANCE_TAGS", "RESERVED_SLICE_NAME"]
