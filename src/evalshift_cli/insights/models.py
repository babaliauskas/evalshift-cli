"""Machine-written narrative explaining a run.

The dataclasses here mirror the server's ``Insights`` model
(``evalshift-server/app/runs/bundle.py``) field-for-field, and the limit
constants mirror its validators. Both are load-bearing: that model is
``extra="forbid"`` with ``min_length=1`` on every prose field, so an empty
summary, an eleventh finding or an over-long title is a 400 at finalize —
*after* the bundle has already been uploaded. The caps live here as module
constants so ``templates.py`` and ``generator.py`` cannot disagree about them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

InsightKind = Literal["positive", "negative", "warning"]

#: The only values the server accepts for ``finding.kind``.
INSIGHT_KINDS: frozenset[str] = frozenset({"positive", "negative", "warning"})

MAX_MODEL_CHARS: int = 200
MAX_SUMMARY_CHARS: int = 2000
MAX_FINDINGS: int = 10
MAX_FINDING_TITLE_CHARS: int = 200
MAX_FINDING_DETAIL_CHARS: int = 2000

#: Cap on each text field of a regression sample sent to the model. The
#: difference between a bounded prompt and one that grows with the user's suite.
MAX_SAMPLE_TEXT_CHARS: int = 2000
#: How many regressions the model gets to look at.
MAX_REGRESSION_SAMPLES: int = 8

#: ``Insight.model`` when the prose was templated rather than generated.
FALLBACK_MODEL: str = "none"

#: One numeric token as it may appear in prose: an optional sign, an optional
#: dollar sign, digits with thousands separators, an optional fractional part
#: and an optional percent sign. The sign class carries the Unicode minus
#: (U+2212) as well as the ASCII hyphen because the facts render negatives with
#: U+2212 to match the design.
NUMERIC_TOKEN_RE: re.Pattern[str] = re.compile(r"[+−-]?\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True, slots=True)
class InsightFinding:
    """One behavioral change the target exhibits, with its icon class."""

    kind: InsightKind
    title: str
    detail: str


@dataclass(frozen=True, slots=True)
class Insight:
    """Prose only.

    Every headline figure on the report is rendered from data; this carries
    the sentences around them. ``model`` and ``generated_at`` are provenance —
    a reader must be able to tell the prose is machine-written.
    """

    model: str
    generated_at: datetime
    verdict_summary: str
    advisory_summary: str
    economics_summary: str
    recommendation: str
    findings: list[InsightFinding] = field(default_factory=list)


#: The server's ``Insights`` keys, exactly. It is ``extra="forbid"``, so this
#: set is both what :func:`insight_to_dict` emits and what
#: :func:`insight_from_dict` will accept.
INSIGHT_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "generated_at",
        "verdict_summary",
        "advisory_summary",
        "economics_summary",
        "recommendation",
        "findings",
    }
)
_FINDING_KEYS: frozenset[str] = frozenset({"kind", "title", "detail"})


def insight_to_dict(insight: Insight) -> dict[str, Any]:
    """Serialise an :class:`Insight` into the server's ``Insights`` shape.

    Emits :data:`INSIGHT_KEYS` and nothing else. Anything the CLI wants to
    remember *about* a narrative — a cache key, a generation attempt count —
    belongs in the envelope around this, never beside the prose.
    """
    return {
        "model": insight.model,
        "generated_at": _format_z(insight.generated_at),
        "verdict_summary": insight.verdict_summary,
        "advisory_summary": insight.advisory_summary,
        "economics_summary": insight.economics_summary,
        "recommendation": insight.recommendation,
        "findings": [
            {"kind": finding.kind, "title": finding.title, "detail": finding.detail}
            for finding in insight.findings
        ],
    }


def insight_from_dict(payload: Mapping[str, Any]) -> Insight:
    """Rebuild an :class:`Insight` from :func:`insight_to_dict` output.

    Strict on purpose: this reads a file a user can hand-edit, and the caller
    treats any :class:`ValueError` as a cache miss. Regenerating a narrative
    costs one call; shipping one the server rejects costs the whole upload.

    Raises:
        ValueError: If a key is missing, unexpected, or the wrong type.
    """
    unexpected = sorted(set(payload) - INSIGHT_KEYS)
    if unexpected:
        raise ValueError(f"unexpected key(s) in insight: {', '.join(unexpected)}")
    missing = sorted(INSIGHT_KEYS - {"findings"} - set(payload))
    if missing:
        raise ValueError(f"missing key(s) in insight: {', '.join(missing)}")

    findings = payload.get("findings") or []
    if not isinstance(findings, list):
        raise ValueError(f"findings must be a list, got {type(findings).__name__}")
    return Insight(
        model=_text(payload, "model"),
        generated_at=_parse_z(_text(payload, "generated_at")),
        verdict_summary=_text(payload, "verdict_summary"),
        advisory_summary=_text(payload, "advisory_summary"),
        economics_summary=_text(payload, "economics_summary"),
        recommendation=_text(payload, "recommendation"),
        findings=[_finding_from_dict(item) for item in findings],
    )


def _finding_from_dict(payload: Any) -> InsightFinding:
    if not isinstance(payload, Mapping):
        raise ValueError(f"finding must be an object, got {type(payload).__name__}")
    unexpected = sorted(set(payload) - _FINDING_KEYS)
    if unexpected:
        raise ValueError(f"unexpected key(s) in finding: {', '.join(unexpected)}")
    kind = _text(payload, "kind")
    if kind not in INSIGHT_KINDS:
        raise ValueError(f"unknown finding kind: {kind!r}")
    return InsightFinding(
        kind=cast(InsightKind, kind),
        title=_text(payload, "title"),
        detail=_text(payload, "detail"),
    )


def _text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _format_z(value: datetime) -> str:
    """Render a timestamp the way the bundle spec's ``date-time`` fields read."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_z(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"generated_at is not an ISO-8601 timestamp: {value!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def clamp_text(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, preferring a word boundary.

    Used on the way out of both the templates and the generator. A narrative
    that ran long is still a good narrative — it just cannot be shipped at
    full length, because the server rejects the whole bundle over it.
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    cut = stripped[:limit]
    boundary = cut.rfind(" ")
    # Only honour the boundary when it isn't throwing most of the text away;
    # a single unbroken token would otherwise collapse to nothing.
    if boundary > limit // 2:
        cut = cut[:boundary]
    return cut.rstrip()


__all__ = [
    "FALLBACK_MODEL",
    "INSIGHT_KEYS",
    "INSIGHT_KINDS",
    "MAX_FINDINGS",
    "MAX_FINDING_DETAIL_CHARS",
    "MAX_FINDING_TITLE_CHARS",
    "MAX_MODEL_CHARS",
    "MAX_REGRESSION_SAMPLES",
    "MAX_SAMPLE_TEXT_CHARS",
    "MAX_SUMMARY_CHARS",
    "NUMERIC_TOKEN_RE",
    "Insight",
    "InsightFinding",
    "InsightKind",
    "clamp_text",
    "insight_from_dict",
    "insight_to_dict",
]
