"""Evaluator: did the target pass similar arguments to matched tool calls?

Greedy match by ``(tool_name, nearest sequence_index)`` — when both
source and target call the same tool multiple times we line them up by
position. Hungarian-algorithm matching is opt-in for v0.3 (PRD risk #3).

With ``against: expected`` the comparison is against ground truth instead:
each ``example.expected_tools`` entry that carries arguments is paired with
an actual call of the same name, on **both** sides independently, so a
source model that passed a value that does not exist scores as wrong.

Per-field strategies:

* ``exact`` — string equality of repr-stringified values.
* ``subset`` — recursive structural subset (dict/list aware).
* ``numeric`` — relative-error decay clamped by ``numeric_tolerance``.
* ``semantic`` — cosine similarity of embedded values via an injected
  ``embeddings_fn``. Falls back to ``exact`` when ``embeddings_fn`` is
  ``None`` (typical in unit tests).
* ``auto`` — the default. A ladder, cheapest rung first:

  1. **Normalized exact.** Strings that compare equal after normalization
     (case, surrounding and repeated whitespace) score 1.0. No I/O, no schema,
     no API call.
  2. **Schema dispatch.** The field is looked up in the toolset the capture
     recorded for this example (injected via ``toolset_resolver``) and its
     declared type picks the strategy: identifiers, enums, booleans and
     ``date``/``date-time``/``uuid``/``email`` formats are scored ``exact``,
     numbers ``numeric``, objects and arrays ``subset``. See
     :func:`_schema_strategy`.
  3. **Graded similarity.** Whatever the schema did not decide — free text,
     and anything with no schema at all — is graded: ``semantic`` when an
     ``embeddings_fn`` is available, ``difflib`` ratio when it is not, so
     partial credit survives with no embedding model. Numbers still go
     through ``numeric``, dicts and lists through ``subset``, everything
     else through ``exact``.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from evalshift.config.models import ToolArgumentsEvaluatorConfig
from evalshift.evaluators.base import EvalRecord, PairedScore
from evalshift.evaluators.failures import ARGUMENT_VALUE_DRIFT
from evalshift.evaluators.tool_models import ToolCall, ToolSpec, ToolTrace
from evalshift.suite.models import ExpectedToolCall, SuiteExample

# Type for the embeddings function: (a, b) -> cosine similarity in [0, 1].
EmbeddingsFn = Callable[[str, str], Awaitable[float]]

#: Resolve one example's toolset — inline ``tools`` or a ``toolset_ref``
#: sidecar. Returns ``None`` when this example has no resolvable toolset;
#: raising is tolerated too (see :meth:`ToolArgumentsEvaluator._resolve_tools`).
ToolsetResolver = Callable[[SuiteExample], "list[ToolSpec] | None"]


#: Stable evaluator-type slug stamped onto every record. The analysis
#: layer selects policy rows on this, never on the user-chosen name.
KIND = "tool_arguments"


class ToolArgumentsEvaluator:
    """Score how similar the target's tool arguments are to the source's."""

    kind = KIND

    def __init__(
        self,
        config: ToolArgumentsEvaluatorConfig,
        *,
        embeddings_fn: EmbeddingsFn | None = None,
        toolset_resolver: ToolsetResolver | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self._embeddings_fn = embeddings_fn
        self._toolset_resolver = toolset_resolver

    async def score_pair(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        """Score a (source, target) trace pair's arguments for one example.

        Returns ``None`` when ``against: expected`` and the example carries
        no ground-truth arguments — nothing was measured, so no row.
        """
        if self.config.against == "expected":
            return await self._score_against_expected(
                run_id=run_id,
                prompt_id=prompt_id,
                example=example,
                source_trace=source_trace,
                target_trace=target_trace,
            )

        tools = self._resolve_tools(example)
        matched = _match_calls(source_trace, target_trace)
        if not matched:
            # Either side has no calls of any matched tool name → no pairs
            # to compare. Treat as 1.0 if both empty, 0.0 if target made
            # uncomparable calls (likely a regression vs. source).
            target_score = 1.0 if target_trace.call_count == 0 else 0.0
            return self._record(
                run_id=run_id,
                prompt_id=prompt_id,
                example_id=example.id,
                paired=PairedScore(
                    source_score=1.0,
                    target_score=target_score,
                    metadata={
                        "reason": "no matched calls between source and target",
                        "failure_categories": (
                            [ARGUMENT_VALUE_DRIFT] if target_score < 1.0 else []
                        ),
                    },
                ),
            )

        per_call: list[float] = []
        per_call_meta: list[dict[str, Any]] = []
        for src_call, tgt_call in matched:
            score, detail = await self._score_call_args(
                src_call.tool_name,
                src_call.arguments,
                tgt_call.arguments,
                tools=tools,
            )
            per_call.append(score)
            per_call_meta.append(detail)

        target_score = sum(per_call) / len(per_call)
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=1.0,
                target_score=target_score,
                metadata={
                    "per_call": per_call_meta,
                    "failure_categories": [ARGUMENT_VALUE_DRIFT] if target_score < 1.0 else [],
                },
            ),
        )

    # ------------------------------------------------------------------
    # Ground-truth scoring (``against: expected``)
    # ------------------------------------------------------------------

    async def _score_against_expected(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord | None:
        """Score both sides' arguments against ``example.expected_tools``.

        Each expected call that carries arguments is paired with the first
        not-yet-consumed actual call of the same tool name, then scored
        field-by-field with the configured strategies. An expected call the
        model never made scores 0 — a missing call cannot have correct
        arguments. Extra calls beyond the expectation are ignored here;
        ``tool_selection`` is what scores call *presence*.

        Args:
            run_id: Run this record belongs to.
            prompt_id: Prompt this pair was produced under.
            example: Suite row carrying the ground-truth expectations.
            source_trace: Tool calls the source model made.
            target_trace: Tool calls the target model made.

        Returns:
            One :class:`EvalRecord`, or ``None`` when the example carries no
            expected arguments to score against — there is no ground truth,
            so nothing was measured and no row is written. The 1.0/1.0 this
            used to return read as a full argument match.
        """
        expected = [e for e in (example.expected_tools or []) if e.arguments]
        if not expected:
            return None

        tools = self._resolve_tools(example)
        source_score, source_meta, target_score, target_meta = await self._score_sides(
            expected,
            source_trace,
            target_trace,
            tools,
        )
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            paired=PairedScore(
                source_score=source_score,
                target_score=target_score,
                metadata={
                    "against": "expected",
                    "expected_calls": len(expected),
                    # Travels on the record because the analysis layer never
                    # sees suite rows: it is what lets the run disclose that
                    # ground truth transcribed from the source model pins
                    # `source_score` at 1.0 by construction.
                    "gt_provenance": _ground_truth_provenance(expected),
                    "per_call_source": source_meta,
                    "per_call": target_meta,
                    # A regression, not a correctness score: both sides equally
                    # short of ground truth is a suite the source never
                    # satisfied either, and stamping it counts one migration
                    # defect twice.
                    "failure_categories": (
                        [ARGUMENT_VALUE_DRIFT] if target_score < source_score else []
                    ),
                },
            ),
        )

    async def _score_sides(
        self,
        expected: list[ExpectedToolCall],
        source_trace: ToolTrace,
        target_trace: ToolTrace,
        tools: Sequence[ToolSpec] | None = None,
    ) -> tuple[float, list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Mean per-expected-call argument score for both models' traces.

        Both sides are scored together rather than independently because one
        judgement needs both of them: a ground-truth field that *neither*
        model produced is a stale expectation, not a model defect, so it is
        dropped from that call's denominator on both sides (and disclosed as
        ``unmeasured_fields``). Scored, it would cap the call below 1.0 for
        good — no model change could ever lift it.

        Args:
            expected: Ground-truth calls that carry arguments.
            source_trace: Tool calls the source model made.
            target_trace: Tool calls the target model made.
            tools: Resolved toolset for schema dispatch, or ``None``.

        Returns:
            ``(source_score, source_per_call, target_score, target_per_call)``.
        """
        source_matches = _pair_with_expected(expected, source_trace)
        target_matches = _pair_with_expected(expected, target_trace)
        source_meta: list[dict[str, Any]] = []
        target_meta: list[dict[str, Any]] = []
        source_total = 0.0
        target_total = 0.0
        for exp, src_call, tgt_call in zip(expected, source_matches, target_matches, strict=True):
            exp_args = exp.arguments or {}
            recorded = {k for k in exp_args if not k.startswith("_")}
            produced = {
                key for call in (src_call, tgt_call) if call is not None for key in call.arguments
            }
            unmeasured = recorded - produced
            # ``exact`` expectations also police arguments the ground truth
            # did not record; ``subset`` / ``contains_per_field`` — what
            # promotion writes — score the recorded keys only.
            keys = None if exp.match_strategy == "exact" else recorded
            source_score, source_detail = await self._score_expected_call(
                exp, src_call, keys=keys, unmeasured=unmeasured, tools=tools
            )
            target_score, target_detail = await self._score_expected_call(
                exp, tgt_call, keys=keys, unmeasured=unmeasured, tools=tools
            )
            source_total += source_score
            source_meta.append(source_detail)
            target_total += target_score
            target_meta.append(target_detail)
        return (
            source_total / len(expected),
            source_meta,
            target_total / len(expected),
            target_meta,
        )

    async def _score_expected_call(
        self,
        exp: ExpectedToolCall,
        call: ToolCall | None,
        *,
        keys: set[str] | None,
        unmeasured: set[str],
        tools: Sequence[ToolSpec] | None,
    ) -> tuple[float, dict[str, Any]]:
        """Score one model's answer to one expectation."""
        if call is None:
            # A call the model never made cannot have right arguments.
            return 0.0, {"tool_name": exp.tool_name, "missing": True}
        return await self._score_call_args(
            exp.tool_name,
            exp.arguments or {},
            call.arguments,
            keys=keys,
            tools=tools,
            unmeasured=unmeasured,
        )

    # ------------------------------------------------------------------
    # Per-call / per-field scoring
    # ------------------------------------------------------------------

    def _resolve_tools(self, example: SuiteExample) -> Sequence[ToolSpec] | None:
        """Resolve this example's toolset for schema dispatch, or ``None``.

        Called once per scored pair rather than once per field: resolving a
        ``toolset_ref`` reads a sidecar off disk, and every field of every
        call in one example shares the same toolset.

        A resolver that returns ``None`` or raises means "no schema
        available" — ``auto`` then falls back to its schema-free ladder.
        Schema dispatch is an accuracy refinement, never a precondition:
        a missing or broken sidecar must not fail a scoring run.
        """
        if self._toolset_resolver is None:
            return None
        try:
            return self._toolset_resolver(example)
        except Exception:
            return None

    def _record(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example_id: str,
        paired: PairedScore,
    ) -> EvalRecord:
        """Wrap a :class:`PairedScore` in this evaluator's identity."""
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example_id,
            evaluator_name=self.name,
            kind=KIND,
            source_score=paired.source_score,
            target_score=paired.target_score,
            delta=paired.delta,
            metadata=paired.metadata,
        )

    async def _score_call_args(
        self,
        tool_name: str,
        src_args: dict[str, Any],
        tgt_args: dict[str, Any],
        *,
        keys: set[str] | None = None,
        tools: Sequence[ToolSpec] | None = None,
        unmeasured: set[str] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        # Sentinel keys record parser-level errors — unparseable arguments
        # are a total failure, not an empty comparison.
        if "_parse_error" in src_args or "_parse_error" in tgt_args:
            return 0.0, {
                "tool_name": tool_name,
                "field_scores": {},
                "_parse_error": True,
            }

        all_keys = set(src_args) | set(tgt_args) if keys is None else set(keys)
        all_keys = {k for k in all_keys if not k.startswith("_")}
        dropped = sorted(all_keys & unmeasured) if unmeasured else []
        all_keys -= set(dropped)
        # Reported so a stale expectation surfaces as suite maintenance
        # instead of disappearing into a score nobody can move.
        disclosure: dict[str, Any] = {"unmeasured_fields": dropped} if dropped else {}
        if not all_keys:
            return 1.0, {"tool_name": tool_name, "field_scores": {}, **disclosure}

        field_scores: dict[str, float] = {}
        for key in all_keys:
            strategy = self.config.strategies.get(key, self.config.default_strategy)
            field_scores[key] = await self._score_field(
                src_args.get(key),
                tgt_args.get(key),
                strategy,
                tool_name=tool_name,
                field=key,
                tools=tools,
            )
        avg = sum(field_scores.values()) / len(field_scores)
        return avg, {
            "tool_name": tool_name,
            "field_scores": field_scores,
            **disclosure,
        }

    async def _score_field(
        self,
        src_val: Any,
        tgt_val: Any,
        strategy: str,
        *,
        tool_name: str = "",
        field: str = "",
        tools: Sequence[ToolSpec] | None = None,
    ) -> float:
        if src_val is None and tgt_val is None:
            return 1.0
        if src_val is None or tgt_val is None:
            # One side omitted the field entirely. Under ``strict`` that is a
            # total mismatch; under ``lenient`` it is a partial one — an
            # omitted optional filter is not the same class of error as a
            # wrong value, and scoring it 0.0 turns every tool with optional
            # parameters into a false regression.
            return 0.0 if self.config.optional_fields_scored == "strict" else 0.5
        if strategy == "exact":
            return 1.0 if src_val == tgt_val else 0.0
        if strategy == "subset":
            return 1.0 if _is_subset(src_val, tgt_val) else 0.0
        if strategy == "numeric":
            return self._score_numeric(src_val, tgt_val)
        if strategy == "semantic":
            return await self._score_semantic(src_val, tgt_val)
        if strategy == "auto":
            return await self._score_auto(
                src_val,
                tgt_val,
                tool_name=tool_name,
                field=field,
                tools=tools,
            )
        # Unknown strategy: fall back to exact.
        return 1.0 if src_val == tgt_val else 0.0

    async def _score_auto(
        self,
        src_val: Any,
        tgt_val: Any,
        *,
        tool_name: str = "",
        field: str = "",
        tools: Sequence[ToolSpec] | None = None,
    ) -> float:
        """Pick a strategy from the field's schema, or from the values themselves.

        Free-text arguments are the reason this exists: ``exact`` reads a
        capitalization difference as a wrong value, which turns prose
        arguments into phantom regressions. The schema rung is what keeps
        the opposite from happening — an id, a timestamp, or an enum must
        not collect partial credit for looking similar.
        """
        both_strings = isinstance(src_val, str) and isinstance(tgt_val, str)
        if both_strings and _normalize(src_val) == _normalize(tgt_val):
            return 1.0
        schema_strategy = _schema_strategy(tools, tool_name, field)
        if schema_strategy is not None:
            return await self._score_field(src_val, tgt_val, schema_strategy)
        if both_strings:
            if self._embeddings_fn is not None:
                return await self._score_semantic(src_val, tgt_val)
            return difflib.SequenceMatcher(None, src_val, tgt_val).ratio()
        if _is_number(src_val) and _is_number(tgt_val):
            return self._score_numeric(src_val, tgt_val)
        if isinstance(src_val, dict | list) and isinstance(tgt_val, dict | list):
            return 1.0 if _is_subset(src_val, tgt_val) else 0.0
        return 1.0 if src_val == tgt_val else 0.0

    def _score_numeric(self, src_val: Any, tgt_val: Any) -> float:
        try:
            s = float(src_val)
            t = float(tgt_val)
        except (TypeError, ValueError):
            return 0.0
        if s == 0 and t == 0:
            return 1.0
        denom = max(abs(s), abs(t), 1e-9)
        rel = abs(s - t) / denom
        tolerance = max(self.config.numeric_tolerance, 1e-9)
        # Linear decay: 1.0 at zero error, 0.0 once error == tolerance.
        return max(0.0, 1.0 - rel / tolerance)

    async def _score_semantic(self, src_val: Any, tgt_val: Any) -> float:
        if self._embeddings_fn is None:
            return 1.0 if str(src_val) == str(tgt_val) else 0.0
        try:
            sim = await self._embeddings_fn(str(src_val), str(tgt_val))
        except Exception:
            return 1.0 if str(src_val) == str(tgt_val) else 0.0
        if math.isnan(sim):
            return 0.0
        return max(0.0, min(1.0, float(sim)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ground_truth_provenance(expected: Sequence[ExpectedToolCall]) -> str:
    """Summarise where a row's scored expectations came from.

    Args:
        expected: The expected calls actually scored (those carrying
            arguments); an expectation with no arguments contributes no
            ground truth to disclose.

    Returns:
        ``"captured"`` or ``"reviewed"`` when every expectation agrees,
        ``"mixed"`` otherwise -- a row a human has half-checked is neither
        fully source-derived nor fully trustworthy, and the disclosure is
        only ever made about a run whose every row is the former.
    """
    kinds = {e.provenance for e in expected}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _pair_with_expected(
    expected: Sequence[ExpectedToolCall],
    trace: ToolTrace,
) -> list[ToolCall | None]:
    """Pair each expectation with the first unconsumed call of the same name.

    Args:
        expected: Ground-truth calls, in order.
        trace: One model's recorded calls.

    Returns:
        One entry per expectation, positionally aligned: the matched call, or
        ``None`` when the model never made a call of that name.
    """
    remaining = list(trace.calls)
    matches: list[ToolCall | None] = []
    for exp in expected:
        match = next((c for c in remaining if c.tool_name == exp.tool_name), None)
        if match is not None:
            remaining.remove(match)
        matches.append(match)
    return matches


def _match_calls(
    source: ToolTrace,
    target: ToolTrace,
) -> list[tuple[ToolCall, ToolCall]]:
    """Greedy nearest-index match between same-named source and target calls.

    Documented v0.2 simplification — works perfectly when each tool
    appears at most once and degrades gracefully with repeats. Hungarian
    matching deferred to v0.3 (PRD risk #3).
    """
    matched: list[tuple[ToolCall, ToolCall]] = []
    used: set[int] = set()
    for src in source.calls:
        best: tuple[int, ToolCall] | None = None
        best_distance = math.inf
        for i, tgt in enumerate(target.calls):
            if i in used or tgt.tool_name != src.tool_name:
                continue
            distance = abs(tgt.sequence_index - src.sequence_index)
            if distance < best_distance:
                best_distance = distance
                best = (i, tgt)
        if best is not None:
            used.add(best[0])
            matched.append((src, best[1]))
    return matched


#: Runs of any whitespace collapse to a single space before comparison.
_WHITESPACE_RUN = re.compile(r"\s+")

#: A field whose name ends in ``id``/``ids`` (on a word boundary) names an
#: identifier, whatever JSON type it was declared as. ``userid`` is
#: deliberately not matched -- ``_ID`` requires the separator so ``valid``
#: and ``hybrid`` do not become identifiers by accident.
_ID_FIELD = re.compile(r"(?:^|_)ids?$", re.IGNORECASE)

#: JSON Schema ``format`` values whose strings are machine-readable: a
#: reworded one is a different value, not a similar one.
_EXACT_FORMATS = frozenset({"date", "date-time", "time", "uuid", "email", "uri"})


def _field_schema(
    tools: Sequence[ToolSpec] | None,
    tool_name: str,
    field: str,
) -> dict[str, Any] | None:
    """The JSON Schema subtree describing ``field`` of ``tool_name``, if any."""
    if not tools or not tool_name or not field:
        return None
    spec = next((t for t in tools if t.name == tool_name), None)
    if spec is None:
        return None
    properties = spec.input_schema.get("properties")
    if not isinstance(properties, dict):
        return None
    prop = properties.get(field)
    return prop if isinstance(prop, dict) else None


def _declared_type(prop: dict[str, Any]) -> str | None:
    """The field's JSON type, unwrapping the ``["string", "null"]`` union form."""
    declared = prop.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        return next((t for t in declared if isinstance(t, str) and t != "null"), None)
    return None


def _schema_strategy(
    tools: Sequence[ToolSpec] | None,
    tool_name: str,
    field: str,
) -> str | None:
    """Pick a scoring strategy for one argument from the tool's own schema.

    The capture already recorded the toolset every call was dispatched with,
    which is the only place that knows an argument is a timestamp rather than
    prose. Rungs, in priority order:

    * ``object`` / ``array`` -> ``subset``. Structural containers compare
      structurally; :func:`_is_subset` still compares leaves with ``==``, so
      an id *inside* an id array is matched exactly and only extras are
      forgiven.
    * ``enum`` present, or ``boolean``, or a machine-readable ``format``
      (:data:`_EXACT_FORMATS`) -> ``exact``.
    * a name ending in ``id`` / ``ids`` -> ``exact``, whatever the declared
      type. Numeric tolerance on an identifier is nonsense: 41 is not nearly
      42 when it names a different project.
    * ``integer`` / ``number`` -> ``numeric``.

    Args:
        tools: The toolset resolved for the example being scored, or ``None``.
        tool_name: The tool whose arguments are being compared.
        field: The argument name.

    Returns:
        The strategy name, or ``None`` when the schema has nothing to say --
        an unresolvable toolset, an unknown tool or field, or a plain
        ``string`` with no ``enum`` or ``format``, which is exactly the
        free-text case ``auto``'s graded-similarity rung exists for.
    """
    prop = _field_schema(tools, tool_name, field)
    if prop is None:
        return None
    declared = _declared_type(prop)
    if declared in ("object", "array"):
        return "subset"
    if prop.get("enum") is not None or declared == "boolean":
        return "exact"
    if prop.get("format") in _EXACT_FORMATS:
        return "exact"
    if _ID_FIELD.search(field):
        return "exact"
    if declared in ("integer", "number"):
        return "numeric"
    return None


def _normalize(value: str) -> str:
    """Casefold and collapse whitespace so cosmetic differences compare equal."""
    return _WHITESPACE_RUN.sub(" ", value).strip().casefold()


def _is_number(value: Any) -> bool:
    """True for ints and floats but not bools — a flipped flag is a wrong value."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_subset(a: Any, b: Any) -> bool:
    """Structural subset: every leaf in ``a`` is present (== or contains) in ``b``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and _is_subset(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return all(any(_is_subset(item, x) for x in b) for item in a)
    return bool(a == b)


__all__ = ["EmbeddingsFn", "ToolArgumentsEvaluator", "ToolsetResolver"]
