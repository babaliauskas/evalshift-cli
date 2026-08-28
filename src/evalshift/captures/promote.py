"""Promote a capture into a golden suite case.

Promotion turns a recorded agent run into a self-contained golden test case.
The capture supplies the **ground truth** — what tools the agent called and
what it ultimately answered — which becomes the ``expected_tools`` /
``expected`` an evaluator scores a candidate model against.

Two halves:

* :func:`build_example_from_capture` — the pure, side-effect-free mapping from
  a :class:`CaptureEnvelope` to a :class:`SuiteExample`. Easy to unit-test.
* :func:`write_promoted_case` / :func:`rebuild_golden_jsonl` — the disk side.
  Each promoted case is written as a canonical :class:`PromotedCase` file, and
  ``golden.jsonl`` is *regenerated* from every case in the suite dir so the
  dir form and the run-facing JSONL can never drift.

Known limitation: a capture stores a one-way ``input_hash``, not the raw agent
input. ``inputs`` recovery is therefore best-effort — see
:func:`_recover_inputs`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from evalshift.captures.models import CaptureEnvelope, PromotedCase
from evalshift.captures.reader import CaptureRecord, suites_root, toolset_path
from evalshift.suite.models import ChatMessage, ExpectedToolCall, HistoryToolCall, SuiteExample
from evalshift.suite.tags import CAPTURED_TAG
from evalshift.traces.models import (
    ErrorEvent,
    FinalOutputEvent,
    ModelCallEvent,
    ToolCallEvent,
    ToolResultEvent,
)

# Context-loss heuristic: warn when the capture's recorded ``input_tokens``
# imply far more context was actually sent to the model than we managed to
# recover into ``inputs``/``history`` — a strong signal the SDK caller only
# passed the current-turn string to ``capture.model_call(input=...)`` instead
# of the full messages list, so a replay would silently drop the system
# prompt / conversation history.
_CONTEXT_LOSS_RATIO = 4
_CONTEXT_LOSS_MIN_GAP_TOKENS = 300


@dataclass(frozen=True, slots=True)
class PromoteOptions:
    """Knobs controlling how a capture maps to a golden case.

    Attributes:
        name: Case id (and dir-form filename stem). Defaults to the capture id.
        input_var: Template-variable name used when the recorded model input is
            a bare string (``inputs = {input_var: <string>}``).
        tags: Extra tags appended after the automatic ``["captured", <suite>]``.
        strict_args: Use ``match_strategy="exact"`` for expected tools (default
            ``subset``).
        names_only: Drop recorded arguments — match tool *names* only.
        tool_count: Also pin ``expected_tool_count`` to the recorded count.
        allow_errored: Promote a capture whose turn recorded an ``error``
            event anyway. The case is still never given
            ``expected_no_tools=True`` — a failed turn is not evidence that
            calling nothing was correct.
        rounds: Which recorded agent rounds become ``expected_tools``.
            ``first`` (default) scopes ground truth to the first round —
            the only round a single-shot replay can reproduce. ``all``
            flattens every round, which over-specifies multi-round traces
            but matches pre-v0.3 promotion.
        tool_properties: Declared argument property names per tool, resolved
            from the recorded toolset of the capture(s) being promoted (see
            ``cli.commands.capture._declared_tool_properties``). Used only to
            recognise (and undo) a capture that recorded a wrapper function's
            parameters instead of the arguments the model passed — see
            :func:`_unwrap_recorded_arguments`. ``None`` disables the check:
            without the schema there is no evidence, and guessing at a
            recorded shape is worse than leaving it alone.
    """

    name: str | None = None
    input_var: str = "input"
    tags: tuple[str, ...] = ()
    strict_args: bool = False
    names_only: bool = False
    tool_count: bool = False
    allow_errored: bool = False
    rounds: Literal["first", "all"] = "first"
    tool_properties: Mapping[str, frozenset[str]] | None = None


@dataclass(frozen=True, slots=True)
class BuiltExample:
    """A promoted :class:`SuiteExample` plus any best-effort warnings.

    Attributes:
        example: The mapped case.
        warnings: Non-fatal notes for the operator.
        blocked: When set, the reason this capture should not be promoted.
            Callers must skip it unless the operator opted in explicitly —
            a capture whose turn failed records what the agent *couldn't*
            do, which is the opposite of ground truth.
        blocked_reason: The *kind* of refusal ``blocked`` describes, for
            callers that report an aggregate summary and need to route each
            refusal to an accurate counter/message rather than lump every
            refusal under one (see ``cli/commands/capture.py``'s
            ``capture sync`` summary). ``"errored"`` is the one reason
            ``--allow-errored`` can rescue; ``"no_toolset"`` covers both a
            capture missing ``toolset_ref`` entirely and one whose
            ``toolset_ref`` has no resolvable sidecar; ``"multi_toolset"``
            covers a capture whose ``model_call`` events recorded more than
            one distinct ``toolset_ref`` (the SDK stamps one per call, so a
            runtime toolset switch is legitimate — but ``SuiteExample``
            carries exactly one toolset for its whole trace, so such a
            capture cannot yet be promoted as a single case). Neither of the
            latter two is rescued by ``--allow-errored``: re-capturing fixes
            the first but reproduces the second identically. ``None`` when
            ``blocked`` is ``None``.
    """

    example: SuiteExample
    warnings: list[str] = field(default_factory=list)
    blocked: str | None = None
    blocked_reason: Literal["errored", "no_toolset", "multi_toolset"] | None = None


@dataclass(frozen=True, slots=True)
class RecoveredInput:
    """Result of recovering ``inputs`` (+ optional ``history``) from a trace.

    Attributes:
        inputs: Template-variable values for the current turn.
        history: The conversation prefix recorded before the current turn,
            when the capture's first ``model_call.input`` was a full messages
            list. ``None`` when the recorded input was a bare dict/string (no
            history to recover).
    """

    inputs: dict[str, Any]
    history: list[ChatMessage] | None = None


def build_example_from_capture(
    envelope: CaptureEnvelope,
    opts: PromoteOptions,
    *,
    base: Path | None = None,
) -> BuiltExample:
    """Map a capture to a :class:`SuiteExample`.

    Almost pure: the only disk access is a single ``Path.is_file()`` check
    (see :func:`~evalshift.captures.reader.toolset_path`) confirming the
    recorded ``toolset_ref``'s sidecar actually exists, so a capture whose
    sidecar was deleted (or never written) is refused here rather than
    promoting cleanly and crashing later, the first time something tries to
    resolve it. It does not read the sidecar's contents — promotion carries
    the *reference*, not the tool bodies.

    Args:
        envelope: The capture to map.
        opts: Promotion knobs.
        base: Capture base dir the sidecar existence check resolves
            against. ``None`` resolves via
            :func:`~evalshift.captures.reader.capture_base`.
    """
    events = envelope.trace.events
    warnings: list[str] = []

    recovered = _recover_inputs(events, input_var=opts.input_var, warnings=warnings)
    loss = _context_loss_warning(events, recovered.inputs, recovered.history)
    if loss is not None:
        warnings.append(loss)
    tags = _build_tags(envelope.suite, opts.tags)
    expected = _recover_expected(events, warnings)
    tool_rounds = _tool_rounds(events)
    scoped_calls = (
        [call for round_ in tool_rounds for call in round_]
        if opts.rounds == "all"
        else (tool_rounds[0] if tool_rounds else [])
    )

    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    toolset_ref = first_model_call.toolset_ref if first_model_call is not None else None
    tools_offered = first_model_call.tools_offered if first_model_call is not None else None

    # Every distinct toolset_ref the capture's model_call events recorded, in
    # first-seen (round) order. The SDK stamps toolset_ref per call, so more
    # than one here means the agent switched toolsets mid-run -- legitimate,
    # not corruption (see the module docstring's "several distinct
    # toolset_ref values" note) -- but SuiteExample has only one toolset
    # field for the whole example (I1 of the final review).
    distinct_toolset_refs = list(
        dict.fromkeys(
            e.toolset_ref
            for e in events
            if isinstance(e, ModelCallEvent) and e.toolset_ref is not None
        ),
    )

    errors = _error_events(events)
    blocked: str | None = None
    blocked_reason: Literal["errored", "no_toolset", "multi_toolset"] | None = None
    if toolset_ref is None:
        # Unconditional -- unlike an errored turn, --allow-errored cannot
        # rescue this: there is no recorded toolset to promote at all, only
        # re-capturing with an SDK that stamps toolset_ref produces one.
        blocked = (
            f"capture {envelope.capture_id!r} has no toolset_ref on its first model call — "
            "the SDK did not record which tools were offered (or failed to write the "
            "sidecar). Re-capture with a current evalshift-sdk to promote it."
        )
        blocked_reason = "no_toolset"
    elif not (sidecar_path := toolset_path(toolset_ref, base=base)).is_file():
        # Also unconditional, same reasoning as the missing-ref case above:
        # the SDK *did* record a toolset_ref, but its sidecar isn't on disk
        # (deleted by `capture clean`, moved, or never written) -- there is
        # no toolset to carry, existence-only, deliberately not a full
        # `load_toolset` resolution (promotion never needs the tool bodies,
        # and resolving every capture's sidecar here would cost every
        # existing promote fixture a real file on disk for no benefit).
        blocked = (
            f"capture {envelope.capture_id!r} references toolset_ref {toolset_ref!r}, but its "
            f"sidecar is missing ({sidecar_path}) — it may have been deleted, moved by "
            "`capture clean`, or never written. Re-capture with a current evalshift-sdk to "
            "promote it."
        )
        blocked_reason = "no_toolset"
    elif len(distinct_toolset_refs) > 1:
        # Unconditional too, and independent of opts.rounds: expected_tool_rounds
        # (populated below) always retains every round regardless of --rounds, so
        # even the default --rounds first would carry a later round's calls
        # against a toolset the example's single toolset_ref doesn't name.
        refs_list = ", ".join(repr(r) for r in distinct_toolset_refs)
        blocked = (
            f"capture {envelope.capture_id!r} recorded {len(distinct_toolset_refs)} distinct "
            f"toolsets across its model_call events ({refs_list}) — a promoted example carries "
            "exactly one toolset for its whole trace, so a capture whose agent switched "
            "toolsets mid-run cannot yet be promoted as a single case. Promote or hand-edit "
            "each toolset's rounds separately."
        )
        blocked_reason = "multi_toolset"
    elif errors:
        detail = errors[0].message.strip().replace("\n", " ")[:160]
        note = (
            f"capture recorded {len(errors)} error event(s) — the turn failed before the agent "
            f"finished, so it is not ground truth. First error: {detail}"
        )
        if opts.allow_errored:
            warnings.append(note)
        else:
            blocked = note
            blocked_reason = "errored"

    failed_results = _failed_tool_results(events)
    if failed_results:
        names = ", ".join(sorted(set(failed_results)))
        warnings.append(
            f"{len(failed_results)} recorded tool result(s) failed ({names}) — this turn is "
            "promoted as ground truth including the failing call(s). Review whether that is "
            "the behaviour you want a candidate model to reproduce.",
        )

    if _has_no_scoreable_ground_truth(events):
        warnings.append(
            "turn has no scoreable ground truth (empty recorded output, no tool calls); "
            "case will only exercise structural checks",
        )

    if opts.rounds == "first" and len(tool_rounds) > 1:
        dropped = sum(len(r) for r in tool_rounds[1:])
        warnings.append(
            f"capture spans {len(tool_rounds)} agent round(s); expected_tools was scoped to "
            f"round 1 ({len(tool_rounds[0])} call(s)) and {dropped} later call(s) were moved to "
            "expected_tool_rounds. A single-shot replay cannot reach round 2 — promote with "
            "--rounds all only if you score the flattened trace deliberately.",
        )

    match_strategy: Literal["exact", "subset", "contains_per_field"] = (
        "exact" if opts.strict_args else "subset"
    )

    unwrapped_keys: set[str] = set()

    def _expected_call(call: ToolCallEvent) -> ExpectedToolCall:
        arguments: dict[str, Any] | None = None
        if not opts.names_only:
            arguments, wrapper = _unwrap_recorded_arguments(
                call.name,
                dict(call.arguments),
                opts.tool_properties,
            )
            if wrapper is not None:
                unwrapped_keys.add(wrapper)
        return ExpectedToolCall(
            tool_name=call.name,
            arguments=arguments,
            match_strategy=match_strategy,
            # Stated at the write site, not left to the field default: these
            # arguments are a transcript of what the source model did, which
            # nobody has yet confirmed is what it *should* have done. A human
            # who checks a row flips it to `reviewed`.
            provenance="captured",
        )

    if scoped_calls:
        expected_tools: list[ExpectedToolCall] | None = [_expected_call(c) for c in scoped_calls]
        expected_no_tools = False
    else:
        expected_tools = None
        # expected_no_tools asserts "tools were offered and none were
        # called" -- it must never fire when nothing was offered (there is
        # nothing a candidate model could have failed to call, and
        # _score_no_tools would award 1.0/1.0 to an unmeasurable row), and a
        # turn that errored produced no tool calls because it never ran, not
        # because calling nothing was right.
        expected_no_tools = bool(tools_offered) and not errors
    expected_tool_rounds = (
        [[_expected_call(c) for c in r] for r in tool_rounds] if tool_rounds else None
    )
    if unwrapped_keys:
        keys = ", ".join(sorted(unwrapped_keys))
        warnings.append(
            f"recorded tool arguments were wrapped in {{{keys}}} — the capture recorded the "
            "wrapper function's parameters, not the arguments the model passed. The declared "
            "tool schema confirms the inner shape, so expected_tools was unwrapped to match "
            "what a model can actually produce.",
        )

    generation_config: dict[str, Any] | None = None
    if first_model_call is not None:
        raw = first_model_call.metadata.get("generation_config")
        if isinstance(raw, dict) and raw:
            generation_config = raw

    # SuiteExample requires exactly one of toolset_ref/tools. A blocked
    # capture with no toolset_ref at all (see the first blocking condition
    # above) has no real toolset to carry, but .example must still construct
    # -- callers gate on .blocked, not on catching a ValidationError -- so it
    # gets the inert tools=[] placeholder every other "we don't know" site in
    # this codebase uses (see cli/commands/evaluate.py's _score_one_tool).
    # SuiteExample now rejects tools=[] paired with tool-call ground truth
    # (I2): scoped_calls/tool_rounds above were computed from the trace
    # regardless of whether a toolset_ref exists to dispatch them against
    # (e.g. a tool_call recorded with no preceding model_call at all still
    # opens a defensive round -- see _tool_rounds), so the placeholder must
    # drop any ground truth here too to stay genuinely inert. A blocked
    # capture's .example is never scored, so nothing of substance is lost.
    no_real_toolset = toolset_ref is None
    example = SuiteExample(
        id=opts.name or envelope.capture_id,
        inputs=recovered.inputs,
        tags=tags,
        expected=expected,
        expected_tools=None if no_real_toolset else expected_tools,
        expected_tool_rounds=None if no_real_toolset else expected_tool_rounds,
        expected_tool_count=(
            None if no_real_toolset else (len(scoped_calls) if opts.tool_count else None)
        ),
        expected_no_tools=expected_no_tools,
        history=recovered.history,
        conversation_id=envelope.conversation_id,
        turn_index=envelope.turn_index,
        generation_config=generation_config,
        toolset_ref=toolset_ref,
        tools=[] if no_real_toolset else None,
    )
    return BuiltExample(
        example=example,
        warnings=warnings,
        blocked=blocked,
        blocked_reason=blocked_reason,
    )


def _unwrap_recorded_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tool_properties: Mapping[str, frozenset[str]] | None,
) -> tuple[dict[str, Any], str | None]:
    """Undo a single-dict wrapper around recorded tool arguments.

    A capture SDK that decorates a Python function records that *function's*
    parameters. When the function takes one dict — ``def archive_project(
    tool_args: dict)`` — every call is recorded as
    ``{"tool_args": {"project_name": ...}}``, while the model only ever saw the
    declared schema's flat properties. No model can produce the recorded shape,
    so ground truth in it scores 0 against every candidate.

    Unwrapping requires the declared schema to *confirm* the shape on both
    sides: the wrapper key must not be a declared property, and the inner keys
    must all be declared ones. Without that evidence the recording is returned
    untouched — a wrong guess here silently rewrites ground truth.

    Args:
        tool_name: Name of the tool the arguments belong to.
        arguments: The arguments exactly as recorded.
        tool_properties: Declared property names per tool, or ``None``.

    Returns:
        ``(arguments, wrapper_key)`` — the arguments to use, and the wrapper
        key that was removed, or ``None`` when nothing was unwrapped.
    """
    if tool_properties is None or len(arguments) != 1:
        return arguments, None
    declared = tool_properties.get(tool_name)
    if not declared:
        return arguments, None

    (key,), (value,) = tuple(arguments.keys()), tuple(arguments.values())
    if key in declared or not isinstance(value, dict):
        return arguments, None
    # An empty wrapper is the no-argument call — vacuously consistent with the
    # schema, and the shape this bug most often takes.
    if value and not set(value).issubset(declared):
        return arguments, None
    return dict(value), key


def _error_events(events: list[Any]) -> list[ErrorEvent]:
    """Error events recorded during the turn, in sequence order."""
    return [e for e in events if isinstance(e, ErrorEvent)]


def _failed_tool_results(events: list[Any]) -> list[str]:
    """Names of tools whose recorded result signals failure.

    Recognises an explicit ``error`` on the event and the common
    ``{"success": false}`` convention in a dict result. Anything else is
    opaque to us and is left alone — a tool's result schema is the app's,
    not ours.
    """
    failed: list[str] = []
    for event in events:
        if not isinstance(event, ToolResultEvent):
            continue
        if event.error or (isinstance(event.result, dict) and event.result.get("success") is False):
            failed.append(event.name)
    return failed


def _has_no_scoreable_ground_truth(events: list[Any]) -> bool:
    """True when a turn recorded no final output, no non-empty model output, no tool calls."""
    if any(isinstance(e, ToolCallEvent) for e in events):
        return False
    if any(isinstance(e, FinalOutputEvent) and e.text for e in events):
        return False
    return not any(isinstance(e, ModelCallEvent) and _nonempty_text(e.output) for e in events)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _tool_rounds(events: list[Any]) -> list[list[ToolCallEvent]]:
    """Group tool calls into agent rounds, splitting at each ``model_call``.

    A capture's events are sequence-ordered: a ``model_call``, then the tool
    calls that model call emitted, then the next ``model_call``, and so on.
    Each ``model_call`` therefore opens a round. Rounds that emitted no tool
    call (typically the final, text-producing round) are dropped, so
    ``rounds[0]`` is always the first round that actually called something.

    Args:
        events: The capture's trace events, in sequence order.

    Returns:
        One list of :class:`ToolCallEvent` per tool-emitting round, in order.
        Empty when the capture called no tools at all.
    """
    rounds: list[list[ToolCallEvent]] = []
    for event in events:
        if isinstance(event, ModelCallEvent):
            rounds.append([])
        elif isinstance(event, ToolCallEvent):
            if not rounds:
                # Defensive: a trace whose first tool call precedes any
                # model_call still gets a round to live in.
                rounds.append([])
            rounds[-1].append(event)
    return [r for r in rounds if r]


def _coerce_history_role(role: Any) -> Literal["system", "user", "assistant", "tool"] | None:
    """Map a recorded message role to a supported history role, or None to drop it."""
    if role == "model":
        return "assistant"
    if role == "system":
        return "system"
    if role == "user":
        return "user"
    if role == "assistant":
        return "assistant"
    if role == "tool":
        return "tool"
    return None


def _looks_like_messages_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in value)
    )


def _recover_inputs(
    events: list[Any],
    *,
    input_var: str,
    warnings: list[str],
) -> RecoveredInput:
    """Best-effort recovery of template-variable ``inputs`` (+ ``history``) from the trace.

    The capture only stores a one-way ``input_hash``, not the raw args, so we
    fall back to the first ``model_call`` input:

    * A dict is used directly as ``inputs`` (no history).
    * A messages list (``[{"role": ..., "content": ...}, ...]``) is split:
      the last ``user`` message becomes the current-turn ``inputs``, and
      everything before it becomes ``history``. ``model`` roles are coerced
      to ``assistant``; unsupported roles (e.g. ``tool``) are dropped with an
      aggregated warning.
    * A non-empty string is wrapped as ``{input_var: string}`` (no history).
    * Anything else yields empty ``inputs`` with a warning.
    """
    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    if first_model_call is None:
        warnings.append(
            "no model_call event found — inputs left empty; edit the case before running.",
        )
        return RecoveredInput(inputs={})

    value = first_model_call.input

    if _looks_like_messages_list(value):
        return _recover_inputs_from_messages(value, input_var=input_var, warnings=warnings)

    if isinstance(value, dict):
        return RecoveredInput(inputs=dict(value))
    if isinstance(value, str) and value:
        return RecoveredInput(inputs={input_var: value})

    warnings.append(
        "could not recover structured inputs from the capture "
        f"(model_call input was {type(value).__name__}); "
        "inputs left empty; edit the case before running.",
    )
    return RecoveredInput(inputs={})


def _history_message(
    msg: dict[str, Any],
    role: Literal["system", "user", "assistant", "tool"],
) -> ChatMessage:
    """Build one history entry from a recorded message dict.

    Recorded ``tool_calls`` are carried through on ``assistant`` turns and
    ``tool_call_id`` on ``tool`` turns, so a replay sees the same agent loop
    production saw.
    """
    raw_calls = msg.get("tool_calls") if role == "assistant" else None
    tool_calls: list[HistoryToolCall] | None = None
    if isinstance(raw_calls, list) and raw_calls:
        tool_calls = [
            HistoryToolCall(
                id=(str(c["id"]) if isinstance(c, dict) and c.get("id") is not None else None),
                name=str(c.get("name", "")) if isinstance(c, dict) else "",
                arguments=dict(c.get("arguments") or {}) if isinstance(c, dict) else {},
            )
            for c in raw_calls
        ]
    tool_call_id = str(msg["tool_call_id"]) if role == "tool" and msg.get("tool_call_id") else None
    return ChatMessage(
        role=role,
        content=str(msg.get("content", "")),
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def _recover_inputs_from_messages(
    messages: list[dict[str, Any]],
    *,
    input_var: str,
    warnings: list[str],
) -> RecoveredInput:
    last_user_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    if last_user_idx is None:
        warnings.append(
            "could not recover structured inputs from the capture "
            "(messages list had no user message); "
            "inputs left empty; edit the case before running.",
        )
        return RecoveredInput(inputs={})

    current = messages[last_user_idx]
    prefix = messages[:last_user_idx]

    history: list[ChatMessage] = []
    dropped_roles: set[str] = set()
    dropped_count = 0
    unpaired_tool_results = 0
    for position, msg in enumerate(prefix):
        role = _coerce_history_role(msg.get("role"))
        if role is None:
            dropped_count += 1
            dropped_roles.add(str(msg.get("role")))
            continue
        if role == "tool" and not msg.get("tool_call_id"):
            # Synthesise an id so the strict model validates; a recording that
            # lost the provider call id is a gap worth naming, not a reason to
            # delete the tool result from the replayed context.
            msg = {**msg, "tool_call_id": f"_pos{position}"}
            unpaired_tool_results += 1
        history.append(_history_message(msg, role))

    if dropped_count:
        roles = ", ".join(sorted(dropped_roles))
        warnings.append(
            f"dropped {dropped_count} history message(s) with unrecognised role(s) {{{roles}}}",
        )
    if unpaired_tool_results:
        warnings.append(
            f"{unpaired_tool_results} tool result(s) in history had no tool_call_id; "
            "synthetic ids were assigned. Record the provider call id for exact pairing.",
        )

    return RecoveredInput(
        inputs={input_var: str(current.get("content", ""))},
        history=history,
    )


def _context_loss_warning(
    events: list[Any],
    inputs: dict[str, Any],
    history: list[ChatMessage] | None,
) -> str | None:
    """Warning when recorded ``input_tokens`` imply far more context than recovered, else None.

    Returned (not appended) so conversation grouping can re-evaluate the check
    after reconstruction supplies a history prefix the per-capture pass
    couldn't see.
    """
    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    if first_model_call is None or first_model_call.input_tokens == 0:
        return None

    input_tokens = first_model_call.input_tokens
    recovered_chars = len(json.dumps(inputs, default=str)) + sum(
        len(m.content) for m in history or []
    )
    est_tokens = max(1, recovered_chars // 4)

    if (
        input_tokens >= _CONTEXT_LOSS_RATIO * est_tokens
        and input_tokens - est_tokens > _CONTEXT_LOSS_MIN_GAP_TOKENS
    ):
        return (
            f"capture recorded {input_tokens} input tokens but only ~{est_tokens} were "
            "recovered — the system prompt / conversation history was probably not "
            "captured. Pass the full messages list to capture.model_call(input=...)."
        )
    return None


def _build_tags(suite: str, extra: tuple[str, ...]) -> list[str]:
    tags = [CAPTURED_TAG, suite]
    for tag in extra:
        if tag not in tags:
            tags.append(tag)
    return tags


def _recover_expected(events: list[Any], warnings: list[str]) -> dict[str, Any] | None:
    """Recover the turn's user-visible reply as text ground truth.

    A ``FinalOutputEvent`` is the explicit signal and always wins. The manual
    SDK API does not emit one (only the LangChain adapter does), so fall back
    to the last ``model_call`` whose ``output`` is non-empty text — on an agent
    turn that is the reply the user saw, after the tool round-trips. ``output``
    is typed ``Any``; anything that is not a ``str`` is left alone rather than
    stringified into fake ground truth.
    """
    finals = [e for e in events if isinstance(e, FinalOutputEvent)]
    if finals:
        return {"final_output": finals[-1].text}
    for event in reversed(events):
        if isinstance(event, ModelCallEvent) and isinstance(event.output, str):
            text = event.output.strip()
            if text:
                return {"final_output": text}
    warnings.append(
        "no text ground truth: this capture has no final_output event and no "
        "model_call with text output, so example.expected is null. Text "
        "evaluators compare the two candidate models to each other, never to "
        "what production actually said.",
    )
    return None


# ---------------------------------------------------------------------------
# Conversation grouping
# ---------------------------------------------------------------------------


def duplicate_turn_warnings(envelopes: list[CaptureEnvelope]) -> list[str]:
    """Warn when two captures claim the same conversation turn.

    A retried turn produces two captures with identical
    ``(conversation_id, turn_index)`` — typically one that errored and one
    that succeeded. Both get promoted, both appear in the suite, and the
    reconstruction order between them is arbitrary. Surface it so the
    operator can drop one.

    Args:
        envelopes: Every capture being synced, in any order.

    Returns:
        One warning per colliding ``(conversation_id, turn_index)`` pair.
    """
    seen: dict[tuple[str, int], list[str]] = {}
    for env in envelopes:
        if env.conversation_id is None or env.turn_index is None:
            continue
        seen.setdefault((env.conversation_id, env.turn_index), []).append(env.capture_id)

    out: list[str] = []
    for (conversation_id, turn_index), ids in sorted(seen.items()):
        if len(ids) < 2:
            continue
        joined = ", ".join(sorted(ids))
        out.append(
            f"conversation {conversation_id} turn {turn_index} is claimed by "
            f"{len(ids)} captures ({joined}) — usually a retried turn. Both will be promoted "
            "and their reconstruction order is arbitrary; drop the one you don't want.",
        )
    return out


def _turn_sort_key(record: CaptureRecord) -> tuple[int, int, str, str]:
    """Sort key ordering group members by turn_index (None last), then created_at, then id.

    The leading int flags a ``None`` turn_index so those members always sort
    after every explicitly-indexed member, regardless of their numeric value.
    """
    turn_index = record.envelope.turn_index
    has_index = 0 if turn_index is not None else 1
    return (
        has_index,
        turn_index if turn_index is not None else 0,
        record.envelope.created_at,
        record.envelope.capture_id,
    )


def _turn_current_text(built: BuiltExample, opts: PromoteOptions) -> str:
    """Best-effort text for the current turn, for reconstructing later turns' history."""
    value = built.example.inputs.get(opts.input_var)
    if isinstance(value, str):
        return value
    return json.dumps(built.example.inputs, default=str)


def _turn_assistant_text(record: CaptureRecord) -> str | None:
    """The text a reconstructed history entry should use for a prior turn's reply.

    Prefers the last :class:`FinalOutputEvent`; falls back to the last
    non-empty :class:`ModelCallEvent` output. ``None`` when neither exists.
    """
    events = record.envelope.trace.events
    finals = [e for e in events if isinstance(e, FinalOutputEvent)]
    if finals:
        return finals[-1].text
    model_calls = [e for e in events if isinstance(e, ModelCallEvent) and _nonempty_text(e.output)]
    if model_calls:
        return str(model_calls[-1].output)
    return None


def _earliest_prior_system_message(
    prior: list[tuple[CaptureRecord, BuiltExample]],
    *,
    warnings: list[str],
) -> ChatMessage | None:
    """The earliest prior turn's recovered system message, if any turn has one.

    A prior turn carries a system message when its own capture recorded a full
    messages list (or its history was itself seeded by an earlier turn). The
    earliest turn's wins; a single aggregated warning is emitted when later
    turns recorded a different system prompt.
    """
    found: ChatMessage | None = None
    found_capture_id: str | None = None
    mismatched: list[str] = []
    for prior_record, prior_built in prior:
        hist = prior_built.example.history
        if not hist or hist[0].role != "system":
            continue
        if found is None:
            found = hist[0]
            found_capture_id = prior_record.envelope.capture_id
        elif hist[0].content != found.content:
            mismatched.append(prior_record.envelope.capture_id)
    if mismatched:
        warnings.append(
            f"prior turns recorded different system prompts — using the earliest turn's "
            f"({found_capture_id!r}); differing: {', '.join(repr(c) for c in mismatched)}",
        )
    return found


def _reconstruct_history(
    prior: list[tuple[CaptureRecord, BuiltExample]],
    opts: PromoteOptions,
    *,
    warnings: list[str],
) -> list[ChatMessage]:
    """Build a conversation prefix from earlier turns in a group.

    The prefix is seeded with the earliest prior turn's recovered system
    message, when one exists (see :func:`_earliest_prior_system_message`).
    Each prior turn then contributes a ``user`` message (its recovered
    current-turn text) and an ``assistant`` message (its final/model-call
    output). A prior turn with no recoverable assistant text contributes only
    the ``user`` message, with a warning.
    """
    history: list[ChatMessage] = []
    system = _earliest_prior_system_message(prior, warnings=warnings)
    if system is not None:
        history.append(system)
    for prior_record, prior_built in prior:
        history.append(ChatMessage(role="user", content=_turn_current_text(prior_built, opts)))
        assistant_text = _turn_assistant_text(prior_record)
        if assistant_text is None:
            warnings.append(
                f"conversation {prior_record.envelope.conversation_id!r} turn "
                f"{prior_record.envelope.capture_id!r} has no recoverable assistant reply — "
                "reconstructed history is missing its assistant message for that turn",
            )
            continue
        history.append(ChatMessage(role="assistant", content=assistant_text))
    return history


def build_conversation_examples(
    records: list[CaptureRecord], opts: PromoteOptions, *, base: Path | None = None
) -> list[tuple[CaptureRecord, BuiltExample]]:
    """Build one :class:`SuiteExample` per capture, grouping multi-turn conversations.

    Records sharing a non-``None`` ``envelope.conversation_id`` are grouped and
    ordered by ``turn_index`` (missing indices sort last, tie-broken by
    ``created_at`` then ``capture_id``). Each turn's history is either kept
    verbatim (when the capture itself recorded a full messages list — the
    source of truth) or reconstructed from the group's prior turns' recovered
    text and replies.

    Records with ``conversation_id is None`` are built independently via
    :func:`build_example_from_capture`, unaffected by grouping.

    Args:
        records: Captures to build, in any order.
        opts: Promotion knobs, shared across every record.
        base: Capture base dir threaded to :func:`build_example_from_capture`
            for its toolset-sidecar existence check. ``None`` resolves via
            :func:`~evalshift.captures.reader.capture_base`.

    Returns:
        ``(record, built)`` pairs, one per input record — order follows group
        turn order for conversation members, and input order for standalone
        records interleaved with the groups' first appearance.
    """
    groups: dict[str, list[CaptureRecord]] = {}
    order: list[str] = []
    standalone: list[CaptureRecord] = []

    for record in records:
        conv_id = record.envelope.conversation_id
        if conv_id is None:
            standalone.append(record)
            continue
        if conv_id not in groups:
            groups[conv_id] = []
            order.append(conv_id)
        groups[conv_id].append(record)

    results: list[tuple[CaptureRecord, BuiltExample]] = []

    for record in standalone:
        results.append((record, build_example_from_capture(record.envelope, opts, base=base)))

    for conv_id in order:
        members = sorted(groups[conv_id], key=_turn_sort_key)
        built_so_far: list[tuple[CaptureRecord, BuiltExample]] = []
        for ordinal, record in enumerate(members):
            built = build_example_from_capture(record.envelope, opts, base=base)
            example = built.example
            warnings = list(built.warnings)

            if example.history is None:
                history = _reconstruct_history(built_so_far, opts, warnings=warnings)
                # The per-capture context-loss check ran without this prefix;
                # re-evaluate it against what reconstruction just supplied.
                events = record.envelope.trace.events
                stale = _context_loss_warning(events, example.inputs, None)
                if stale is not None and stale in warnings:
                    warnings.remove(stale)
                fresh = _context_loss_warning(events, example.inputs, history)
                if fresh is not None:
                    warnings.append(fresh)
                example = example.model_copy(update={"history": history})

            turn_index = (
                record.envelope.turn_index if record.envelope.turn_index is not None else ordinal
            )
            example = example.model_copy(
                update={"conversation_id": conv_id, "turn_index": turn_index},
            )

            final_built = BuiltExample(
                example=example,
                warnings=warnings,
                blocked=built.blocked,
                blocked_reason=built.blocked_reason,
            )
            # A blocked turn is never promoted, so it must not seed later
            # turns' history either: a turn that died before the agent acted
            # contributes a user message with no reply, and a retried turn
            # would inject the same user text twice.
            if final_built.blocked is None:
                built_so_far.append((record, final_built))
            results.append((record, final_built))

    return results


# ---------------------------------------------------------------------------
# Disk side
# ---------------------------------------------------------------------------

GOLDEN_FILENAME = "golden.jsonl"


def _safe_segment(value: str) -> str:
    """Sanitise a user-supplied path segment (suite / case name).

    Strips path separators and parent refs so a crafted ``--suite`` or
    ``--as`` can't escape the suites directory.
    """
    cleaned = value.replace("/", "_").replace("\\", "_").replace("..", "_").strip()
    return cleaned or "_"


def promoted_suite_dir(suite: str, *, base: Path | None = None) -> Path:
    """Return the promoted-suite directory ``<base>/suites/<suite>`` for ``suite``.

    Path segments are sanitised the same way :func:`write_promoted_case` writes
    them, so callers (e.g. ``capture sync``) can locate a suite's dir without
    re-deriving the layout.
    """
    return suites_root(base) / _safe_segment(suite)


def write_promoted_case(
    case: PromotedCase,
    *,
    base: Path | None = None,
    force: bool = False,
) -> Path:
    """Write ``case`` to ``<base>/suites/<suite>/<name>.json``.

    Raises:
        FileExistsError: if the target exists and ``force`` is False.
    """
    suite_dir = promoted_suite_dir(case.suite, base=base)
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{_safe_segment(case.name)}.json"
    if path.exists() and not force:
        raise FileExistsError(path)
    path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def example_content_key(example: SuiteExample) -> str:
    """A stable key over the *replayed content* of ``example`` (inputs + history).

    Two examples sharing this key replay identically, so promoting both would
    duplicate the case, double ``n``, and inflate every downstream p-value and
    effect size. Deliberately **not** the capture's ``input_hash``: the SDK
    salts that with ``conversation_id`` (and derives it from the agent's bound
    arguments), so identical replay content routinely carries distinct hashes.
    """
    return json.dumps(
        {
            "inputs": example.inputs,
            "history": [m.model_dump() for m in example.history or []],
        },
        sort_keys=True,
        default=str,
    )


def iter_promoted_cases(suite_dir: Path) -> list[PromotedCase]:
    """Load every promoted case file in ``suite_dir`` (``golden.jsonl`` excluded).

    Raises:
        pydantic.ValidationError: if a case file is not a valid ``PromotedCase``.
    """
    cases: list[PromotedCase] = []
    for path in sorted(suite_dir.glob("*.json")):
        if path.name == GOLDEN_FILENAME:
            continue
        cases.append(PromotedCase.model_validate_json(path.read_text(encoding="utf-8")))
    return cases


def rebuild_golden_jsonl(suite_dir: Path) -> Path:
    """Regenerate ``<suite_dir>/golden.jsonl`` from every promoted case file.

    The JSONL is the run-facing index: one ``SuiteExample`` per line, sorted by
    id for determinism. Regenerated (never appended) so it can't drift from the
    canonical dir-form case files.
    """
    cases = iter_promoted_cases(suite_dir)
    cases.sort(key=lambda c: c.example.id)
    golden_path = suite_dir / GOLDEN_FILENAME
    golden_path.write_text(
        "".join(c.example.model_dump_json() + "\n" for c in cases),
        encoding="utf-8",
    )
    return golden_path


__all__ = [
    "GOLDEN_FILENAME",
    "BuiltExample",
    "PromoteOptions",
    "RecoveredInput",
    "build_conversation_examples",
    "build_example_from_capture",
    "duplicate_turn_warnings",
    "example_content_key",
    "iter_promoted_cases",
    "promoted_suite_dir",
    "rebuild_golden_jsonl",
    "write_promoted_case",
]
