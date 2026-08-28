"""End-to-end: multi-turn SDK captures → sync → run → evaluate → report.

This is the capstone test for multi-turn conversation support. It proves the
whole chain works for a 2-turn conversation recorded by the ``evalshift-sdk``
(``model_call.input`` as a messages list, ``conversation_id`` / ``turn_index``
envelope fields) plus one standalone single-turn capture with no conversation
provenance at all, with **zero** changes to the orchestrator or evaluators:

1. ``capture sync`` groups the conversation's captures by ``conversation_id``,
   orders them by ``turn_index``, and splits each turn's recorded messages
   list into a ``history`` prefix + the current turn's ``inputs`` — verbatim,
   not reconstructed, because turn 1's own capture already recorded the full
   messages list.
2. The run stage calls ``run_orchestrator(client=...)`` directly — there is
   no CLI ``--offline`` flag; ``evalshift run`` always calls a real model —
   with a ``ReplayClient`` test double (``tests/integration/replay_client.py``)
   as the injected client, so the multi-turn example dispatches through
   ``ReplayClient.complete_messages()`` (message-mode) without a real model
   call. Fixture matching alone can't distinguish "history reached the
   client" from "history was dropped": ``ReplayClient.complete_messages()``
   matches against the *last* user message only (see its module docstring),
   which for a single current turn is the same text ``ReplayClient.complete()``
   would have received had the orchestrator (incorrectly) fallen back to
   plain-prompt dispatch. So this test wraps ``ReplayClient.complete_messages``
   with a spy that records the exact ``messages`` list each call received,
   and asserts directly that turn 1's call carried all four messages — the
   recorded ``system`` / ``user`` / ``assistant`` history prefix plus the
   current turn's ``user`` message — in order, with the right role/content
   pairs. That's the load-bearing assertion: it proves the history prefix
   reached the (replayed) model call, not just that the run completed.
3. ``evaluate`` + ``report`` run against the replayed outputs (a
   deterministic ``regex`` structural evaluator — no embeddings, no judge
   model) and the HTML report surfaces the conversation: a "turn" badge on
   the multi-turn example and a collapsed "Conversation context" transcript
   on its regression.

The CLI-argument-parsing surface of ``evalshift run`` (flags, precheck,
friendly error rendering) is covered separately by
``tests/unit/test_run_command.py`` — this test's job is the pipeline, not
the command line.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift.captures.toolset import EMPTY_TOOLSET_FINGERPRINT, fingerprint_tools
from evalshift.cli.commands._suites import SUITES_MARKER_BEGIN, SUITES_MARKER_END
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.cli.main import app
from evalshift.config.loader import load_config
from evalshift.reports.html import REPORT_HTML_FILENAME
from evalshift.runner.orchestrator import run_orchestrator
from evalshift.suite.loader import load_jsonl
from tests.integration.replay_client import ReplayClient

runner = CliRunner()

SUITE_NAME = "main_chat"
CONVERSATION_ID = "conv1"

_TURN0_QUESTION = "Create new meeting for July 9th with Jeff Bezos"
_TURN0_REPLY = "What time on July 9th should I schedule that for?"
_TURN1_INPUT = "1pm"
_TURN1_REPLY = "Scheduled for July 9th at 1pm."

_SYSTEM_PROMPT = "You are a scheduling assistant."

_SOLO_INPUT = "What's the weather like?"
_SOLO_REPLY = "I don't have access to real-time weather data."


# ---------------------------------------------------------------------------
# Capture payload builders (schema-shaped exactly as evalshift-sdk writes)
# ---------------------------------------------------------------------------


# Turn 1 actually calls `add_event` (see `_turn1_capture` below), so ITS toolset must
# genuinely include it -- an empty toolset would now conflict with that recorded tool call
# once promoted (I2/Fix 1) and, separately, a sidecar whose `tools` didn't actually contain
# it would fail `load_toolset`'s fingerprint verification (Fix 2) the moment `capture
# sync`/`run` resolved it. Turn 0 and the solo capture never call anything, so they keep the
# genuinely empty toolset -- giving them add_event too would dispatch them through the
# tools-aware client path at replay time (ReplayClient.complete_with_tools, `kind: "tools"`
# fixtures) for no reason this test needs, churning fixtures below unrelated to its actual
# subject (message-history propagation, scored by a text-only regex evaluator).
_ADD_EVENT_TOOL: dict[str, Any] = {
    "name": "add_event",
    "description": "Add an event to the user's calendar.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string"},
            "time": {"type": "string"},
        },
        "required": ["title", "date"],
    },
}
_ADD_EVENT_TOOLSET_REF = fingerprint_tools([_ADD_EVENT_TOOL])


def _model_call_event(
    *, input_value: Any, output: str, toolset_ref: str, sequence_index: int = 0
) -> dict[str, Any]:
    return {
        "type": "model_call",
        "sequence_index": sequence_index,
        "timestamp": "2026-07-08T12:00:00+00:00",
        "metadata": {},
        "model_id": "m",
        "input": input_value,
        "output": output,
        "toolset_ref": toolset_ref,
        "tools_offered": ["add_event"] if toolset_ref == _ADD_EVENT_TOOLSET_REF else [],
    }


def _final_output_event(*, text: str, sequence_index: int) -> dict[str, Any]:
    return {
        "type": "final_output",
        "sequence_index": sequence_index,
        "timestamp": "2026-07-08T12:00:01+00:00",
        "metadata": {},
        "text": text,
    }


def _tool_call_event(
    *, name: str, arguments: dict[str, Any], call_id: str, sequence_index: int
) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "sequence_index": sequence_index,
        "timestamp": "2026-07-08T12:00:02+00:00",
        "metadata": {},
        "name": name,
        "arguments": arguments,
        "call_id": call_id,
    }


def _tool_result_event(
    *, name: str, call_id: str, result: dict[str, Any], sequence_index: int
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "sequence_index": sequence_index,
        "timestamp": "2026-07-08T12:00:03+00:00",
        "metadata": {},
        "name": name,
        "call_id": call_id,
        "result": result,
    }


def _turn0_capture() -> dict[str, Any]:
    """Turn 0 — no history yet, model asks a follow-up (no tool calls)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _TURN0_QUESTION},
    ]
    return {
        "schema_version": "2.0.0",
        "capture_id": "cap_conv1_t0",
        "suite": SUITE_NAME,
        "input_hash": "h0",
        "code_version": "git:deadbeef",
        "created_at": "2026-07-08T12:00:00+00:00",
        "conversation_id": CONVERSATION_ID,
        "turn_index": 0,
        "parent_capture_id": None,
        "trace": {
            "run_id": "cap_conv1_t0",
            "prompt_id": SUITE_NAME,
            "example_id": "cap_conv1_t0",
            "role": "source",
            "events": [
                _model_call_event(
                    input_value=messages,
                    output=_TURN0_REPLY,
                    toolset_ref=EMPTY_TOOLSET_FINGERPRINT,
                    sequence_index=0,
                ),
                _final_output_event(text=_TURN0_REPLY, sequence_index=1),
            ],
        },
    }


def _turn1_capture() -> dict[str, Any]:
    """Turn 1 — full messages list including turn 0, a tool call, and the reply."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _TURN0_QUESTION},
        {"role": "assistant", "content": _TURN0_REPLY},
        {"role": "user", "content": _TURN1_INPUT},
    ]
    return {
        "schema_version": "2.0.0",
        "capture_id": "cap_conv1_t1",
        "suite": SUITE_NAME,
        "input_hash": "h1",
        "code_version": "git:deadbeef",
        "created_at": "2026-07-08T12:00:10+00:00",
        "conversation_id": CONVERSATION_ID,
        "turn_index": 1,
        "parent_capture_id": "cap_conv1_t0",
        "trace": {
            "run_id": "cap_conv1_t1",
            "prompt_id": SUITE_NAME,
            "example_id": "cap_conv1_t1",
            "role": "source",
            "events": [
                _model_call_event(
                    input_value=messages,
                    output=_TURN1_REPLY,
                    toolset_ref=_ADD_EVENT_TOOLSET_REF,
                    sequence_index=0,
                ),
                _tool_call_event(
                    name="add_event",
                    arguments={
                        "title": "Meeting with Jeff Bezos",
                        "date": "2026-07-09",
                        "time": "13:00",
                    },
                    call_id="call_1",
                    sequence_index=1,
                ),
                _tool_result_event(
                    name="add_event",
                    call_id="call_1",
                    result={"status": "ok", "event_id": "evt_1"},
                    sequence_index=2,
                ),
                _final_output_event(text=_TURN1_REPLY, sequence_index=3),
            ],
        },
    }


def _solo_capture() -> dict[str, Any]:
    """A bare-string-input, single-turn capture with no conversation provenance at all."""
    return {
        "schema_version": "2.0.0",
        "capture_id": "cap_solo",
        "suite": SUITE_NAME,
        "input_hash": "hsolo",
        "code_version": "git:deadbeef",
        "created_at": "2026-07-08T11:00:00+00:00",
        "trace": {
            "run_id": "cap_solo",
            "prompt_id": SUITE_NAME,
            "example_id": "cap_solo",
            "role": "source",
            "events": [
                _model_call_event(
                    input_value=_SOLO_INPUT,
                    output=_SOLO_REPLY,
                    toolset_ref=EMPTY_TOOLSET_FINGERPRINT,
                    sequence_index=0,
                ),
                _final_output_event(text=_SOLO_REPLY, sequence_index=1),
            ],
        },
        # No conversation_id/turn_index/parent_capture_id — a standalone
        # (non-conversation) capture never carries these; the envelope must
        # still parse.
    }


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

_CONFIG = f"""
version: 1

prompts:
  - id: chat
    detection: manual
    content: "{{input}}"
    variables: [input]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  concurrency: 2
  cache: false

evaluators:
  structural:
    - type: regex
      pattern: "Scheduled"

{SUITES_MARKER_BEGIN}
suites: {{}}
{SUITES_MARKER_END}
"""


def _write_captures(tmp_path: Path) -> None:
    capture_dir = tmp_path / ".evalshift" / "captures" / SUITE_NAME
    capture_dir.mkdir(parents=True, exist_ok=True)
    for payload in (_turn0_capture(), _turn1_capture(), _solo_capture()):
        path = capture_dir / f"{payload['capture_id']}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    # build_example_from_capture refuses to promote a toolset_ref whose sidecar doesn't
    # resolve (see captures/promote.py) -- every _model_call_event above records a
    # toolset_ref (either EMPTY_TOOLSET_FINGERPRINT or _ADD_EVENT_TOOLSET_REF), so
    # `capture sync` (no --base; CWD is tmp_path via monkeypatch.chdir) needs a real sidecar
    # for each under the default .evalshift/toolsets/. `load_toolset` now also verifies each
    # sidecar's content actually fingerprints to the ref naming it (the hardening pass), so
    # both must hold their real content at their own real fingerprint -- not an arbitrary
    # placeholder.
    toolsets_dir = tmp_path / ".evalshift" / "toolsets"
    toolsets_dir.mkdir(parents=True, exist_ok=True)
    (toolsets_dir / f"{EMPTY_TOOLSET_FINGERPRINT.removeprefix('sha256:')}.json").write_text(
        '{"tools": []}', encoding="utf-8"
    )
    (toolsets_dir / f"{_ADD_EVENT_TOOLSET_REF.removeprefix('sha256:')}.json").write_text(
        json.dumps({"tools": [_ADD_EVENT_TOOL]}), encoding="utf-8"
    )


def _write_fixtures(tmp_path: Path) -> Path:
    """Fixtures for BOTH source and target models, one per (example, model).

    Each fixture ``match``es its example's own current-turn text (the last
    user message under message-mode dispatch: ``_TURN0_QUESTION`` for turn 0,
    ``_TURN1_INPUT`` ("1pm") for turn 1, ``_SOLO_INPUT`` for the solo
    capture) — all mutually distinct substrings, so there is no fixture
    collision to trip over. That history genuinely reached the model call
    (not just that turn 1's current-turn text happened to match) is proven
    separately, via a spy on ``ReplayClient.complete_messages`` — see
    ``test_conversation_capture_sync_replay_evaluate_report``.

    The target model's turn-1 reply omits "Scheduled" so the structural
    regex evaluator scores a regression (source 1.0 / target 0.0) — this is
    what puts the example into the report's "Top regressions" section,
    which is what renders the "Conversation context" transcript block.
    """
    fixtures_path = tmp_path / "fixtures.jsonl"
    records = [
        # Turn 0 — same reply on both sides (no regression on turn 0).
        {
            "model": "gemini/gemini-2.5-flash",
            "match": _TURN0_QUESTION,
            "kind": "text",
            "result": {"text": _TURN0_REPLY},
        },
        {
            "model": "gemini/gemini-2.5-pro",
            "match": _TURN0_QUESTION,
            "kind": "text",
            "result": {"text": _TURN0_REPLY},
        },
        # Turn 1 — matched on "1pm" (turn 1's own last user message). `kind: "tools"`,
        # not "text": turn 1's own toolset (_ADD_EVENT_TOOLSET_REF) is genuinely
        # non-empty (it recorded a real add_event call), so the orchestrator dispatches
        # it through complete_messages_with_tools. The replay itself makes no tool
        # calls -- calls=[] -- text-only is a perfectly ordinary response from a model
        # that was offered a tool and chose not to use it, and this test's own subject
        # (message-history propagation, scored by the text-only regex evaluator below)
        # doesn't need it to. Source replies with "Scheduled..."; target replies
        # without it, to produce a regression.
        {
            "model": "gemini/gemini-2.5-flash",
            "match": _TURN1_INPUT,
            "kind": "tools",
            "result": {"calls": [], "final_text": _TURN1_REPLY},
        },
        {
            "model": "gemini/gemini-2.5-pro",
            "match": _TURN1_INPUT,
            "kind": "tools",
            "result": {"calls": [], "final_text": "Got it, meeting noted."},
        },
        # The solo capture — single-turn, matched on the whole
        # (only) rendered prompt as usual.
        {
            "model": "gemini/gemini-2.5-flash",
            "match": _SOLO_INPUT,
            "kind": "text",
            "result": {"text": _SOLO_REPLY},
        },
        {
            "model": "gemini/gemini-2.5-pro",
            "match": _SOLO_INPUT,
            "kind": "text",
            "result": {"text": _SOLO_REPLY},
        },
    ]
    fixtures_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return fixtures_path


@pytest.mark.integration
def test_conversation_capture_sync_replay_evaluate_report(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "evalshift.yaml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    _write_captures(tmp_path)
    fixtures_path = _write_fixtures(tmp_path)

    # ---- 1. capture sync -------------------------------------------------
    sync_result = runner.invoke(app, ["capture", "sync", "--config", str(config_path)])
    assert sync_result.exit_code == 0, sync_result.stdout

    golden_path = tmp_path / ".evalshift" / "suites" / SUITE_NAME / "golden.jsonl"
    assert golden_path.exists()
    suite = load_jsonl(golden_path)
    assert suite.ids() == {"cap_conv1_t0", "cap_conv1_t1", "cap_solo"}

    turn0 = next(e for e in suite.examples if e.id == "cap_conv1_t0")
    turn1 = next(e for e in suite.examples if e.id == "cap_conv1_t1")
    solo = next(e for e in suite.examples if e.id == "cap_solo")

    # Turn 1's history is recovered VERBATIM from its own recorded messages
    # list (system, user, assistant) — not reconstructed, since the capture
    # already recorded the full prefix.
    assert turn1.conversation_id == CONVERSATION_ID
    assert turn1.turn_index == 1
    assert turn1.history is not None
    assert [m.role for m in turn1.history] == ["system", "user", "assistant"]
    assert turn1.history[0].content == _SYSTEM_PROMPT
    assert turn1.history[1].content == _TURN0_QUESTION
    assert turn1.history[2].content == _TURN0_REPLY
    assert turn1.inputs == {"input": _TURN1_INPUT}

    assert turn0.conversation_id == CONVERSATION_ID
    assert turn0.turn_index == 0

    # The solo capture promotes fine with no conversation provenance at all.
    assert solo.conversation_id is None
    assert solo.turn_index is None
    assert solo.history is None
    assert solo.inputs == {"input": _SOLO_INPUT}

    # sync must have wired the suite into evalshift.yaml between the markers.
    updated_config = config_path.read_text(encoding="utf-8")
    assert f"suites:\n  {SUITE_NAME}:" in updated_config

    # ---- 2. run, dispatched via run_orchestrator(client=...) directly -----
    # There is no CLI ``--offline`` flag any more — ``evalshift run`` always
    # calls a real model. The seam this test needs (inject a replaying
    # ModelClient) is ``run_orchestrator(client=...)`` itself
    # (``runner/orchestrator.py``; see also ``tests/unit/test_orchestrator.py``),
    # so this stage calls it directly instead of going through the CLI.
    #
    # Spy on ReplayClient.complete_messages AND complete_messages_with_tools so we can
    # assert on the *exact* messages list the (replayed) model call received — the
    # load-bearing proof that the history prefix reached the model call, not just that
    # the run completed. See the module docstring for why fixture-match behaviour alone
    # can't distinguish this. Both methods are spied into the SAME list: turn 1's
    # toolset is genuinely non-empty (it recorded a real add_event call), so the
    # orchestrator dispatches it via the tools-aware variant, while turn 0 (empty
    # toolset) still goes through the plain one — the assertions below group by
    # ``messages[-1]["content"]``, not by which method carried it, so this doesn't
    # change what's proven, only which seam it comes through.
    recorded_calls: list[dict[str, Any]] = []
    original_complete_messages = ReplayClient.complete_messages
    original_complete_messages_with_tools = ReplayClient.complete_messages_with_tools

    async def spying_complete_messages(self: ReplayClient, **kwargs: Any) -> Any:
        recorded_calls.append(kwargs)
        return await original_complete_messages(self, **kwargs)

    async def spying_complete_messages_with_tools(self: ReplayClient, **kwargs: Any) -> Any:
        recorded_calls.append(kwargs)
        return await original_complete_messages_with_tools(self, **kwargs)

    monkeypatch.setattr(ReplayClient, "complete_messages", spying_complete_messages)
    monkeypatch.setattr(
        ReplayClient, "complete_messages_with_tools", spying_complete_messages_with_tools
    )

    cfg = load_config(config_path)
    assert cfg.defaults.source_model is not None
    assert cfg.defaults.target_model is not None
    # A plain (non-async) test function, with the one async call wrapped in
    # asyncio.run() -- exactly the shape run.py's own CLI command uses. The
    # stages below dispatch through the CLI (runner.invoke), and those
    # commands (e.g. evaluate.run_evaluate) call asyncio.run() internally;
    # nesting that inside an already-running loop (an async test function,
    # or a bare await here) raises "asyncio.run() cannot be called from a
    # running event loop".
    run_result = asyncio.run(
        run_orchestrator(
            config=cfg,
            config_path=config_path,
            suite=suite,
            suite_path=golden_path,
            source_model=cfg.defaults.source_model,
            target_model=cfg.defaults.target_model,
            yes=True,
            run_slug=SUITE_NAME,
            client=ReplayClient(fixtures_path),
        ),
    )

    runs_dir = tmp_path / ".evalshift" / "runs"
    run_ids = sorted(p.name for p in runs_dir.iterdir())
    assert len(run_ids) == 1
    run_id = run_ids[0]
    assert run_id == run_result.run_id
    run_dir = runs_dir / run_id

    raw_rows = [
        json.loads(line)
        for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 3 examples x 2 roles (source, target) = 6 calls, all live (no errors).
    assert len(raw_rows) == 6
    assert all(row["error"] is None for row in raw_rows), raw_rows

    # Every dispatched call for the multi-turn examples (turn 0 and turn 1) went
    # through SOME message-mode variant (complete_messages or, for turn 1's
    # non-empty toolset, complete_messages_with_tools) — never plain complete().
    # Turn 0 has a 1-message history ([system]) and turn 1 has a 3-message
    # history ([system, user, assistant]); the solo capture has no
    # history at all and must NOT appear here.
    assert len(recorded_calls) == 4  # turn0 x2 roles + turn1 x2 roles

    turn1_calls = [c for c in recorded_calls if c["messages"][-1]["content"] == _TURN1_INPUT]
    assert len(turn1_calls) == 2  # source + target
    for call in turn1_calls:
        messages = call["messages"]
        # The load-bearing assertion: turn 1's dispatched messages list
        # carries the FULL recorded history prefix — not just the current
        # turn — in the exact recorded order and content.
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == _SYSTEM_PROMPT
        assert messages[1]["content"] == _TURN0_QUESTION
        assert messages[2]["content"] == _TURN0_REPLY
        assert messages[3]["content"] == _TURN1_INPUT

    turn0_calls = [c for c in recorded_calls if c["messages"][-1]["content"] == _TURN0_QUESTION]
    assert len(turn0_calls) == 2
    for call in turn0_calls:
        messages = call["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == _SYSTEM_PROMPT
        assert messages[1]["content"] == _TURN0_QUESTION

    # The solo capture has no history — it must go through the plain
    # complete() path, not complete_messages().
    assert not any(c["messages"][-1]["content"] == _SOLO_INPUT for c in recorded_calls)

    by_example_role = {(r["example_id"], r["role"]): r for r in raw_rows}
    assert by_example_role[("cap_conv1_t1", "source")]["text"] == _TURN1_REPLY
    assert by_example_role[("cap_conv1_t1", "target")]["text"] == "Got it, meeting noted."
    assert by_example_role[("cap_conv1_t0", "source")]["text"] == _TURN0_REPLY
    assert by_example_role[("cap_conv1_t0", "target")]["text"] == _TURN0_REPLY
    assert by_example_role[("cap_solo", "source")]["text"] == _SOLO_REPLY
    assert by_example_role[("cap_solo", "target")]["text"] == _SOLO_REPLY

    # ---- 3. evaluate + report ---------------------------------------------
    evaluate_result = runner.invoke(app, ["evaluate", run_id])
    assert evaluate_result.exit_code == 0, evaluate_result.stdout

    scores_path = run_dir / SCORES_FILENAME
    score_rows = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turn1_score = next(r for r in score_rows if r["example_id"] == "cap_conv1_t1")
    assert turn1_score["source_score"] == 1.0
    assert turn1_score["target_score"] == 0.0
    assert turn1_score["delta"] == -1.0

    analyze_result = runner.invoke(app, ["analyze", run_id])
    assert analyze_result.exit_code == 0, analyze_result.stdout

    report_result = runner.invoke(app, ["report", run_id])
    assert report_result.exit_code == 0, report_result.stdout

    report_html_path = run_dir / REPORT_HTML_FILENAME
    assert report_html_path.exists()
    html = report_html_path.read_text(encoding="utf-8")

    assert "Conversation context" in html
    assert "turn 1" in html
    assert "cap_conv1_t1" in html
