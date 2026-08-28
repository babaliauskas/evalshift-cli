"""End-to-end: SDK capture → promote → run → evaluate → verdict.

This is the Phase 9 acceptance test. It proves the full capture lifecycle works
with **zero** changes to the orchestrator or evaluators: a hand-written
SDK-shaped capture file is promoted into a golden suite, wired into
``evalshift.yaml`` via ``suites:``, and scored by ``run --suite-name`` →
``evaluate``. The promoted capture supplies the ground-truth ``expected_tools``
the ``tool_selection`` evaluator scores a candidate model against.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift.captures.toolset import fingerprint_tools
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.cli.main import app
from evalshift.evaluators.tool_models import ToolCall, ToolTrace
from evalshift.models.client import ModelClient, ToolCompletionResult
from evalshift.runner import orchestrator as orch_module

runner = CliRunner()

_QUERY = "Where is order 12345?"

_CONFIG = """
version: 1

prompts:
  - id: routing
    detection: manual
    content: "Route this: {query}"
    variables: [query]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  concurrency: 4
  cache: false

evaluators:
  tool_selection:
    - name: routing_selection
      conformance: expected
      divergence: set

suites:
  promoted:
    source: captured
    path: .evalshift/suites/support_agent/golden.jsonl
"""

# The sidecar's JSON wire shape -- what the promoted example's toolset_ref
# (below) must resolve to for the run stage to dispatch with them (Task 8:
# dispatch reads the example's own resolved toolset).
_SIDECAR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_orders",
        "description": "Search the customer database.",
        "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}},
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund.",
        "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
]
# `load_toolset` now verifies a sidecar's `tools` against the ref that named it (the
# hardening pass this test's fixture predates) -- so the capture's `toolset_ref` and the
# sidecar's filename must both be the REAL fingerprint of `_SIDECAR_TOOLS`, not an arbitrary
# placeholder.
_SIDECAR_REF = fingerprint_tools(_SIDECAR_TOOLS)


def _capture_payload() -> dict[str, Any]:
    """A capture exactly as the evalshift-sdk FileSink would write it."""
    return {
        "schema_version": "2.0.0",
        "capture_id": "cap_demo",
        "suite": "support_agent",
        "input_hash": "abc123",
        "code_version": "git:deadbeef",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": {
            "run_id": "cap_demo",
            "prompt_id": "support_agent",
            "example_id": "cap_demo",
            "role": "source",
            "events": [
                {
                    "type": "model_call",
                    "sequence_index": 0,
                    "timestamp": "2026-06-16T12:00:00+00:00",
                    "metadata": {"evalshift": {"span_id": "m1", "start_ts": 1.0, "end_ts": 1.5}},
                    "model_id": "claude-opus-4-8",
                    "input": _QUERY,
                    "output": "looking it up",
                    "toolset_ref": _SIDECAR_REF,
                    "tools_offered": ["search_orders", "issue_refund"],
                },
                {
                    "type": "tool_call",
                    "sequence_index": 1,
                    "timestamp": "2026-06-16T12:00:01+00:00",
                    "metadata": {"evalshift": {"span_id": "c1", "start_ts": 2.0, "end_ts": 2.2}},
                    "name": "search_orders",
                    "arguments": {"customer_id": "c42"},
                    "call_id": "c1",
                },
                {
                    "type": "final_output",
                    "sequence_index": 2,
                    "timestamp": "2026-06-16T12:00:02+00:00",
                    "metadata": {},
                    "text": "Your order ships tomorrow.",
                },
            ],
        },
    }


def _scaffold(tmp_path: Path) -> None:
    (tmp_path / "evalshift.yaml").write_text(_CONFIG, encoding="utf-8")
    capture_dir = tmp_path / ".evalshift" / "captures" / "support_agent"
    capture_dir.mkdir(parents=True, exist_ok=True)
    (capture_dir / "cap_demo.json").write_text(json.dumps(_capture_payload()), encoding="utf-8")
    # build_example_from_capture refuses to promote a toolset_ref whose sidecar doesn't
    # resolve (see captures/promote.py) -- the capture above records _SIDECAR_REF, so
    # `capture promote` (no --base; CWD is tmp_path via monkeypatch.chdir) needs a real
    # sidecar under the default .evalshift/toolsets/. Content must match the capture's
    # `tools_offered` (["search_orders", "issue_refund"]) rather than being an empty
    # placeholder: promotion only checks the sidecar *exists*, but the orchestrator
    # dispatches with whatever it actually contains -- an empty toolset here would route
    # the promoted example through the plain (no-tools) path, and _patch_tools below only
    # mocks complete_with_tools. `load_toolset` also now verifies this content actually
    # fingerprints to the ref naming it (the hardening pass), so the filename below must be
    # _SIDECAR_REF's own hash, not an arbitrary placeholder.
    toolsets_dir = tmp_path / ".evalshift" / "toolsets"
    toolsets_dir.mkdir(parents=True, exist_ok=True)
    (toolsets_dir / f"{_SIDECAR_REF.removeprefix('sha256:')}.json").write_text(
        json.dumps({"tools": _SIDECAR_TOOLS}),
        encoding="utf-8",
    )


def _patch_tools(monkeypatch: pytest.MonkeyPatch, *, target_tool: str) -> None:
    """Both sides call ``search_orders`` except the target calls ``target_tool``."""

    async def fake(self: ModelClient, **kwargs: Any) -> ToolCompletionResult:
        model = str(kwargs["model"])
        role = "target" if "pro" in model else "source"
        tool_name = target_tool if role == "target" else "search_orders"
        return ToolCompletionResult(
            trace=ToolTrace(
                calls=[
                    ToolCall(
                        tool_name=tool_name, arguments={"customer_id": "c42"}, sequence_index=0
                    )
                ],
                final_text=None,
            ),
            model_id=model,
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.0,
            latency_ms=10,
            raw_provider_response={},
        )

    monkeypatch.setattr(ModelClient, "complete_with_tools", fake)
    orch_module.asyncio = asyncio


def _promote_run_evaluate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    promote = runner.invoke(
        app,
        ["capture", "promote", "cap_demo", "--as", "case1", "--input-var", "query"],
    )
    assert promote.exit_code == 0, promote.stdout

    # The promoted golden suite exists and is loadable.
    golden = tmp_path / ".evalshift" / "suites" / "support_agent" / "golden.jsonl"
    assert golden.exists()

    run = runner.invoke(app, ["run", "--yes", "--suite-name", "promoted"])
    assert run.exit_code == 0, run.stdout

    runs = sorted((tmp_path / ".evalshift" / "runs").iterdir())
    assert len(runs) == 1
    run_id = runs[0].name

    evaluate = runner.invoke(app, ["evaluate", run_id])
    assert evaluate.exit_code == 0, evaluate.stdout
    return tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME


def test_promoted_capture_scores_matching_target_as_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _scaffold(tmp_path)
    _patch_tools(monkeypatch, target_tool="search_orders")

    scores_path = _promote_run_evaluate(monkeypatch, tmp_path)

    rows = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_rows = [r for r in rows if r["evaluator_name"] == "routing_selection"]
    assert tool_rows, f"no tool_selection record written; got {[r['evaluator_name'] for r in rows]}"
    record = tool_rows[0]
    assert record["example_id"] == "case1"
    # Target matched the promoted ground-truth expected_tools → perfect score.
    assert record["target_score"] == 1.0
    assert record["source_score"] == 1.0


def test_promoted_capture_detects_regressing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _scaffold(tmp_path)
    # Target calls the WRONG tool — must score below the source against the
    # capture's ground truth, i.e. a regression the verdict can see.
    _patch_tools(monkeypatch, target_tool="issue_refund")

    scores_path = _promote_run_evaluate(monkeypatch, tmp_path)

    rows = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(r for r in rows if r["evaluator_name"] == "routing_selection")
    assert record["source_score"] == 1.0
    assert record["target_score"] == 0.0
    assert record["delta"] == -1.0  # regression signal
