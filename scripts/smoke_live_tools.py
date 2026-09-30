"""Live smoke test for tool-aware LLM calls.

Run by hand with real provider API keys; **not** part of CI.

Usage:
    python scripts/smoke_live_tools.py

What it does, for each model whose provider key is set (others are skipped):
    1. For each prompt, call ModelClient.complete_with_tools and print a
       one-line summary (tool names + cost).
    2. Save the raw provider response to
       ``tests/unit/fixtures/tool_responses/<dir>/<prompt_name>_live.json``
       (``<dir>`` is the response shape, or ``deepseek`` for DeepSeek) and
       warn if it differs from the previous capture — early warning for
       LiteLLM upstream drift.
    3. Round 1: replay a tool round whose assistant turn the model never
       wrote (ModelClient.complete_messages_with_tools) — DeepSeek 400s this
       unless the placeholder ``reasoning_content`` is sent.
    4. History: replay a text-only chat history (user / assistant / user,
       no tools) through ModelClient.complete_messages — the placeholder
       rides on that assistant turn too.

Failures are reported per check and the script keeps going so you see the
full picture rather than bailing on the first error.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow running before `pip install -e .` finishes.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evalshift_cli.evaluators.tool_models import ToolSpec  # noqa: E402
from evalshift_cli.evaluators.tool_parser import detect_provider  # noqa: E402
from evalshift_cli.models.client import ModelClient  # noqa: E402
from evalshift_cli.models.registry import PROVIDER_ENV_VARS, resolve_model  # noqa: E402

FIXTURE_ROOT = ROOT / "tests" / "unit" / "fixtures" / "tool_responses"

TOOLS = [
    ToolSpec(
        name="search_db",
        description="Search the customer database for matching records.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="send_email",
        description="Send an email to a customer.",
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    ),
]

PROMPTS: list[tuple[str, str]] = [
    ("single_tool_call", "Find ACME Corp's most recent order."),
    (
        "parallel_tool_calls",
        "Find ACME's last 3 orders AND email a summary to billing@acme.com.",
    ),
    ("text_only", "What's our standard refund policy in one sentence?"),
]

MODELS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-3.1-flash-lite-preview",
    "deepseek-flash",
    "deepseek-v4-pro",
]


async def main() -> int:
    client = ModelClient()
    failures = 0
    for model in MODELS:
        provider = _provider_or_skip(model)
        if provider is None:
            print(f"skip {model}: no API key set for the corresponding provider")
            continue
        for prompt_name, prompt in PROMPTS:
            print(f"=== {model} / {prompt_name}")
            try:
                result = await client.complete_with_tools(
                    model=model,
                    prompt=prompt,
                    tools=TOOLS,
                )
                print(f"  calls: {result.trace.tool_names}")
                print(f"  cost: ${result.cost_usd:.6f}  latency: {result.latency_ms}ms")
                _save_fixture(provider, prompt_name, result.raw_provider_response)
            except Exception as exc:
                failures += 1
                print(f"  FAILED: {exc.__class__.__name__}: {exc}")
        try:
            await _multi_round(client, model)
        except Exception as exc:
            failures += 1
            print(f"  round 1 FAILED: {exc.__class__.__name__}: {exc}")
        try:
            await _chat_history(client, model)
        except Exception as exc:
            failures += 1
            print(f"  history FAILED: {exc.__class__.__name__}: {exc}")
    return 0 if failures == 0 else 1


def _provider_or_skip(model: str) -> str | None:
    """Return the fixture directory for ``model`` iff its provider's key is set."""
    meta = resolve_model(model)
    keys = PROVIDER_ENV_VARS.get(meta.provider, ())
    if not any(os.environ.get(k) for k in keys):
        return None
    # detect_provider names a response shape; DeepSeek shares OpenAI's, but its
    # live captures must not overwrite the OpenAI ones.
    return "deepseek" if meta.provider == "deepseek" else detect_provider(model)


async def _multi_round(client: ModelClient, model: str) -> None:
    """Round 1 of a teacher-forced replay: an assistant turn DeepSeek never wrote."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Look up ACME's Q3 revenue."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_r0_0",
                    "type": "function",
                    "function": {"name": "search_db", "arguments": '{"query": "ACME Q3"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_r0_0", "content": '{"revenue_musd": 12.4}'},
    ]
    result = await client.complete_messages_with_tools(model=model, messages=messages, tools=TOOLS)
    print(f"  round 1: calls={result.trace.tool_names} text={bool(result.trace.final_text)}")


async def _chat_history(client: ModelClient, model: str) -> None:
    """A text-only replayed history: an assistant turn with no reasoning_content."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hi, I ordered a kettle last week."},
        {"role": "assistant", "content": "Thanks! How can I help with your kettle order?"},
        {"role": "user", "content": "What's your standard refund policy, in one sentence?"},
    ]
    result = await client.complete_messages(model=model, messages=messages)
    print(f"  history: text={bool(result.text)}")


def _save_fixture(provider: str, name: str, payload: dict[str, Any]) -> None:
    target_dir = FIXTURE_ROOT / provider
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}_live.json"
    new_text = json.dumps(payload, indent=2, default=str)
    if target.exists():
        old_text = target.read_text(encoding="utf-8")
        if old_text != new_text:
            print(f"  drift: {target} changed shape vs. previous capture")
    target.write_text(new_text, encoding="utf-8")
    print(f"  saved: {target}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
