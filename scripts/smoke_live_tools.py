"""Live smoke test for tool-aware LLM calls.

Run by hand with real provider API keys; **not** part of CI.

Usage:
    python scripts/smoke_live_tools.py

What it does:
    1. For each (model, prompt) combination, call ModelClient.complete_with_tools.
    2. Print a one-line summary (tool names + cost).
    3. Save the raw provider response to
       ``tests/unit/fixtures/tool_responses/<provider>/<prompt_name>.json``
       so unit-test fixtures stay current.
    4. Diff against the existing fixture (if any) and warn if the shape
       changed — early-warning for LiteLLM upstream drift.

Failures are reported per (model, prompt) and the script keeps going so
you see the full picture rather than bailing on the first error.
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

from evalshift.evaluators.tool_models import ToolSpec  # noqa: E402
from evalshift.evaluators.tool_parser import detect_provider  # noqa: E402
from evalshift.models.client import ModelClient  # noqa: E402

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
    return 0 if failures == 0 else 1


def _provider_or_skip(model: str) -> str | None:
    """Return the provider name iff the matching env var is set."""
    provider = detect_provider(model)
    env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }[provider]
    return provider if os.environ.get(env) else None


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
