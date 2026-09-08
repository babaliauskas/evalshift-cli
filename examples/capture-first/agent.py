"""The instrumented agent that produced this example's captures.

A deterministic, dependency-free stand-in for a real on-call triage agent: it
takes an alert, "decides" what to do with a hard-coded rule, calls tools, and
writes a short summary. No LLM is called — the point of this file is the
*instrumentation*, not the intelligence.

Three things make a capture promotable into a golden case:

* ``record_model_call(input=<the fully rendered prompt string>, ...)`` — the
  first model call's ``input`` is what ``capture promote``/``capture sync``
  recovers as the case's ``inputs``. A bare string becomes
  ``{"input": "<string>"}``, which is exactly what the passthrough ``replay``
  prompt that ``evalshift init`` scaffolds renders back.
* ``tools=`` on every model call — the toolset the model was offered. It is a
  required argument with no default; ``capture sync`` refuses to promote a
  capture whose model call recorded no toolset.
* ``@capture.tool`` — each recorded call becomes an ``expected_tools`` entry,
  the ground truth the candidate model is scored against.

Regenerate the captures under ``.evalshift/captures/`` with::

    uv venv --python 3.11 /tmp/sdk-venv          # keep the SDK out of the CLI venv:
    uv pip install --python /tmp/sdk-venv/bin/python evalshift-sdk
    cd examples/capture-first
    EVALSHIFT_CAPTURE=1 /tmp/sdk-venv/bin/python agent.py

The SDK and the CLI both import as ``evalshift``, so they must live in
separate virtual environments.
"""

from __future__ import annotations

from typing import Any

from evalshift import capture, record_model_call

# The toolset this agent's router model is offered. Anthropic shape; the SDK
# also accepts OpenAI and Gemini shapes. `capture sync` writes it once to
# `.evalshift/toolsets/<sha256>.json` and every capture points at it by
# content hash, so the golden suite records exactly which tools were on the
# table when the ground truth was recorded.
TRIAGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_service_health",
        "description": (
            "Read the current error rate, p99 latency and deploy marker for one "
            "service. Use this first on any alert that names a service."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service name, e.g. checkout-api."},
            },
            "required": ["service"],
        },
    },
    {
        "name": "search_logs",
        "description": (
            "Search a service's recent logs. Use it to find the failing code path "
            "once health metrics confirm something is wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service name to search."},
                "query": {"type": "string", "description": "Substring or error class to match."},
                "window_minutes": {
                    "type": "integer",
                    "description": "How far back to search, in minutes.",
                },
            },
            "required": ["service", "query"],
        },
    },
    {
        "name": "open_incident",
        "description": (
            "Open an incident record. Use it for anything customer-visible or "
            "anything that will outlive the current shift."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "One-line incident title."},
                "service": {"type": "string", "description": "Primary affected service."},
                "severity": {"type": "string", "enum": ["sev1", "sev2", "sev3"]},
            },
            "required": ["title", "service", "severity"],
        },
    },
    {
        "name": "page_oncall",
        "description": (
            "Page the on-call engineer for a team. Reserve it for sev1 and sev2 — "
            "a page wakes someone up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Owning team, e.g. payments."},
                "severity": {"type": "string", "enum": ["sev1", "sev2"]},
                "summary": {"type": "string", "description": "One sentence for the pager."},
            },
            "required": ["team", "severity", "summary"],
        },
    },
]

SYSTEM_PROMPT = """\
You are the on-call triage assistant for a payments platform.

Given an alert, decide what to do and act:
  - Always check service health before drawing a conclusion.
  - Open an incident for anything customer-visible.
  - Page the owning team only for sev1 and sev2.
  - Answer ownership and policy questions directly, without calling tools.

Be concise. Reply with what you did and why, in at most three sentences."""


@capture.tool(name="get_service_health")
def get_service_health(service: str) -> dict[str, Any]:
    """Return canned health metrics for a service."""
    return {"service": service, "error_rate": 0.18, "p99_ms": 4200, "deployed_minutes_ago": 11}


@capture.tool(name="search_logs")
def search_logs(service: str, query: str, window_minutes: int = 30) -> dict[str, Any]:
    """Return canned log matches for a service."""
    return {"service": service, "matches": 214, "top_message": f"{query} in worker pool"}


@capture.tool(name="open_incident")
def open_incident(title: str, service: str, severity: str) -> dict[str, Any]:
    """Open a canned incident record."""
    return {"incident_id": "INC-4471", "title": title, "service": service, "severity": severity}


@capture.tool(name="page_oncall")
def page_oncall(team: str, severity: str, summary: str) -> dict[str, Any]:
    """Page a canned on-call rotation."""
    return {"paged": team, "severity": severity, "acknowledged": False}


@capture.agent(suite="oncall_triage", redact=True, tools=TRIAGE_TOOLS)
def triage(alert: str) -> str:
    """Triage one alert. One agent invocation == one capture file."""
    prompt = f"{SYSTEM_PROMPT}\n\nAlert: {alert}"

    # Round 1: the router model sees the prompt and the toolset, and picks tools.
    # `input` is the fully rendered prompt because that is what a replay has to
    # send to the candidate model to reproduce this turn.
    record_model_call(
        model_id="demo/triage-router",
        tools=TRIAGE_TOOLS,
        input=prompt,
        output="",
        generation_config={"temperature": 0.0},
    )

    lowered = alert.lower()
    if "error rate" in lowered or "5xx" in lowered:
        get_service_health(service="checkout-api")
        open_incident(
            title="checkout-api 5xx spike after deploy",
            service="checkout-api",
            severity="sev1",
        )
        page_oncall(
            team="payments",
            severity="sev1",
            summary="checkout-api is returning 5xx for roughly a fifth of requests.",
        )
        reply = (
            "Health checks confirm checkout-api is failing about 18 percent of requests "
            "since a deploy 11 minutes ago. I opened INC-4471 at sev1 and paged the "
            "payments on-call."
        )
    elif "slow" in lowered or "latency" in lowered or "job" in lowered:
        get_service_health(service="batch-reporting")
        search_logs(service="batch-reporting", query="timeout", window_minutes=60)
        reply = (
            "batch-reporting is degraded but not customer-visible: the nightly job is "
            "overrunning on worker timeouts. I left it unpaged and noted it for the "
            "morning handover."
        )
    else:
        # No tool call at all. Promotion records this as expected_no_tools, which
        # is its own ground truth: a candidate model that starts paging someone
        # here has regressed just as surely as one that stops paging above.
        reply = (
            "The notifications service is owned by the messaging team. No action was "
            "needed, so I did not open an incident or page anyone."
        )

    # Round 2: the model turns the tool results into the reply the user saw.
    # Promotion reads the last text output as the case's `expected`.
    record_model_call(
        model_id="demo/triage-router",
        tools=TRIAGE_TOOLS,
        input=prompt,
        output=reply,
        generation_config={"temperature": 0.0},
    )
    return reply


ALERTS = [
    "checkout-api 5xx error rate has been above 15 percent for the last 6 minutes.",
    "The nightly batch-reporting job is running slow - 12 minutes past its usual finish.",
    "Which team owns the notifications service, and do I need to open an incident for a "
    "single failed webhook retry?",
]


if __name__ == "__main__":
    for alert in ALERTS:
        print(triage(alert))
    print(
        "\ndone - with EVALSHIFT_CAPTURE=1 set, one capture per alert is now under "
        ".evalshift/captures/oncall_triage/",
    )
