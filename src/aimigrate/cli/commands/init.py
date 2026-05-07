"""Implementation of ``aimigrate init``.

``init`` scaffolds a working starter project so a new user can go from
a blank directory to a runnable ``aimigrate run`` in three commands:

.. code-block:: shell

    aimigrate init
    aimigrate doctor
    aimigrate run --yes

Files written:

* ``aimigrate.yaml`` — heavily commented config showing every common option.
* ``prompts.py`` — example agent prompt referenced from the yaml.
* ``tools.yaml`` — six tool specs (Anthropic-shape; LiteLLM accepts this
  format on every provider).
* ``golden.jsonl`` — 40 example rows, big enough that the analysis layer
  produces meaningful severity classifications instead of bailing on
  small-sample inputs.

Files are never overwritten unless ``--force`` is passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console

from aimigrate.cli.commands.doctor import CONFIG_FILENAME

PROMPTS_FILENAME: Final = "prompts.py"
SUITE_FILENAME: Final = "golden.jsonl"
TOOLS_FILENAME: Final = "tools.yaml"


_AIMIGRATE_YAML_TEMPLATE: Final = """\
# AIMigrate configuration. See https://github.com/babaliauskas/AIMigrate for docs.
#
# This starter config defaults to a Gemini-only setup (one provider, one
# API key) so it works out of the box for users who only have a Google
# AI Studio key. To use a different provider, set the model ids and
# matching API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY) in your env.
version: 1

prompts:
  - id: customer_routing
    # detection: how AIMigrate finds the prompt body.
    #   manual         - the body is inline below as `content`.
    #   python_string  - AIMigrate AST-walks `path` for a module-level
    #                    string assigned to `variable`.
    detection: python_string
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]
    # tools_path turns this into an agent prompt. The orchestrator
    # routes through ModelClient.complete_with_tools and tool
    # evaluators apply per (source, target) pair.
    tools_path: tools.yaml

# Defaults applied to every run; CLI flags override these.
defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-3.1-flash-lite-preview
  judge_model: gemini-2.5-pro
  concurrency: 4
  cache: true
  max_cost_usd: 50.0

evaluators:
  # tool_selection is the most useful evaluator for agent migrations:
  # it tells you whether the new model picks the same tools as ground
  # truth. severity_floor: high means a regression here can never be
  # downgraded below 'high' regardless of effect size.
  tool_selection:
    - name: routing
      mode: expected
      severity_floor: high

  # Optional: enable these once you've decided you want to score
  # argument drift / call-count changes / refusal regressions.
  # tool_arguments:
  #   - name: routing_args
  # tool_trace_structure:
  #   - name: routing_structure

# Slices break the suite into subsets analysed separately. The `filter`
# is a tag string - examples whose `tags` list contains the value land
# in this slice.
slices:
  - name: security
    filter: security
  - name: routine
    filter: routine
  - name: refund
    filter: refund
  - name: customer_lookup
    filter: customer_lookup
  - name: text_only
    filter: text_only
"""

_PROMPTS_PY_TEMPLATE: Final = '''\
"""Agent system prompt for the customer-support routing example."""

AGENT_SYSTEM_PROMPT = """\\
You are a customer-support routing agent. Read the inbound message and
decide which tools to call, in order, before answering the user.

You have these tools:

  - search_orders(customer_id): look up a customer's recent orders.
  - lookup_customer(query): find a customer by id or email address.
  - issue_refund(order_id, amount_usd, reason): refund an order.
  - update_order_status(order_id, status): change order status
    (pending, shipped, delivered, cancelled, returned).
  - send_email(to, subject, body): send a confirmation/follow-up email.
  - notify_security_team(severity, summary): page the security team.

Routing rules:

  - For routine requests (order lookups, refunds, status updates), call
    only the tools you actually need.
  - For refunds, always issue_refund first, then send_email when the
    customer expects confirmation.
  - For security-sensitive messages (failed logins, account takeover
    attempts, unauthorized data access, phishing), ALWAYS call
    notify_security_team before any other tool.
  - For pure information questions (\"what is your refund policy?\"),
    answer directly with no tool calls.

Inbound message:
{query}
"""
'''

_TOOLS_YAML_TEMPLATE: Final = """\
# Tool specifications for the customer-support routing agent.
# Anthropic-shape (LiteLLM accepts this on every provider).
- name: search_orders
  description: |
    Look up a customer's recent orders by customer id. Use this for any
    inbound request mentioning order status, refunds, or shipping.
  input_schema:
    type: object
    properties:
      customer_id: {type: string, description: "Customer ID to search."}
    required: [customer_id]

- name: lookup_customer
  description: |
    Find a customer record by id or email address. Use when the message
    mentions an email but no customer id, or when you need profile
    info before acting (e.g. before issuing a refund).
  input_schema:
    type: object
    properties:
      query:
        type: string
        description: "Customer id or email address."
    required: [query]

- name: issue_refund
  description: |
    Issue a refund on an existing order. Use only when the customer
    has clearly asked for a refund or the situation warrants one
    (damaged item, late delivery, duplicate charge).
  input_schema:
    type: object
    properties:
      order_id: {type: string, description: "Order id to refund."}
      amount_usd:
        type: number
        description: "Refund amount in US dollars."
      reason: {type: string, description: "Short reason for the refund."}
    required: [order_id, amount_usd, reason]

- name: update_order_status
  description: |
    Change the status of an existing order. Use after a refund to
    mark the order cancelled, or when a customer reports delivery.
  input_schema:
    type: object
    properties:
      order_id: {type: string, description: "Order id to update."}
      status:
        type: string
        enum: [pending, shipped, delivered, cancelled, returned]
    required: [order_id, status]

- name: send_email
  description: Send a confirmation or follow-up email to a customer.
  input_schema:
    type: object
    properties:
      to: {type: string, description: "Recipient email address."}
      subject: {type: string}
      body: {type: string}
    required: [to, subject, body]

- name: notify_security_team
  description: |
    Page the security team about suspicious or sensitive activity.
    Always use this for messages mentioning failed logins, account
    takeover attempts, unauthorized data access, or phishing.
  input_schema:
    type: object
    properties:
      severity:
        type: string
        enum: [low, medium, high, critical]
      summary:
        type: string
        description: "One-sentence description of the incident."
    required: [severity, summary]
"""

# 40 rows: 12 security + 12 routine + 6 text_only + 5 refund + 5 customer_lookup.
# Mix of single-tool, multi-tool, and expected_no_tools cases so every tool
# evaluator has something to score, and each slice clears MIN_N_FOR_TEST=5.
_GOLDEN_JSONL_TEMPLATE: Final = "\n".join(
    [
        '{"id": "ex_security_01", "inputs": {"query": "User account_42 had 5 failed login attempts in the last hour"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_02", "inputs": {"query": "Someone is trying to access account_77 from a new IP in another country"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_03", "inputs": {"query": "Customer reports they cannot log in and admin panel was probed 8 times"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_security_04", "inputs": {"query": "Our SOC alerted on user_99 - possible account takeover"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_05", "inputs": {"query": "alice@example.com tried to access the admin dashboard 8 times in 5 minutes"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_06", "inputs": {"query": "Suspicious activity on customer_55 - multiple sessions from different countries"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_07", "inputs": {"query": "Alert: customer_18 reported their account was compromised; lock the session"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_security_08", "inputs": {"query": "Possible API key leak detected for tenant_3 - multiple unusual calls"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_09", "inputs": {"query": "Customer support flagged that user_404 wallet was drained in 1 minute"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_security_10", "inputs": {"query": "Mass-credential-stuffing attempt against /login from 200 IPs"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}]}',
        '{"id": "ex_security_11", "inputs": {"query": "Internal scan found customer_7 PII exposed in logs"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_security_12", "inputs": {"query": "Confirmed phishing email targeting customer_92 - they clicked a link"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_01", "inputs": {"query": "Where is order #12345 for customer_42?"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_02", "inputs": {"query": "Customer customer_77 is asking about their refund - please look up their orders"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_03", "inputs": {"query": "Send a follow-up email to billing@acme.com about ticket 203"}, "tags": ["routine"], "expected_tools": [{"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_04", "inputs": {"query": "Please look up customer_88 last 3 orders"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_05", "inputs": {"query": "Confirmation email needed for order #98765 to ops@acme.com"}, "tags": ["routine"], "expected_tools": [{"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_06", "inputs": {"query": "Customer customer_31 wants their order history emailed to billing@acme.com"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_07", "inputs": {"query": "What was the most recent purchase from customer_12?"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_08", "inputs": {"query": "Send a welcome email to new customer onboarding@example.com"}, "tags": ["routine"], "expected_tools": [{"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_09", "inputs": {"query": "Look up orders for customer_50 then email a summary to support@acme.com"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_10", "inputs": {"query": "Customer customer_5 asked about the status of order #66677"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_11", "inputs": {"query": "Pull up the order list for customer_63 - they want a refund estimate"}, "tags": ["routine"], "expected_tools": [{"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_routine_12", "inputs": {"query": "Send a return-shipping label to customer_19 at returns@acme.com for order #54321"}, "tags": ["routine"], "expected_tools": [{"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_refund_01", "inputs": {"query": "Issue a $49.99 refund on order #ord_12345 for damaged item"}, "tags": ["refund"], "expected_tools": [{"tool_name": "issue_refund", "match_strategy": "subset"}]}',
        '{"id": "ex_refund_02", "inputs": {"query": "Customer customer_42 wants $120 back on order #ord_99887, says product never arrived. Email confirmation to customer_42@example.com."}, "tags": ["refund"], "expected_tools": [{"tool_name": "issue_refund", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_refund_03", "inputs": {"query": "Look up customer_77 by email and refund their last order $200 for late delivery"}, "tags": ["refund"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}, {"tool_name": "issue_refund", "match_strategy": "subset"}]}',
        '{"id": "ex_refund_04", "inputs": {"query": "Refund order #ord_55543 ($30) for customer_8 and mark it cancelled"}, "tags": ["refund"], "expected_tools": [{"tool_name": "issue_refund", "match_strategy": "subset"}, {"tool_name": "update_order_status", "match_strategy": "subset"}]}',
        '{"id": "ex_refund_05", "inputs": {"query": "Process a $75 refund on order #ord_44231 - reason: duplicate charge"}, "tags": ["refund"], "expected_tools": [{"tool_name": "issue_refund", "match_strategy": "subset"}]}',
        '{"id": "ex_customer_lookup_01", "inputs": {"query": "Look up the customer with email alice@example.com"}, "tags": ["customer_lookup"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}]}',
        '{"id": "ex_customer_lookup_02", "inputs": {"query": "Find the customer record for customer_99"}, "tags": ["customer_lookup"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}]}',
        '{"id": "ex_customer_lookup_03", "inputs": {"query": "Pull up bob@example.com profile and email them a security tip at the same address"}, "tags": ["customer_lookup"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}, {"tool_name": "send_email", "match_strategy": "subset"}]}',
        '{"id": "ex_customer_lookup_04", "inputs": {"query": "Who is customer_31? Show me their orders too."}, "tags": ["customer_lookup"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}, {"tool_name": "search_orders", "match_strategy": "subset"}]}',
        '{"id": "ex_customer_lookup_05", "inputs": {"query": "Find customer info for support@example.com"}, "tags": ["customer_lookup"], "expected_tools": [{"tool_name": "lookup_customer", "match_strategy": "subset"}]}',
        '{"id": "ex_text_01", "inputs": {"query": "What is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_02", "inputs": {"query": "How do I reach billing?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_03", "inputs": {"query": "What hours is your support team available?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_04", "inputs": {"query": "Do you ship internationally?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_05", "inputs": {"query": "What payment methods are accepted?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_06", "inputs": {"query": "Where can I find your terms of service?"}, "tags": ["text_only"], "expected_no_tools": true}',
        "",
    ],
)


def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing files. Off by default to protect work.",
        ),
    ] = False,
    directory: Annotated[
        Path,
        typer.Option(
            "--directory",
            "-d",
            help="Target directory (default: current working directory).",
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("."),
) -> None:
    """Scaffold ``aimigrate.yaml`` + agent prompt + tools + golden suite."""
    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        CONFIG_FILENAME: _AIMIGRATE_YAML_TEMPLATE,
        PROMPTS_FILENAME: _PROMPTS_PY_TEMPLATE,
        TOOLS_FILENAME: _TOOLS_YAML_TEMPLATE,
        SUITE_FILENAME: _GOLDEN_JSONL_TEMPLATE,
    }

    console = Console()
    if not force:
        existing = [name for name in files if (target / name).exists()]
        if existing:
            console.print(
                "[red]Refusing to overwrite existing files:[/red] " + ", ".join(existing),
            )
            console.print(
                "Re-run with [bold]--force[/bold] to overwrite, or move/delete them first.",
            )
            raise typer.Exit(code=1)

    for name, body in files.items():
        (target / name).write_text(body, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {target / name}")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. [cyan]aimigrate doctor[/cyan]     - verify your environment")
    console.print(
        "  2. [cyan]aimigrate run --yes[/cyan] - execute the run "
        "(uses the configured Gemini defaults)",
    )
    console.print("  3. [cyan]aimigrate evaluate <run-id>[/cyan]")
    console.print("  4. [cyan]aimigrate analyze <run-id>[/cyan]")
    console.print("  5. [cyan]aimigrate report <run-id> --open[/cyan]")


__all__ = [
    "PROMPTS_FILENAME",
    "SUITE_FILENAME",
    "TOOLS_FILENAME",
    "init",
]
