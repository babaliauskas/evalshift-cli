"""Implementation of ``aimigrate init``.

``init`` scaffolds a working starter project so a new user can go from
a blank directory to a runnable ``aimigrate run`` in three commands:

.. code-block:: shell

    aimigrate init                    # plain text-prompt project (default)
    aimigrate init --agent            # v0.2 agent project with tools

Files written by the default flow:

* ``aimigrate.yaml`` — heavily commented config showing every common option.
* ``prompts.py`` — example prompt module referenced from the yaml.
* ``golden.jsonl`` — 24 example rows so the statistics layer can produce
  meaningful severity classifications instead of bailing on small-sample
  inputs.

Agent flow (``--agent``) adds:

* ``tools.yaml`` — three tool specs (Anthropic-shape; LiteLLM accepts
  this format on every provider).

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


# ---------------------------------------------------------------------------
# Default (text-prompt) templates
# ---------------------------------------------------------------------------


_AIMIGRATE_YAML_TEMPLATE: Final = """\
# AIMigrate configuration. See https://github.com/babaliauskas/AIMigrate for docs.
#
# This starter config defaults to a Gemini-only setup (one provider, one
# API key) so it works out of the box for users who only have a Google
# AI Studio key. To use a different provider, set the model ids and
# matching API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY) in your env.
#
# For an agent-style project (with tools), run `aimigrate init --agent`.
version: 1

# Prompts to evaluate. Each entry produces one row per suite example per model.
prompts:
  - id: greet
    # detection: how AIMigrate finds the prompt body.
    #   manual         — the body is inline below as `content`.
    #   python_string  — AIMigrate AST-walks `path` for a module-level
    #                    string assigned to `variable`.
    detection: python_string
    path: prompts.py
    variable: GREET_PROMPT
    # variables: template placeholders the prompt expects in suite inputs.
    variables: [name, tone]

# Defaults applied to every run; CLI flags override these.
defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  judge_model: gemini-2.5-pro
  concurrency: 10
  cache: true
  max_cost_usd: 50.0

# Evaluators score the (source, target) outputs. The starter config
# only enables structural checks because they need no extra API keys.
# Uncomment the semantic / llm_judge blocks once you've decided which
# provider to use for them.
evaluators:
  structural:
    - type: length
      min_chars: 5
      max_chars: 500

  # semantic:
  #   embedding_model: gemini/text-embedding-004    # needs GOOGLE_API_KEY

  # llm_judge:
  #   - criterion_name: tone_match
  #     criterion_prompt: |
  #       Which output more clearly matches the requested tone?
  #       Reply with strict JSON: {"winner": "A" | "B" | "tie", "reason": "..."}.
  #     judge_model: gemini-2.5-pro

# Slices break the suite into subsets analysed separately. The `filter`
# is a tag string — examples whose `tags` list contains the value land
# in this slice.
slices:
  - name: formal
    filter: formal
  - name: casual
    filter: casual
"""

_PROMPTS_PY_TEMPLATE: Final = '''\
"""Example prompt module referenced by aimigrate.yaml.

AIMigrate finds prompts here by AST-walking for module-level string
assignments. Keep prompts as plain string literals (no f-strings or
concatenations at the top level) so the parser can extract them safely.
"""

GREET_PROMPT = """\\
You are a friendly assistant. Greet {name} in a {tone} tone.
Reply with one short sentence."""
'''

# 24 rows — 12 formal + 12 casual. Big enough for every slice (formal /
# casual) to clear the n>=5 threshold (so analysis runs at all) and for
# the implicit "all" slice to clear n>=20 (so the row doesn't carry the
# "uncertain — small sample" note).
_GOLDEN_JSONL_TEMPLATE: Final = "\n".join(
    [
        '{"id": "ex_formal_01", "inputs": {"name": "Alex",     "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_02", "inputs": {"name": "Jamie",    "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_03", "inputs": {"name": "Casey",    "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_04", "inputs": {"name": "Morgan",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_05", "inputs": {"name": "Quinn",    "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_06", "inputs": {"name": "Taylor",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_07", "inputs": {"name": "Robin",    "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_08", "inputs": {"name": "Skyler",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_09", "inputs": {"name": "Hayden",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_10", "inputs": {"name": "Parker",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_11", "inputs": {"name": "Dakota",   "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_formal_12", "inputs": {"name": "Phoenix",  "tone": "formal"}, "tags": ["formal"]}',
        '{"id": "ex_casual_01", "inputs": {"name": "Sam",      "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_02", "inputs": {"name": "Riley",    "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_03", "inputs": {"name": "Drew",     "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_04", "inputs": {"name": "Avery",    "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_05", "inputs": {"name": "Reese",    "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_06", "inputs": {"name": "Charlie",  "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_07", "inputs": {"name": "Sasha",    "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_08", "inputs": {"name": "Indigo",   "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_09", "inputs": {"name": "Kai",      "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_10", "inputs": {"name": "River",    "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_11", "inputs": {"name": "Sage",     "tone": "casual"}, "tags": ["casual"]}',
        '{"id": "ex_casual_12", "inputs": {"name": "Rowan",    "tone": "casual"}, "tags": ["casual"]}',
        "",
    ],
)


# ---------------------------------------------------------------------------
# Agent (--agent) templates
# ---------------------------------------------------------------------------


_AGENT_AIMIGRATE_YAML_TEMPLATE: Final = """\
# AIMigrate v0.2 agent project: a customer-routing agent with 3 tools.
# See https://github.com/babaliauskas/AIMigrate/blob/main/docs/agents.md
version: 1

prompts:
  - id: customer_routing
    detection: python_string
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]
    # tools_path turns this into an agent prompt. The orchestrator
    # routes through ModelClient.complete_with_tools and tool
    # evaluators apply per (source, target) pair.
    tools_path: tools.yaml

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  judge_model: gemini-2.5-pro
  concurrency: 4
  cache: true
  max_cost_usd: 50.0

evaluators:
  # Cheap structural baseline on the model's text response.
  structural:
    - type: length
      min_chars: 1
      max_chars: 2000

  # tool_selection is the most useful evaluator for migrations: it
  # tells you whether the new model picks the same tools as ground
  # truth.
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

slices:
  - name: security
    filter: security
  - name: routine
    filter: routine
  - name: text_only
    filter: text_only
"""

_AGENT_PROMPTS_PY_TEMPLATE: Final = '''\
"""Agent system prompt for the customer-routing example."""

AGENT_SYSTEM_PROMPT = """\\
You are a customer-routing agent. Read the inbound message and decide which
tools to call, in order, before answering the user. You have:

  - search_orders(customer_id): look up a customer's recent orders.
  - notify_security_team(severity, summary): page the security team.
  - send_email(to, subject, body): send a confirmation email.

For routine requests (refund questions, order lookups), call only the tools
you actually need.

For security-sensitive requests (suspicious activity, multiple failed
logins, attempted account access), ALWAYS call notify_security_team
before any other tool.

Inbound message:
{query}
"""
'''

_AGENT_TOOLS_YAML_TEMPLATE: Final = """\
# Tool specifications for the customer-routing agent.
# Anthropic-shape (LiteLLM accepts this on every provider).
- name: search_orders
  description: |
    Look up a customer's recent orders. Use this for any inbound request
    that mentions order status, refunds, or shipping questions.
  input_schema:
    type: object
    properties:
      customer_id: {type: string, description: "Customer ID to search."}
    required: [customer_id]

- name: notify_security_team
  description: |
    Page the security team about suspicious or sensitive activity.
    Always use this for messages mentioning failed logins, account
    takeover attempts, or unauthorized data access.
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

- name: send_email
  description: Send a confirmation or follow-up email to a customer.
  input_schema:
    type: object
    properties:
      to: {type: string, description: "Recipient email address."}
      subject: {type: string}
      body: {type: string}
    required: [to, subject, body]
"""

# 30 rows — 12 security + 12 routine + 6 text_only. Mirrors the suite
# in examples/agent/golden.jsonl. Mixes single-tool, multi-tool, and
# expected_no_tools cases so every tool evaluator has something to score.
_AGENT_GOLDEN_JSONL_TEMPLATE: Final = "\n".join(
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
        '{"id": "ex_text_01", "inputs": {"query": "What is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_02", "inputs": {"query": "How do I reach billing?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_03", "inputs": {"query": "What hours is your support team available?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_04", "inputs": {"query": "Do you ship internationally?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_05", "inputs": {"query": "What payment methods are accepted?"}, "tags": ["text_only"], "expected_no_tools": true}',
        '{"id": "ex_text_06", "inputs": {"query": "Where can I find your terms of service?"}, "tags": ["text_only"], "expected_no_tools": true}',
        "",
    ],
)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


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
    agent: Annotated[
        bool,
        typer.Option(
            "--agent",
            help="Scaffold a v0.2 agent project (with tools.yaml) instead of "
            "the plain text-prompt template.",
        ),
    ] = False,
) -> None:
    """Scaffold ``aimigrate.yaml`` + example prompts + golden suite."""
    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)

    if agent:
        files: dict[str, str] = {
            CONFIG_FILENAME: _AGENT_AIMIGRATE_YAML_TEMPLATE,
            PROMPTS_FILENAME: _AGENT_PROMPTS_PY_TEMPLATE,
            TOOLS_FILENAME: _AGENT_TOOLS_YAML_TEMPLATE,
            SUITE_FILENAME: _AGENT_GOLDEN_JSONL_TEMPLATE,
        }
    else:
        files = {
            CONFIG_FILENAME: _AIMIGRATE_YAML_TEMPLATE,
            PROMPTS_FILENAME: _PROMPTS_PY_TEMPLATE,
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
    if agent:
        console.print("  1. [cyan]aimigrate doctor[/cyan]     — verify your environment")
        console.print(
            "  2. [cyan]aimigrate run --yes[/cyan] — execute the agent run "
            "(uses the configured Gemini defaults)",
        )
        console.print("  3. [cyan]aimigrate evaluate <run-id>[/cyan]")
        console.print("  4. [cyan]aimigrate analyze <run-id>[/cyan]")
        console.print("  5. [cyan]aimigrate report <run-id> --open[/cyan]")
    else:
        console.print("  1. [cyan]aimigrate doctor[/cyan]     — verify your environment")
        console.print(
            "  2. [cyan]aimigrate run --from <source> --to <target>[/cyan] — execute the run",
        )
        console.print(
            "     (or run [cyan]aimigrate init --agent[/cyan] for a v0.2 "
            "agent-style project with tools)",
        )


__all__ = [
    "PROMPTS_FILENAME",
    "SUITE_FILENAME",
    "TOOLS_FILENAME",
    "init",
]
