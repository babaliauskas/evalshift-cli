"""Implementation of ``aimigrate test-call``.

A one-shot smoke test for end-to-end LLM connectivity. Useful when:

* You're setting up a new machine and want to confirm your API keys
  work without writing a full ``aimigrate.yaml``.
* You're debugging a model alias and want to know which canonical id
  it resolves to.
* You're verifying LiteLLM hasn't broken between releases (the manual
  smoke step we'll automate as ``scripts/smoke_live.py``).

Hidden from the top-level ``--help`` because it's a debugging aid, not
part of the user-facing happy path. Will be removed (or relocated under
a hidden ``--debug`` group) at the v0.1.0 cut.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from aimigrate.evaluators.tool_loader import ToolLoaderError, load_tools
from aimigrate.evaluators.tool_models import ToolSpec
from aimigrate.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    ModelClientError,
    RateLimitError,
    ToolCompletionResult,
)
from aimigrate.models.registry import resolve_model


def test_call(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical id (e.g. gemini/gemini-2.5-flash) or alias.",
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="The prompt body to send.",
        ),
    ] = "Reply with a single short greeting.",
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            help="Sampling temperature (default: model's registered default).",
            min=0.0,
            max=2.0,
        ),
    ] = 0.0,
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            help="Completion length cap.",
            min=1,
            max=8192,
        ),
    ] = 256,
    tools_path: Annotated[
        Path | None,
        typer.Option(
            "--tools",
            help="Path to a yaml/json file with tool specs. "
            "When set, the call uses the tool-aware path and prints a ToolTrace.",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Send a single prompt to one model. Prints the response or a clear error."""
    console = Console()

    # Permissive resolution: registry first, then prefix-inferred fallback.
    # LiteLLM is the source of truth at call time and will reject genuinely
    # unknown ids with a clear error.
    meta = resolve_model(model)
    if meta.display_name.endswith("(passthrough)"):
        console.print(
            f"[dim]→ {meta.id}[/dim]  "
            "[yellow](not in AIMigrate registry; passing through to LiteLLM)[/yellow]",
        )
    else:
        console.print(
            f"[dim]→ {meta.id}[/dim]"
            + (f"  [dim](alias: {model})[/dim]" if model != meta.id else ""),
        )

    tools: list[ToolSpec] | None = None
    if tools_path is not None:
        try:
            tools = load_tools(tools_path)
        except ToolLoaderError as exc:
            console.print(exc.format_rich())
            raise typer.Exit(code=1) from exc

    try:
        if tools is not None:
            tool_result = asyncio.run(
                _do_tool_call(
                    model=model,
                    prompt=prompt,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            _print_tool_result(console, tool_result)
            return
        result = asyncio.run(
            _do_call(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
    except AuthError as exc:
        console.print(
            Panel(
                f"[bold red]Authentication failed[/bold red]\n\n{exc}\n\n"
                "Set the appropriate API key environment variable and try again.",
                border_style="red",
                title="aimigrate test-call",
            ),
        )
        raise typer.Exit(code=1) from exc
    except RateLimitError as exc:
        console.print(f"[yellow]⚠[/yellow] rate-limited after retries: {exc}")
        raise typer.Exit(code=1) from exc
    except ModelClientError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_result(console, result)


async def _do_call(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> CompletionResult:
    return await ModelClient().complete(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def _do_tool_call(
    *,
    model: str,
    prompt: str,
    tools: list[ToolSpec],
    temperature: float,
    max_tokens: int,
) -> ToolCompletionResult:
    return await ModelClient().complete_with_tools(
        model=model,
        prompt=prompt,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _print_result(console: Console, result: CompletionResult) -> None:
    body = (
        f"{result.text}\n\n"
        f"[dim]model:[/dim]  {result.model_id}\n"
        f"[dim]tokens:[/dim] in={result.input_tokens}  out={result.output_tokens}\n"
        f"[dim]cost:[/dim]   ${result.cost_usd:.6f}\n"
        f"[dim]time:[/dim]   {result.latency_ms} ms"
    )
    console.print(
        Panel(body, border_style="green", title="aimigrate test-call", title_align="left"),
    )


def _print_tool_result(console: Console, result: ToolCompletionResult) -> None:
    """Render a tool-aware test-call result.

    Shows the ordered tool name list, parallelism, refusal flag, and
    final text snippet (truncated). Designed for fast eyeballing during
    smoke tests, not for a full report.
    """
    trace = result.trace
    parallel = "parallel" if trace.has_parallel_calls() else "sequential"
    tool_lines = [
        f"  {i}. [bold]{c.tool_name}[/bold]({_short_args(c.arguments)})"
        for i, c in enumerate(trace.calls)
    ]
    tool_block = "\n".join(tool_lines) if tool_lines else "  (no tool calls)"
    refusal_block = (
        f"[yellow]refusal:[/yellow] {trace.refusal_text or '(no message)'}\n"
        if trace.raised_refusal
        else ""
    )
    final_text_block = (
        f"[dim]final text:[/dim]\n  {_truncate(trace.final_text, 280)}\n"
        if trace.final_text
        else ""
    )
    body = (
        f"[bold]{trace.call_count}[/bold] tool call(s) "
        f"([dim]{parallel}[/dim])\n"
        f"{tool_block}\n\n"
        f"{refusal_block}{final_text_block}"
        f"[dim]model:[/dim]  {result.model_id}\n"
        f"[dim]tokens:[/dim] in={result.input_tokens}  out={result.output_tokens}\n"
        f"[dim]cost:[/dim]   ${result.cost_usd:.6f}\n"
        f"[dim]time:[/dim]   {result.latency_ms} ms"
    )
    console.print(
        Panel(body, border_style="green", title="aimigrate test-call --tools", title_align="left"),
    )


def _short_args(args: dict[str, object]) -> str:
    """Render args as a one-liner, truncated for readability."""
    if not args:
        return ""
    parts = [f"{k}={_truncate(repr(v), 40)}" for k, v in args.items()]
    return ", ".join(parts)


def _truncate(text: str | None, n: int) -> str:
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


__all__ = ["test_call"]
