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
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from aimigrate.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    ModelClientError,
    RateLimitError,
)
from aimigrate.models.registry import UnknownModelError, get_model


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
) -> None:
    """Send a single prompt to one model. Prints the response or a clear error."""
    console = Console()

    # Resolve early so an unknown alias fails before we do anything noisy.
    try:
        meta = get_model(model)
    except UnknownModelError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[dim]→ {meta.id}[/dim]" + (f"  [dim](alias: {model})[/dim]" if model != meta.id else "")
    )

    try:
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


__all__ = ["test_call"]
