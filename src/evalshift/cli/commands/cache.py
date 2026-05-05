"""Implementation of ``evalshift cache clear``.

Tiny utility that deletes every row from the local SQLite cache. We
group cache subcommands under a Typer sub-app so future operations
(``cache info``, ``cache prune``, …) slot in naturally without
polluting the top-level command list.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from evalshift.cache.store import CacheStore

cache_app = typer.Typer(
    name="cache",
    help="Manage the local LLM-response cache.",
    no_args_is_help=True,
    add_completion=False,
)


@cache_app.command(name="clear")
def cache_clear() -> None:
    """Remove every entry from the local cache."""
    console = Console()
    removed = asyncio.run(_clear())
    console.print(f"[green]✓[/green] cleared {removed} cached call(s)")


async def _clear() -> int:
    store = await CacheStore.open()
    try:
        return await store.clear()
    finally:
        await store.close()


__all__ = ["cache_app"]
