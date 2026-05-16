"""Implementation of ``evalshift login`` for hosted token auth."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from evalshift.hosted.client import HostedClient, HostedError
from evalshift.hosted.credentials import (
    DEFAULT_HOST,
    CredentialsError,
    is_insecure_host,
    save_credentials,
)


def login(
    token: Annotated[
        str,
        typer.Option("--token", help="Hosted API token beginning with es_."),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Hosted API base URL."),
    ] = DEFAULT_HOST,
) -> None:
    """Store a hosted API token for future CLI commands.

    Creates an API token via the web UI and pastes it here with --token. The
    browser-based login flow (open browser, sign in, auto-receive token) is
    not yet available; it will ship together with the Phase 6 hosted web app.
    """
    console = Console()
    if is_insecure_host(host):
        console.print(
            "[yellow]![/yellow] host uses plain http against a non-local destination; "
            "your bearer token will transit in cleartext. Prefer https://."
        )
    try:
        me = HostedClient(host=host, token=token).me()
        save_credentials(host, token)
    except (CredentialsError, HostedError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    email = me.get("email", "(unknown user)")
    console.print(f"[green]✓[/green] logged in as {email} at {host.rstrip('/')}")


__all__ = ["login"]
