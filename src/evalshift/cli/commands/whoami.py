"""Implementation of ``evalshift whoami``."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from evalshift.hosted.client import HostedClient, HostedError
from evalshift.hosted.credentials import CredentialsError, resolve_credentials


def whoami(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Hosted API base URL."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Hosted API token."),
    ] = None,
) -> None:
    """Print the hosted identity and visible org roles."""
    console = Console()
    try:
        credentials = resolve_credentials(host=host, token=token)
        client = HostedClient(host=credentials.host, token=credentials.token)
        me = client.me()
        orgs = client.orgs()
    except (CredentialsError, HostedError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[dim]host:[/dim] {credentials.host}")
    console.print(f"[dim]user:[/dim] {me.get('email', '(unknown user)')}")
    if orgs:
        console.print("[dim]orgs:[/dim]")
        for org in orgs:
            console.print(f"  {org.get('slug')}  {org.get('role')}")
    else:
        console.print("[dim]orgs:[/dim] (none)")


__all__ = ["whoami"]
