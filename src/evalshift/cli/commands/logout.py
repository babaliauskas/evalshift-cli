"""Implementation of ``evalshift logout``."""

from __future__ import annotations

import typer
from rich.console import Console

from evalshift.hosted.credentials import CredentialsError, credentials_path, delete_credentials


def logout() -> None:
    """Remove stored hosted EvalShift credentials from this machine."""
    console = Console()
    path = credentials_path()
    try:
        removed = delete_credentials()
    except CredentialsError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if removed:
        console.print(f"[green]✓[/green] removed hosted credentials at {path}")
    else:
        console.print(f"[yellow]![/yellow] no hosted credentials found at {path}")
    console.print("Server API tokens can be revoked from the EvalShift web app.")


__all__ = ["logout"]
