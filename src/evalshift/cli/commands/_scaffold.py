"""Scaffolding content and helpers for ``evalshift init``.

``init`` writes a single, minimal ``evalshift.yaml``. The migration-profile
policy blocks, the optional CI workflow template, and the file-writing
helper live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal

import typer
from rich.console import Console

CI_WORKFLOW_PATH: Final = ".github/workflows/evalshift.yml"

MigrationProfile = Literal[
    "model-upgrade",
    "cost-reduction",
    "local-model",
    "quantization",
    "provider-switch",
]

# Shared typer option for the migration profile (used by ``init``).
ProfileOption = Annotated[
    MigrationProfile,
    typer.Option(
        "--profile",
        help="Migration profile used to scaffold default regression budgets.",
    ),
]

PROFILE_POLICIES: Final[dict[str, str]] = {
    "model-upgrade": """\
migration_policy:
  max_overall_regression_rate: 0.03
  max_critical_regressions: 0
  min_equivalence_rate: 0.95
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.03
  max_cost_increase: 0.20
  max_latency_increase: 0.30
""",
    "cost-reduction": """\
migration_policy:
  max_overall_regression_rate: 0.02
  max_critical_regressions: 0
  min_equivalence_rate: 0.97
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.02
  max_cost_increase: 0.05
  max_latency_increase: 0.30
""",
    "local-model": """\
migration_policy:
  max_overall_regression_rate: 0.05
  max_critical_regressions: 0
  min_equivalence_rate: 0.90
  max_tool_argument_drift: 0.02
  max_tool_divergence: 0.05
  max_cost_increase: 0.00
  max_latency_increase: 0.50
""",
    "quantization": """\
migration_policy:
  max_overall_regression_rate: 0.02
  max_critical_regressions: 0
  min_equivalence_rate: 0.97
  max_tool_argument_drift: 0.005
  max_tool_divergence: 0.02
  max_cost_increase: 0.00
  max_latency_increase: 0.20
""",
    "provider-switch": """\
migration_policy:
  max_overall_regression_rate: 0.03
  max_critical_regressions: 0
  min_equivalence_rate: 0.95
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.03
  max_cost_increase: 0.20
  max_latency_increase: 0.40
""",
}

# ``init`` scaffolds the starting point of a real migration, so its default
# model-upgrade budgets are looser than the other profiles: a first run on a
# fresh suite should report the regressions it found, not fail the gate on a
# couple of reworded tool arguments. These mirror the ``MigrationPolicy``
# field defaults, so a config that omits the block behaves the same as one
# that keeps it.
INIT_PROFILE_POLICIES: Final[dict[str, str]] = {
    **PROFILE_POLICIES,
    "model-upgrade": """\
migration_policy:
  max_overall_regression_rate: 0.30
  max_critical_regressions: 1
  min_equivalence_rate: 0.75
  max_tool_argument_drift: 0.20
  max_tool_divergence: 0.20
  max_cost_increase: 0.30
  max_latency_increase: 0.30
""",
}


CI_WORKFLOW_TEMPLATE: Final = """\
# Run EvalShift on every pull request, push runs to hosted EvalShift, and
# update the PR with a hosted regression summary.
#
# Prerequisites:
#   - A provider API key stored as a repo secret (the example below uses
#     GEMINI_API_KEY; swap or add ANTHROPIC_API_KEY / OPENAI_API_KEY to
#     match the models in your evalshift.yaml).
#   - EVALSHIFT_TOKEN stored as a repo secret. Create it from the hosted
#     app's org settings with project or org scope.
name: evalshift

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read
  pull-requests: write
  issues: write
  statuses: write

jobs:
  evalshift:
    runs-on: ubuntu-latest
    env:
      EVALSHIFT_NONINTERACTIVE: "1"
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
    steps:
      - uses: actions/checkout@v4

      - name: EvalShift hosted regression check
        uses: babaliauskas/evalshift-action@v0
        with:
          token: ${{ secrets.EVALSHIFT_TOKEN }}
          fail-on: regression
"""


def write_scaffold_files(
    *,
    target: Path,
    files: dict[str, str],
    force: bool,
    console: Console,
) -> None:
    """Write ``files`` under ``target``; refuse to clobber unless ``force``.

    Raises ``typer.Exit(1)`` (after printing guidance) if any target file
    already exists and ``force`` is not set.
    """
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
        out_path = target / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {out_path}")


__all__ = [
    "CI_WORKFLOW_PATH",
    "CI_WORKFLOW_TEMPLATE",
    "PROFILE_POLICIES",
    "MigrationProfile",
    "ProfileOption",
    "write_scaffold_files",
]
