"""Implementation of ``aimigrate init``.

``init`` scaffolds a working starter project so a new user can go from a
blank directory to a runnable ``aimigrate run`` in three commands:

.. code-block:: shell

    aimigrate init
    aimigrate doctor
    aimigrate run --from claude-4.5-sonnet --to claude-5-sonnet

The files written are:

* ``aimigrate.yaml`` — heavily commented config showing every common option.
* ``prompts.py`` — example prompt module that ``aimigrate.yaml`` references.
* ``golden.jsonl`` — three example suite rows so ``aimigrate run`` has data
  to chew on immediately.

Files are never overwritten unless ``--force`` is passed; if any target file
already exists, ``init`` exits 1 with a list of the conflicts so users can
inspect them before agreeing to clobber.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console

from aimigrate.cli.commands.doctor import CONFIG_FILENAME

PROMPTS_FILENAME: Final = "prompts.py"
SUITE_FILENAME: Final = "golden.jsonl"

_AIMIGRATE_YAML_TEMPLATE: Final = """\
# AIMigrate configuration. See https://github.com/babaliauskas/AIMigrate for docs.
#
# This starter config defaults to a Gemini-only setup (one provider, one
# API key) so it works out of the box for users who only have a Google
# AI Studio key. To use a different provider, set the model ids and
# matching API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY) in your env.
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
  # Judge defaults to gemini-2.5-pro so a Gemini-only key works
  # end-to-end. Swap to claude-5-sonnet or gpt-4o if you want a
  # different judge (and have the matching key set).
  judge_model: gemini-2.5-pro
  concurrency: 10
  cache: true
  max_cost_usd: 50.0

# Evaluators score the (source, target) outputs. The starter config
# only enables structural checks because they need no extra API keys.
# Uncomment the semantic / llm_judge blocks once you've decided which
# provider to use for them.
evaluators:
  # Structural: deterministic checks on output shape (free, no API calls).
  structural:
    - type: length
      min_chars: 5
      max_chars: 500

  # Semantic: cosine similarity on embeddings. Pick an embedding model
  # that matches a key you have set:
  #   - gemini/text-embedding-004      (needs GOOGLE_API_KEY)
  #   - text-embedding-3-small         (needs OPENAI_API_KEY)
  #   - voyage/voyage-3                (needs VOYAGE_API_KEY)
  # semantic:
  #   embedding_model: gemini/text-embedding-004

  # LLM-as-judge: pairwise comparisons. Each entry adds one judge call
  # per (prompt, example). Set judge_model to anything LiteLLM supports.
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

_GOLDEN_JSONL_TEMPLATE: Final = (
    '{"id": "ex1", "inputs": {"name": "Alex", "tone": "formal"}, '
    '"tags": ["formal"]}\n'
    '{"id": "ex2", "inputs": {"name": "Sam",  "tone": "casual"}, '
    '"tags": ["casual"]}\n'
    '{"id": "ex3", "inputs": {"name": "Jamie","tone": "formal"}, '
    '"tags": ["formal"]}\n'
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
    """Scaffold ``aimigrate.yaml`` + example prompts + golden suite."""
    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
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
    console.print("  1. [cyan]aimigrate doctor[/cyan]  — verify your environment")
    console.print(
        "  2. [cyan]aimigrate run --from <source> --to <target>[/cyan]  — execute the run",
    )


__all__ = [
    "PROMPTS_FILENAME",
    "SUITE_FILENAME",
    "init",
]
