"""Helpers for debug commands that inspect completed run artifacts."""

from __future__ import annotations

from pathlib import Path

from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.runner.checkpoint import iter_calls, run_dir_for
from evalshift_cli.runner.models import Call, representative_calls
from evalshift_cli.traces.loader import TRACES_FILENAME, load_traces_jsonl
from evalshift_cli.traces.models import AgentTrace


def load_scores(run_dir: Path) -> list[EvalRecord]:
    """Load score records from a run directory."""
    path = run_dir / SCORES_FILENAME
    if not path.exists():
        return []
    return [
        EvalRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def calls_for_example(run_dir: Path, example_id: str) -> dict[str, Call]:
    """Return source/target calls for one example id.

    Sample 0's on a repeated-sampling run (``defaults.samples_per_example``):
    the debug views show one output per side.
    """
    out: dict[str, Call] = {}
    for call in representative_calls(iter_calls(run_dir)):
        if call.example_id == example_id:
            out[call.role] = call
    return out


def traces_for_example(run_dir: Path, example_id: str) -> dict[str, AgentTrace]:
    """Return imported source/target traces for one example id, if present."""
    path = run_dir / TRACES_FILENAME
    if not path.exists():
        return {}
    out: dict[str, AgentTrace] = {}
    for trace in load_traces_jsonl(path):
        if trace.example_id == example_id:
            out[trace.role] = trace
    return out


def run_dir(run_id: str, runs_base: Path) -> Path:
    """Resolve a run id under the CLI's runs base."""
    return run_dir_for(run_id, runs_base)


__all__ = ["calls_for_example", "load_scores", "run_dir", "traces_for_example"]
