"""Capture replay fixtures from a completed live run.

Reads ``.evalshift/runs/<run-id>/raw.jsonl`` and the project's
``golden.jsonl`` and emits a ``fixtures.jsonl`` next to the config (or
to ``--out``) compatible with ``evalshift run --offline``.

Usage:

    python scripts/capture_fixtures.py <run-id> \\
        [--suite golden.jsonl] [--out fixtures.jsonl] [--match-var query]

The ``--match-var`` flag picks which suite-input variable becomes the
fixture's ``match`` substring (default: ``query``). Use the variable
whose value is unique per example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evalshift.runner.checkpoint import iter_calls, run_dir_for  # noqa: E402
from evalshift.suite.loader import load_jsonl  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", help="Run id under .evalshift/runs/")
    p.add_argument(
        "--runs-base",
        type=Path,
        default=None,
        help="Override the .evalshift/runs/ base (default: project default).",
    )
    p.add_argument(
        "--suite",
        type=Path,
        default=Path("golden.jsonl"),
        help="Path to the suite (default: ./golden.jsonl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures.jsonl"),
        help="Output fixtures path (default: ./fixtures.jsonl).",
    )
    p.add_argument(
        "--match-var",
        default="query",
        help="Suite input variable used as the fixture 'match' substring (default: query).",
    )
    args = p.parse_args()

    run_dir = run_dir_for(args.run_id, args.runs_base)
    if not run_dir.exists():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    suite = load_jsonl(args.suite)
    inputs_by_id = {ex.id: ex.inputs for ex in suite.examples}

    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for call in iter_calls(run_dir):
            inputs = inputs_by_id.get(call.example_id)
            if inputs is None:
                print(
                    f"warn: example {call.example_id!r} not in suite — skipping",
                    file=sys.stderr,
                )
                continue
            match = inputs.get(args.match_var)
            if not match:
                print(
                    f"warn: example {call.example_id!r} has no {args.match_var!r} — skipping",
                    file=sys.stderr,
                )
                continue
            record = _build_record(call, match=str(match))
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            written += 1

    print(f"wrote {written} fixtures → {args.out}")
    return 0


def _build_record(call, *, match: str) -> dict:
    if call.trace is not None:
        result = {
            "calls": [
                {
                    "tool_name": c.tool_name,
                    "arguments": c.arguments,
                    **({"call_id": c.call_id} if c.call_id else {}),
                    **({"parent_call_id": c.parent_call_id} if c.parent_call_id else {}),
                    "sequence_index": c.sequence_index,
                }
                for c in call.trace.calls
            ],
            "final_text": call.trace.final_text,
            "raised_refusal": call.trace.raised_refusal,
            "refusal_text": call.trace.refusal_text,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": call.cost_usd,
            "latency_ms": call.latency_ms,
        }
        kind = "tools"
    else:
        result = {
            "text": call.text,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": call.cost_usd,
            "latency_ms": call.latency_ms,
        }
        kind = "text"
    return {
        "model": call.model_id,
        "match": match,
        "kind": kind,
        "result": result,
    }


if __name__ == "__main__":
    raise SystemExit(main())
