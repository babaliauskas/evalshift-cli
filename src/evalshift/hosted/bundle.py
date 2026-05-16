"""Build hosted run bundles from local EvalShift run artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

from evalshift.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift.cli.commands.evaluate import SCORES_FILENAME
from evalshift.config.loader import ConfigError, load_config
from evalshift.config.models import EvalShiftConfig
from evalshift.evaluators.base import EvalRecord
from evalshift.runner.checkpoint import (
    RAW_FILENAME,
    STATE_FILENAME,
    iter_calls,
    read_state,
    run_dir_for,
)
from evalshift.runner.models import Call
from evalshift.suite.loader import SuiteError, load_jsonl
from evalshift.suite.models import Suite, SuiteExample

BUNDLE_FILENAME = "run_bundle.json.gz"
BUNDLE_VERSION = 1
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BundleError(Exception):
    """Raised when a local run cannot be packaged for hosted upload."""


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    run_id: str
    path: Path
    manifest: dict[str, Any]
    size_bytes: int


def canonical_json(value: Any) -> bytes:
    """Return canonical JSON bytes for hashing and deterministic bundles."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_hash(value: Any) -> str:
    """Return the bundle-spec SHA-256 hash for a JSON value."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def build_bundle(
    run_id: str,
    *,
    config_path: Path,
    suite_path: Path,
    runs_base: Path,
    output: Path | None = None,
    project: str | None = None,
    env: Mapping[str, str] | None = None,
) -> BundleBuildResult:
    """Package a completed local run as ``run_bundle.json.gz``."""
    run_dir = run_dir_for(run_id, runs_base)
    _require_artifacts(run_dir)
    try:
        cfg = load_config(config_path)
        suite = load_jsonl(suite_path)
    except (ConfigError, SuiteError) as exc:
        raise BundleError(str(exc)) from exc

    project_slug = project or cfg.project
    if not project_slug:
        raise BundleError(
            "hosted project is required; pass --project or set `project` in evalshift.yaml"
        )

    state = read_state(run_dir)
    if state.status != "completed":
        raise BundleError(f"run {run_id} is {state.status!r}; only completed runs can be bundled")
    calls = list(iter_calls(run_dir))
    scores = _read_scores(run_dir)
    _read_json(run_dir / ANALYSIS_FILENAME)
    report_html = (run_dir / "report.html").read_text(encoding="utf-8")

    evaluator_config = _evaluator_config_snapshot(cfg)
    dataset_snapshot = _dataset_snapshot(suite_path, suite)
    examples = _build_examples(suite=suite, calls=calls, scores=scores)
    aggregate = _build_aggregate(
        examples, state_started=state.started_at, state_ended=state.last_checkpoint_at
    )
    git = _git_metadata(env or os.environ)

    created_at = _format_z(state.started_at)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "project_slug": project_slug,
        "source_model": state.models.source,
        "target_model": state.models.target,
        "git_sha": git["git_sha"],
        "branch": git["branch"],
        "pr_number": git["pr_number"],
        "eval_config_hash": compute_hash(evaluator_config),
        "dataset_hash": compute_hash(dataset_snapshot),
        "size_bytes": 0,
        "created_at": created_at,
        "bundle_version": BUNDLE_VERSION,
    }
    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "manifest": manifest,
        "aggregate": aggregate,
        "examples": examples,
        "report_html": report_html,
        "evaluator_config": evaluator_config,
        "dataset_snapshot": dataset_snapshot,
    }
    compressed = _compress_with_converged_size(bundle)
    _validate_bundle(bundle)

    target = output or run_dir / BUNDLE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compressed)
    return BundleBuildResult(
        run_id=run_id,
        path=target,
        manifest=dict(bundle["manifest"]),
        size_bytes=len(compressed),
    )


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a gzipped hosted bundle from disk."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"failed to read bundle {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"bundle {path} must contain a JSON object")
    return data


def _require_artifacts(run_dir: Path) -> None:
    required = [
        STATE_FILENAME,
        RAW_FILENAME,
        SCORES_FILENAME,
        ANALYSIS_FILENAME,
        "report.html",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise BundleError(f"missing run artifact(s) in {run_dir}: {', '.join(missing)}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scores(run_dir: Path) -> list[EvalRecord]:
    scores_path = run_dir / SCORES_FILENAME
    scores: list[EvalRecord] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            scores.append(EvalRecord.model_validate_json(line))
    return scores


def _evaluator_config_snapshot(cfg: EvalShiftConfig) -> dict[str, Any]:
    dumped = cfg.model_dump(mode="json")
    return {
        "version": dumped["version"],
        "prompts": dumped["prompts"],
        "defaults": dumped["defaults"],
        "evaluators": dumped["evaluators"],
        "slices": dumped["slices"],
    }


def _dataset_snapshot(path: Path, suite: Suite) -> dict[str, Any]:
    tags = sorted({tag for example in suite.examples for tag in example.tags})
    return {
        "suite_path": str(path),
        "size": len(suite.examples),
        "slices": tags,
        "examples": [example.model_dump(mode="json") for example in suite.examples],
    }


def _build_examples(
    *,
    suite: Suite,
    calls: list[Call],
    scores: list[EvalRecord],
) -> list[dict[str, Any]]:
    calls_by_pair: dict[tuple[str, str], dict[str, Call]] = defaultdict(dict)
    for call in calls:
        calls_by_pair[(call.prompt_id, call.example_id)][call.role] = call
    scores_by_pair: dict[tuple[str, str], list[EvalRecord]] = defaultdict(list)
    for score in scores:
        scores_by_pair[(score.prompt_id, score.example_id)].append(score)
    suite_by_id = {example.id: example for example in suite.examples}

    rows: list[dict[str, Any]] = []
    for prompt_id, example_id in sorted(calls_by_pair):
        pair = calls_by_pair[(prompt_id, example_id)]
        source = pair.get("source")
        target = pair.get("target")
        example = suite_by_id.get(example_id)
        pair_scores = scores_by_pair.get((prompt_id, example_id), [])
        errors = [item.error for item in (source, target) if item is not None and item.error]
        errors.extend(score.error for score in pair_scores if score.error)
        passed = not errors and all(score.delta >= 0 for score in pair_scores)
        score_values = [score.target_score for score in pair_scores if score.error is None]
        row_score = sum(score_values) / len(score_values) if score_values else None
        rows.append(
            {
                "example_id": f"{prompt_id}/{example_id}",
                "slice": _slice_name(example),
                "input": canonical_json(
                    example.inputs if example else {"example_id": example_id}
                ).decode("utf-8"),
                "expected": _expected_text(example),
                "source_output": source.text if source else None,
                "target_output": target.text if target else None,
                "passed": passed,
                "score": row_score,
                "metrics": _example_metrics(source, target, pair_scores),
                "error": "; ".join(errors) if errors else None,
            }
        )
    return rows


def _slice_name(example: SuiteExample | None) -> str | None:
    if example is None or not example.tags:
        return None
    return example.tags[0]


def _expected_text(example: SuiteExample | None) -> str | None:
    if example is None or example.expected is None:
        return None
    return canonical_json(example.expected).decode("utf-8")


def _example_metrics(
    source: Call | None,
    target: Call | None,
    scores: list[EvalRecord],
) -> dict[str, Any]:
    return {
        "latency_ms_source": source.latency_ms if source else None,
        "latency_ms_target": target.latency_ms if target else None,
        "cost_usd_source": source.cost_usd if source else None,
        "cost_usd_target": target.cost_usd if target else None,
        "scores": [
            {
                "evaluator_name": score.evaluator_name,
                "source_score": score.source_score,
                "target_score": score.target_score,
                "delta": score.delta,
                "error": score.error,
            }
            for score in scores
        ],
    }


def _build_aggregate(
    examples: list[dict[str, Any]],
    *,
    state_started: datetime,
    state_ended: datetime | None,
) -> dict[str, Any]:
    total = len(examples)
    passed = sum(1 for example in examples if example["passed"])
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        if example.get("slice"):
            by_slice[str(example["slice"])].append(example)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "duration_seconds": _duration_seconds(state_started, state_ended),
        "slices": [
            {
                "name": name,
                "total": len(rows),
                "passed": sum(1 for row in rows if row["passed"]),
                "pass_rate": (sum(1 for row in rows if row["passed"]) / len(rows) if rows else 0.0),
            }
            for name, rows in sorted(by_slice.items())
        ],
    }


def _duration_seconds(started: datetime, ended: datetime | None) -> float:
    if ended is None:
        return 0.0
    return max(0.0, (ended - started).total_seconds())


def _git_metadata(env: Mapping[str, str]) -> dict[str, Any]:
    sha = (env.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "").lower()
    if not _GIT_SHA_RE.match(sha):
        raise BundleError(
            "could not determine a valid 40-character git SHA; run inside git or set GITHUB_SHA",
        )
    branch = (
        env.get("GITHUB_HEAD_REF")
        or env.get("GITHUB_REF_NAME")
        or _git("rev-parse", "--abbrev-ref", "HEAD")
        or "local"
    )
    return {"git_sha": sha, "branch": branch, "pr_number": _github_pr_number(env)}


def _github_pr_number(env: Mapping[str, str]) -> int | None:
    ref = env.get("GITHUB_REF", "")
    match = re.match(r"refs/pull/([1-9][0-9]*)/", ref)
    if match:
        return int(match.group(1))
    event_path = env.get("GITHUB_EVENT_PATH")
    if event_path:
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return None
        number = payload.get("number") if isinstance(payload, dict) else None
        return number if isinstance(number, int) and number > 0 else None
    return None


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _format_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _compress_with_converged_size(bundle: dict[str, Any]) -> bytes:
    compressed = _compress_with_level(bundle, compresslevel=9, max_iterations=64)
    if compressed is not None:
        return compressed
    # In rare cases DEFLATE can oscillate by a byte as the embedded
    # size_bytes value changes. Store mode makes size a simple function of
    # JSON length, which reliably converges once the digit count is stable.
    compressed = _compress_with_level(bundle, compresslevel=0, max_iterations=12)
    if compressed is not None:
        return compressed
    raise BundleError("failed to converge bundle size metadata")


def _compress_with_level(
    bundle: dict[str, Any],
    *,
    compresslevel: int,
    max_iterations: int,
) -> bytes | None:
    seen: set[int] = set()
    for _ in range(max_iterations):
        compressed = gzip.compress(canonical_json(bundle), compresslevel=compresslevel, mtime=0)
        size = len(compressed)
        if bundle["manifest"]["size_bytes"] == size:
            return compressed
        if size in seen:
            return None
        seen.add(size)
        bundle["manifest"]["size_bytes"] = size
    return None


def _validate_bundle(bundle: dict[str, Any]) -> None:
    schema_path = resources.files("evalshift.hosted").joinpath("bundle_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema["bundle"]).validate(bundle)
    except JsonSchemaValidationError as exc:
        raise BundleError(f"bundle failed schema validation: {exc.message}") from exc


__all__ = [
    "BUNDLE_FILENAME",
    "BundleBuildResult",
    "BundleError",
    "build_bundle",
    "canonical_json",
    "compute_hash",
    "load_bundle",
]
