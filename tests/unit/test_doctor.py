"""Tests for ``evalshift doctor`` (:mod:`evalshift.cli.commands.doctor`).

Two layers of testing:

* Unit-level: :func:`run_checks` is a pure function of ``cwd`` and ``env``,
  so most behaviour is asserted there with direct calls — no CLI plumbing.
* CLI-level: a couple of :class:`CliRunner` invocations confirm the exit
  codes and that the rendered table reaches stdout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift.captures.toolset import fingerprint_tools
from evalshift.cli.commands.doctor import (
    CONFIG_FILENAME,
    PROVIDER_KEYS,
    CheckResult,
    _tool_consistency_checks,
    run_checks,
    source_conformance_check,
)
from evalshift.cli.main import app
from evalshift.evaluators.base import EvalRecord
from evalshift.evaluators.failures import BROKEN_HARNESS_CAUSES
from evalshift.evaluators.tool_selection import KIND_CONFORMANCE, KIND_DIVERGENCE

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _by_name(results: list[CheckResult], name: str) -> CheckResult:
    matching = [r for r in results if r.name == name]
    assert len(matching) == 1, f"expected one row named {name!r}, got {matching}"
    return matching[0]


def _empty_env() -> Mapping[str, str]:
    return {}


# ---------------------------------------------------------------------------
# run_checks — pure function tests
# ---------------------------------------------------------------------------


class TestRunChecksPython:
    def test_python_row_is_always_ok(self, tmp_path: Path) -> None:
        results = run_checks(cwd=tmp_path, env=_empty_env())
        py = results[0]
        assert py.status == "ok"
        assert py.name.startswith("Python ")


class TestRunChecksAPIKeys:
    def test_missing_keys_warn(self, tmp_path: Path) -> None:
        results = run_checks(cwd=tmp_path, env=_empty_env())
        for aliases in PROVIDER_KEYS:
            row = _by_name(results, aliases[0])
            assert row.status == "warn"
            assert "not set" in row.detail

    def test_set_keys_ok(self, tmp_path: Path) -> None:
        # Set the primary env var for each provider.
        env = {aliases[0]: "sk-test" for aliases in PROVIDER_KEYS}
        results = run_checks(cwd=tmp_path, env=env)
        for aliases in PROVIDER_KEYS:
            assert _by_name(results, aliases[0]).status == "ok"

    def test_partial_keys(self, tmp_path: Path) -> None:
        env = {"ANTHROPIC_API_KEY": "x"}
        results = run_checks(cwd=tmp_path, env=env)
        assert _by_name(results, "ANTHROPIC_API_KEY").status == "ok"
        assert _by_name(results, "OPENAI_API_KEY").status == "warn"
        # Google provider is displayed under its primary name, GEMINI_API_KEY.
        assert _by_name(results, "GEMINI_API_KEY").status == "warn"

    def test_google_alias_accepted(self, tmp_path: Path) -> None:
        # Legacy GOOGLE_API_KEY still authenticates; shown under GEMINI_API_KEY.
        results = run_checks(cwd=tmp_path, env={"GOOGLE_API_KEY": "x"})
        row = _by_name(results, "GEMINI_API_KEY")
        assert row.status == "ok"
        assert row.detail == "set via GOOGLE_API_KEY"


class TestRunChecksConfig:
    def test_missing_config_warns_not_fails(self, tmp_path: Path) -> None:
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "warn"
        assert "not found" in row.detail
        assert "evalshift init" in row.detail

    def test_valid_config_ok(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "ok"
        assert "1 prompt" in row.detail and "prompts" not in row.detail.replace("prompt", "")

    def test_valid_config_pluralizes(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            """
            prompts:
              - {id: a, detection: manual, content: hi}
              - {id: b, detection: manual, content: hello}
            """,
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert "2 prompts" in row.detail

    def test_invalid_config_fails(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts: []\n",  # empty prompts list — schema rejects
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "fail"

    def test_unparseable_yaml_fails(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - id: a\n  detection: manual\n   content: bad-indent\n",
            encoding="utf-8",
        )
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), CONFIG_FILENAME)
        assert row.status == "fail"


# ---------------------------------------------------------------------------
# CLI-level
# ---------------------------------------------------------------------------


class TestDoctorCLI:
    def _isolate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Force doctor to run inside ``tmp_path`` with no env vars set."""
        monkeypatch.chdir(tmp_path)
        for aliases in PROVIDER_KEYS:
            for key in aliases:
                monkeypatch.delenv(key, raising=False)

    def test_doctor_with_no_config_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Both glyphs should appear: ✓ for Python, ✗ for missing keys/config.
        assert "✓" in result.stdout
        assert "✗" in result.stdout

    def test_doctor_with_valid_config_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "evalshift.yaml" in result.stdout
        assert "1 prompt" in result.stdout

    def test_doctor_with_invalid_config_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        (tmp_path / CONFIG_FILENAME).write_text("prompts: []\n", encoding="utf-8")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1

    def test_doctor_with_set_keys_renders_them_ok(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._isolate(monkeypatch, tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "ANTHROPIC_API_KEY" in result.stdout
        assert "set" in result.stdout


# ---------------------------------------------------------------------------
# Toolset consistency (v0.3) — report the toolset each suite carries, flag a
# suite whose examples carry differing toolsets (legal, but worth seeing).
# ---------------------------------------------------------------------------


def _write_config(cwd: Path, *, suites: dict[str, str] | None = None) -> None:
    """A minimal, valid config, optionally wiring a ``suites:`` block."""
    suites_block = ""
    if suites:
        entries = "".join(
            f"  {name}:\n    source: captured\n    path: {path}\n" for name, path in suites.items()
        )
        suites_block = f"suites:\n{entries}"
    (cwd / CONFIG_FILENAME).write_text(
        f"""
version: 1
prompts:
  - id: replay
    detection: manual
    content: "{{input}}"
    variables: [input]
{suites_block}""",
        encoding="utf-8",
    )


def _example(
    example_id: str,
    *,
    tools: list[dict[str, object]] | None = None,
    toolset_ref: str | None = None,
) -> dict[str, object]:
    """One golden-suite row, carrying inline ``tools`` (default: none) or a ``toolset_ref``."""
    row: dict[str, object] = {"id": example_id, "inputs": {"input": "hi"}}
    if toolset_ref is not None:
        row["toolset_ref"] = toolset_ref
    else:
        row["tools"] = tools if tools is not None else []
    return row


def _write_examples(
    cwd: Path,
    examples: list[dict[str, object]],
    *,
    path: str = "golden.jsonl",
) -> None:
    target = cwd / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(ex) for ex in examples) + "\n",
        encoding="utf-8",
    )


_TOOL_A: dict[str, object] = {
    "name": "search",
    "description": "Search.",
    "input_schema": {"type": "object", "properties": {}},
}
_TOOL_B: dict[str, object] = {
    "name": "notify",
    "description": "Notify.",
    "input_schema": {"type": "object", "properties": {}},
}


class TestToolsetConsistencyCheck:
    """One report row per suite, naming the toolset its examples share, or
    flagging that they don't. Both inline ``tools`` and a ``toolset_ref``
    fingerprint the same way (:func:`~evalshift.captures.toolset.fingerprint_tools`),
    so the two spellings of an identical toolset are never flagged as differing.
    """

    def test_reports_ok_when_every_example_shares_the_empty_toolset(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        _write_examples(tmp_path, [_example("a"), _example("b")])

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "ok"
        assert "no tools" in row.detail
        assert "2 example" in row.detail

    def test_reports_ok_when_every_example_shares_a_non_empty_toolset(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        _write_examples(
            tmp_path,
            [_example("a", tools=[_TOOL_A]), _example("b", tools=[_TOOL_A])],
        )

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "ok"
        assert "no tools" not in row.detail

    def test_warns_when_examples_carry_differing_non_empty_toolsets(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        _write_examples(
            tmp_path,
            [_example("a", tools=[_TOOL_A]), _example("b", tools=[_TOOL_A, _TOOL_B])],
        )

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "warn"
        assert "2 different toolsets" in row.detail

    def test_differing_toolset_warning_does_not_leak_the_plan_task_id(self, tmp_path: Path) -> None:
        """Users reading `doctor` output have no idea what 'Task 8' refers to."""
        _write_config(tmp_path)
        _write_examples(
            tmp_path,
            [_example("a", tools=[_TOOL_A]), _example("b", tools=[_TOOL_A, _TOOL_B])],
        )

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert "Task 8" not in row.detail

    def test_empty_and_non_empty_toolsets_count_as_differing(self, tmp_path: Path) -> None:
        """The exact bug this plan exists to catch: some rows offered nothing, others did."""
        _write_config(tmp_path)
        _write_examples(tmp_path, [_example("a"), _example("b", tools=[_TOOL_A])])

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "warn"

    def test_ref_and_inline_examples_with_identical_tools_do_not_count_as_differing(
        self,
        tmp_path: Path,
    ) -> None:
        ref = fingerprint_tools([_TOOL_A])
        _write_config(tmp_path)
        _write_examples(
            tmp_path,
            [_example("a", tools=[_TOOL_A]), _example("b", toolset_ref=ref)],
        )

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "ok"

    def test_a_toolset_ref_is_never_resolved_off_disk(self, tmp_path: Path) -> None:
        """A ref with no sidecar on disk anywhere must still report cleanly.

        The check compares fingerprints, never loads a sidecar -- unlike
        dispatch, doctor must never fail a suite report just because a
        ``.evalshift/`` capture base or a sidecar happens not to exist yet.
        """
        _write_config(tmp_path)
        _write_examples(tmp_path, [_example("a", toolset_ref="sha256:" + "0" * 64)])

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "ok"

    def test_checks_every_configured_suite_independently(self, tmp_path: Path) -> None:
        clean = ".evalshift/suites/clean/golden.jsonl"
        dirty = ".evalshift/suites/dirty/golden.jsonl"
        _write_config(tmp_path, suites={"clean": clean, "dirty": dirty})
        _write_examples(
            tmp_path,
            [_example("a", tools=[_TOOL_A]), _example("b", tools=[_TOOL_A])],
            path=clean,
        )
        _write_examples(
            tmp_path,
            [_example("c", tools=[_TOOL_A]), _example("d", tools=[_TOOL_B])],
            path=dirty,
        )

        results = _tool_consistency_checks(tmp_path)
        assert _by_name(results, "toolset: clean").status == "ok"
        assert _by_name(results, "toolset: dirty").status == "warn"

    def test_flat_layout_used_when_no_suites_are_wired(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        _write_examples(tmp_path, [_example("a", tools=[_TOOL_A, _TOOL_B])])

        row = _by_name(_tool_consistency_checks(tmp_path), "toolset: golden.jsonl")
        assert row.status == "ok"

    def test_silent_when_no_suite_file_exists(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        assert _tool_consistency_checks(tmp_path) == []

    def test_silent_when_the_suite_cannot_be_loaded(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        (tmp_path / "golden.jsonl").write_text("not a valid suite line\n", encoding="utf-8")
        assert _tool_consistency_checks(tmp_path) == []

    def test_silent_when_no_config_exists(self, tmp_path: Path) -> None:
        _write_examples(tmp_path, [_example("a")])
        assert _tool_consistency_checks(tmp_path) == []

    def test_silent_for_a_suite_with_zero_examples(self, tmp_path: Path) -> None:
        _write_config(tmp_path)
        (tmp_path / "golden.jsonl").write_text("", encoding="utf-8")
        assert _tool_consistency_checks(tmp_path) == []


# ---------------------------------------------------------------------------
# source_conformance_check — the broken-harness signal (S5)
# ---------------------------------------------------------------------------


def _conformance(
    example_id: str,
    source_score: float,
    target_score: float = 1.0,
    *,
    kind: str = KIND_CONFORMANCE,
    error: str | None = None,
) -> EvalRecord:
    """One scored row, shaped as ``ToolSelectionEvaluator._record`` writes it."""
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name="routing",
        kind=kind,
        source_score=source_score,
        target_score=target_score,
        delta=target_score - source_score,
        error=error,
    )


class TestSourceConformanceCheck:
    """A source model that fails ground truth recorded from itself.

    The suite's expectations came from the source model, so the source is the
    one model that should always satisfy them. When it does not, the run
    measured the harness — and every rate it reports describes that, not the
    migration.
    """

    def test_a_clean_run_says_nothing(self) -> None:
        assert source_conformance_check([_conformance(f"e{i}", 1.0) for i in range(10)]) is None

    def test_no_conformance_rows_at_all_says_nothing(self) -> None:
        rows = [_conformance(f"e{i}", 0.0, kind=KIND_DIVERGENCE) for i in range(10)]
        assert source_conformance_check(rows) is None

    def test_ten_of_ten_fires_at_doctor_volume_naming_the_rate(self) -> None:
        """The production run. ``r_20260820_project_insights_143a5f`` hit 10/10."""
        check = source_conformance_check([_conformance(f"e{i}", 0.0, 0.0) for i in range(10)])
        assert check is not None
        assert check.status == "fail"
        assert "10 of 10" in check.detail
        assert "100%" in check.detail
        assert BROKEN_HARNESS_CAUSES in check.detail

    def test_a_source_failure_the_target_passed_still_counts(self) -> None:
        """The exclusion selector under-counts this: ``delta`` is ``+1.0``.

        ``is_shared_ground_truth_miss`` needs ``delta == 0`` and the evaluator
        only tags ``TOOL_GROUND_TRUTH_MISS`` when *both* sides miss, so a suite
        the source fails and the target happens to satisfy is invisible to
        both — and it is exactly as broken.
        """
        rows = [_conformance(f"e{i}", 0.0, 1.0) for i in range(10)]
        assert all(r.delta > 0 for r in rows)
        check = source_conformance_check(rows)
        assert check is not None
        assert "10 of 10" in check.detail

    def test_partial_conformance_is_a_failure(self) -> None:
        """``expected`` scores a fraction of the sequence; 0.8 did not conform."""
        check = source_conformance_check([_conformance(f"e{i}", 0.8) for i in range(10)])
        assert check is not None
        assert "10 of 10" in check.detail

    def test_errored_rows_are_neither_numerator_nor_denominator(self) -> None:
        """An errored row scores a neutral 0.5 — a broken measurement, not a miss."""
        rows = [
            *[_conformance(f"ok{i}", 1.0) for i in range(6)],
            *[_conformance(f"err{i}", 0.5, 0.5, error="upstream call failed") for i in range(6)],
        ]
        assert source_conformance_check(rows) is None

    def test_divergence_rows_never_enter_the_denominator(self) -> None:
        """Divergence fixes the source at 1.0 by construction; it has no ground truth."""
        rows = [
            *[_conformance(f"c{i}", 0.0) for i in range(4)],
            *[_conformance(f"d{i}", 1.0, 0.0, kind=KIND_DIVERGENCE) for i in range(40)],
        ]
        check = source_conformance_check(rows)
        assert check is not None
        assert "4 of 4" in check.detail

    def test_half_the_suite_is_the_boundary(self) -> None:
        """At least half: 5 of 10 fires, 4 of 10 does not."""
        ten = [_conformance(f"e{i}", 1.0) for i in range(10)]
        four_bad = [_conformance(f"e{i}", 0.0) for i in range(4)] + ten[4:]
        five_bad = [_conformance(f"e{i}", 0.0) for i in range(5)] + ten[5:]
        assert source_conformance_check(four_bad) is None
        fired = source_conformance_check(five_bad)
        assert fired is not None
        assert "5 of 10" in fired.detail

    def test_four_rows_is_the_smallest_suite_that_can_fire(self) -> None:
        """A 100% rate over three rows cannot support the sentence it would print.

        The 95% Wilson lower bound on 3/3 is 0.44 — under half — so a
        three-example smoke suite is never accused of being broken. At 4/4 the
        bound clears 0.5 and the claim stands on its own data.
        """
        assert source_conformance_check([_conformance(f"e{i}", 0.0) for i in range(3)]) is None
        check = source_conformance_check([_conformance(f"e{i}", 0.0) for i in range(4)])
        assert check is not None
        assert "4 of 4" in check.detail


# ---------------------------------------------------------------------------
# ci pin — reader >= writer across .github/workflows
# ---------------------------------------------------------------------------


class TestCiPinCheck:
    @staticmethod
    def _workflow(root: Path, with_lines: str) -> None:
        path = root / ".github" / "workflows" / "evalshift.yml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "on: push\njobs:\n  evalshift:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: babaliauskas/evalshift-action@v0\n"
            "        with:\n" + with_lines,
            encoding="utf-8",
        )

    def test_no_row_when_no_workflow_uses_the_action(self, tmp_path: Path) -> None:
        results = run_checks(cwd=tmp_path, env=_empty_env())
        assert [r for r in results if r.name == "ci pin"] == []

    def test_stale_pin_warns_without_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evalshift.cli.commands.doctor.__version__", "1.2.3")
        self._workflow(tmp_path, '          evalshift-version: "0.0.1"\n')
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), "ci pin")
        assert row.status == "warn"
        assert "CI installs evalshift 0.0.1" in row.detail
        assert 'evalshift-version: "1.2.3"' in row.detail

    def test_matching_pin_is_ok(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evalshift.cli.commands.doctor.__version__", "1.2.3")
        self._workflow(tmp_path, '          evalshift-version: "1.2.3"\n')
        row = _by_name(run_checks(cwd=tmp_path, env=_empty_env()), "ci pin")
        assert row.status == "ok"
        assert row.detail == "pinned to 1.2.3"

    def test_doctor_cli_renders_the_warning_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("evalshift.cli.commands.doctor.__version__", "1.2.3")
        self._workflow(tmp_path, "          token: x\n")
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "ci pin" in result.stdout
        assert "default" in result.stdout
