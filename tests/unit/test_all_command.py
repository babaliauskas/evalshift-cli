"""Tests for ``evalshift all``: the single-shot pipeline command.

Two flavours:

* Pure unit tests for the verdict picker (``_compose_verdict``) and
  the rendering helpers (``_bar``, ``_evaluator_family_summary``).
* One end-to-end smoke test that drives the full pipeline via
  :class:`CliRunner` against a temporary scaffold + a stubbed
  :class:`ModelClient`. We assert the rendered output mentions every
  stage glyph and that ``report.html`` is written under the run dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift.analysis.statistics import ComparisonResult
from evalshift.cli.commands.all import (
    _bar,
    _compose_verdict,
    _evaluator_family_summary,
)
from evalshift.cli.commands.analyze import AnalyzeResult
from evalshift.cli.commands.analyze import run_analyze as _real_run_analyze
from evalshift.cli.main import app
from evalshift.config.models import (
    EvalShiftConfig,
    EvaluatorsConfig,
    LLMJudgeConfig,
    PromptDefinition,
    SemanticEvaluatorConfig,
    StructuralEvaluatorConfig,
)
from evalshift.models.client import CompletionResult, ModelClient

runner = CliRunner()


# ---------------------------------------------------------------------------
# _bar: block-bar renderer
# ---------------------------------------------------------------------------


class TestBar:
    def test_empty_when_total_zero(self) -> None:
        assert _bar(0, 0) == "▱" * 10

    def test_full_when_complete(self) -> None:
        assert _bar(80, 80) == "▰" * 10

    def test_half_when_half(self) -> None:
        out = _bar(40, 80)
        assert out.count("▰") == 5
        assert out.count("▱") == 5

    def test_clamps_overflow(self) -> None:
        # If completed > total (shouldn't happen, but defensive).
        out = _bar(100, 80)
        assert out == "▰" * 10


# ---------------------------------------------------------------------------
# _evaluator_family_summary
# ---------------------------------------------------------------------------


def _cfg_with(**evaluators: Any) -> EvalShiftConfig:
    return EvalShiftConfig(
        prompts=[
            PromptDefinition(
                id="p1",
                detection="manual",
                content="hello {name}",
                variables=["name"],
            ),
        ],
        evaluators=EvaluatorsConfig(**evaluators),
    )


class TestFamilySummary:
    def test_just_structural(self) -> None:
        cfg = _cfg_with(
            structural=[StructuralEvaluatorConfig(type="length", min_chars=1)],
        )
        assert _evaluator_family_summary(cfg) == "structural"

    def test_full_house(self) -> None:
        cfg = _cfg_with(
            structural=[StructuralEvaluatorConfig(type="length", min_chars=1)],
            semantic=SemanticEvaluatorConfig(
                embedding_model="text-embedding-3-small",
            ),
            llm_judge=[
                LLMJudgeConfig(
                    criterion_name="helpfulness",
                    criterion_prompt="Is this helpful?",
                ),
            ],
        )
        assert _evaluator_family_summary(cfg) == "structural · semantic · judge"

    def test_empty_returns_placeholder(self) -> None:
        cfg = _cfg_with()
        assert _evaluator_family_summary(cfg) == "(none)"


# ---------------------------------------------------------------------------
# _compose_verdict
# ---------------------------------------------------------------------------


def _comp(
    *,
    severity: str,
    effect_size: float = 0.3,
    delta: float = 0.05,
    p_corr: float = 0.01,
    evaluator: str = "length",
) -> ComparisonResult:
    return ComparisonResult(
        prompt_id="p1",
        evaluator_name=evaluator,
        slice_name="all",
        n=20,
        test="paired_t",
        statistic=1.5,
        p_value=0.01,
        p_value_corrected=p_corr,
        effect_size=effect_size,
        effect_size_ci_low=effect_size - 0.1,
        effect_size_ci_high=effect_size + 0.1,
        delta_avg_score=delta,
        severity=severity,  # type: ignore[arg-type]
        notes=[],
    )


class TestVerdict:
    def test_critical_regression_takes_priority(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.4, delta=0.1),
            _comp(severity="critical", effect_size=-0.9, delta=-0.2),
        ]
        verdict = _compose_verdict(comparisons)
        assert "regressed" in verdict.headline.plain
        # Detail line should reflect the regression's numbers (delta = -0.200).
        assert verdict.detail is not None
        assert "-0.200" in verdict.detail.plain

    def test_improved_only_picks_largest_effect(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.2, delta=0.05, evaluator="length"),
            _comp(severity="improved", effect_size=0.6, delta=0.1, evaluator="judge"),
        ]
        verdict = _compose_verdict(comparisons)
        assert "significantly better" in verdict.headline.plain
        # The detail uses the d=0.6 row.
        assert verdict.detail is not None
        assert "0.60" in verdict.detail.plain

    def test_no_change_when_only_none_severity(self) -> None:
        comparisons = [_comp(severity="none", effect_size=0.05, delta=0.0)]
        verdict = _compose_verdict(comparisons)
        assert "no significant change" in verdict.headline.plain
        assert verdict.detail is None
        assert verdict.regression_callout is None

    def test_minor_regressions_show_callout(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.4, delta=0.1),
            _comp(severity="medium", effect_size=-0.4, delta=-0.05, evaluator="judge"),
            _comp(severity="low", effect_size=-0.25, delta=-0.02, evaluator="length"),
        ]
        verdict = _compose_verdict(comparisons)
        assert "significantly better" in verdict.headline.plain
        assert verdict.regression_callout is not None
        callout = verdict.regression_callout.plain
        assert "2 sub-metrics regressed" in callout
        # Worst minor was the medium one (d=-0.4).
        assert "judge" in callout


# ---------------------------------------------------------------------------
# End-to-end smoke through CliRunner
# ---------------------------------------------------------------------------


def _scaffold(tmp_path: Path) -> None:
    """Lay out a minimal valid project under ``tmp_path``."""
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
          concurrency: 4
        evaluators:
          structural:
            - type: length
              min_chars: 1
              max_chars: 200
        """,
        encoding="utf-8",
    )
    rows = "\n".join(f'{{"id": "ex{i}", "inputs": {{"name": "User{i}"}}}}' for i in range(4))
    (tmp_path / "golden.jsonl").write_text(rows + "\n", encoding="utf-8")


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        return CompletionResult(
            text="A short polite reply.",
            model_id=str(kwargs["model"]),
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.0,
            latency_ms=10,
        )

    monkeypatch.setattr(ModelClient, "complete", fake_complete)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


class TestEndToEnd:
    def test_all_runs_full_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        # Final verdict block is printed.
        assert (
            "candidate is significantly better" in result.output
            or "no significant change" in result.output
            or "candidate regressed" in result.output
        )
        # Report wrote an HTML file.
        runs = list((tmp_path / ".evalshift" / "runs").iterdir())
        assert len(runs) == 1
        assert (runs[0] / "report.html").exists()
        assert (runs[0] / "scores.jsonl").exists()
        assert (runs[0] / "analysis.json").exists()

    def test_all_pushes_before_gate_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        pushed: list[str] = []

        def fake_push_local_run(
            *,
            run_id: str,
            config_path: Path,
            suite_path: Path,
            runs_base: Path,
            console: Any,
        ) -> Any:
            pushed.append(run_id)

            class Result:
                view_url = "https://app.test/app/acme/project/runs/" + run_id

            return Result()

        monkeypatch.setattr("evalshift.cli.commands.all.push_local_run", fake_push_local_run)

        def fake_run_analyze(*, run_id: str, config_path: Path, runs_base: Path) -> AnalyzeResult:
            real = _real_run_analyze(run_id=run_id, config_path=config_path, runs_base=runs_base)
            return AnalyzeResult(
                run_id=real.run_id,
                output_path=real.output_path,
                comparisons=(
                    _comp(
                        severity="critical",
                        effect_size=-1.0,
                        delta=-0.4,
                    ),
                ),
                n_records=real.n_records,
            )

        monkeypatch.setattr("evalshift.cli.commands.all.run_analyze", fake_run_analyze)

        result = runner.invoke(app, ["all", "--yes", "--push", "--gate", "critical"])

        assert result.exit_code == 1
        assert len(pushed) == 1
        assert "https://app.test/app/acme/project/runs/" in result.output

    def test_all_aborts_on_missing_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        # Deliberately don't set the API key; ensure it isn't leaked in.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])
        assert result.exit_code == 1
        assert "missing API key" in result.output
