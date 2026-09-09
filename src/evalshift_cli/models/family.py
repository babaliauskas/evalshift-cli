"""Model-family overlap between an LLM judge and the models it grades.

An LLM-as-judge prefers output that reads like its own — the *self-preference*
bias — so a judge from the same provider as one arm tilts every ``llm_judge``
verdict toward that arm. Nothing in the code can remove that; the honest
thing is to say so wherever the configuration is checked (``doctor``,
``validate``) and wherever the verdicts are read (the report). This module
is the one place the check is defined so those three say the same thing.

"Family" is the registry's :data:`~evalshift_cli.models.registry.Provider`
of the resolved model. Provider ``"other"`` — the resolver could not tell —
never matches anything, including another ``"other"``: two models nobody
recognised are not one family just because neither was recognised.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.models.registry import Provider, resolve_model

_ROLES: tuple[str, ...] = ("source", "target")


@dataclass(frozen=True, slots=True)
class JudgeFamilyOverlap:
    """One judge model that shares a provider with a compared model.

    Attributes:
        judge_model: The judge exactly as configured (alias or canonical id).
        provider: The shared provider, from the registry's taxonomy.
        roles: Which arms share it — ``("source",)``, ``("target",)`` or
            ``("source", "target")``, always in that order.
    """

    judge_model: str
    provider: Provider
    roles: tuple[str, ...]


def shared_judge_family(
    *,
    judge_model: str,
    source_model: str,
    target_model: str,
) -> list[str]:
    """Return the roles whose model shares a provider with ``judge_model``.

    Every id goes through :func:`~evalshift_cli.models.registry.resolve_model`,
    so a registry alias and a bare prefix-inferred id compare on the provider
    they resolve to, not on their spelling.

    Args:
        judge_model: The judge, as configured.
        source_model: The source arm, as configured or as ``--from``.
        target_model: The target arm, as configured or as ``--to``.

    Returns:
        ``["source"]``, ``["target"]``, ``["source", "target"]`` or ``[]``.
        Provider ``"other"`` never matches.
    """
    judge_provider = resolve_model(judge_model).provider
    if judge_provider == "other":
        return []
    arms = {"source": source_model, "target": target_model}
    return [role for role in _ROLES if resolve_model(arms[role]).provider == judge_provider]


def judge_family_overlaps(
    *,
    judge_models: Iterable[str],
    source_model: str,
    target_model: str,
) -> list[JudgeFamilyOverlap]:
    """Overlaps for every distinct judge in ``judge_models``, first-seen order.

    A judge configured under several criteria is reported once.
    """
    out: list[JudgeFamilyOverlap] = []
    seen: set[str] = set()
    for judge in judge_models:
        if judge in seen:
            continue
        seen.add(judge)
        roles = shared_judge_family(
            judge_model=judge,
            source_model=source_model,
            target_model=target_model,
        )
        if roles:
            out.append(
                JudgeFamilyOverlap(
                    judge_model=judge,
                    provider=resolve_model(judge).provider,
                    roles=tuple(roles),
                ),
            )
    return out


def configured_judge_models(cfg: EvalShiftConfig) -> list[str]:
    """Every ``llm_judge[*].judge_model`` the config can score with, deduped.

    Covers the top-level ``evaluators:`` block and each ``suites:`` entry's
    override, resolved through
    :meth:`~evalshift_cli.config.models.EvalShiftConfig.evaluators_for` so a
    suite that replaces the judge family is read the same way scoring reads
    it. ``defaults.judge_model`` is *not* included: it seeds only the
    run-narrative model (``insights_model``), never a pairwise verdict.
    """
    out: list[str] = []
    for suite_name in (None, *cfg.suites):
        for judge in cfg.evaluators_for(suite_name).llm_judge:
            if judge.judge_model not in out:
                out.append(judge.judge_model)
    return out


def describe_overlap(overlap: JudgeFamilyOverlap) -> str:
    """The one sentence ``doctor``, ``validate`` and the report all print."""
    roles = " and ".join(overlap.roles)
    return (
        f"judge `{overlap.judge_model}` shares a model family ({overlap.provider}) with the "
        f"{roles} model — LLM judges tend to prefer their own relatives' output "
        f"(self-preference bias), so llm_judge verdicts may lean toward that arm; prefer a "
        f"judge from a third family."
    )


__all__ = [
    "JudgeFamilyOverlap",
    "configured_judge_models",
    "describe_overlap",
    "judge_family_overlaps",
    "shared_judge_family",
]
