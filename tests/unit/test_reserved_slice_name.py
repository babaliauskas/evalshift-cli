"""Phase B3.2: `overall` names the run-level scope and may not name a slice.

`BUNDLE_SPEC.md` has always reserved it — `decision.overall` is the whole-run
summary and `BudgetResult.scope` defaults to `"overall"` — and the server now
rejects a bundle whose `decision.slices` carries that key. A slice name reaches
the bundle from two authoring surfaces -- a suite example tag and a
`migration_policy.slices` key -- so both are refused here, where the error can
point at the line the user wrote rather than at an upload that already happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evalshift_cli.config.models import MigrationPolicy
from evalshift_cli.suite.tags import RESERVED_SLICE_NAME
from tests.unit.suite_examples import suite_example


def test_reserved_name_is_the_scope_the_bundle_spec_names() -> None:
    assert RESERVED_SLICE_NAME == "overall"


def test_suite_example_rejects_a_tag_named_overall() -> None:
    """Tags become slice names directly — see `analysis.slicing._slices_of`."""
    with pytest.raises(ValidationError, match="overall"):
        suite_example(id="ex1", tags=["captured", "overall"])


def test_suite_example_still_accepts_ordinary_tags() -> None:
    assert suite_example(id="ex1", tags=["captured", "refunds"]).tags == ["captured", "refunds"]


def test_migration_policy_rejects_a_per_slice_override_keyed_overall() -> None:
    """The override key is the slice name, so the same reservation applies."""
    with pytest.raises(ValidationError, match="overall"):
        MigrationPolicy.model_validate({"slices": {"overall": {}}})


def test_migration_policy_accepts_an_ordinary_override_key() -> None:
    policy = MigrationPolicy.model_validate({"slices": {"refunds": {}}})

    assert set(policy.slices) == {"refunds"}


def test_config_load_surfaces_the_reserved_name(tmp_path: Path) -> None:
    """The whole point: a bad slice name is a config error, not a push-time 422."""
    from evalshift_cli.config.loader import ConfigError, load_config

    (tmp_path / "evalshift.yaml").write_text(
        """
version: 1
project: acme/model-migration
prompts:
  - id: greet
    detection: manual
    content: "Hello {name}"
    variables: [name]
defaults:
  source_model: gemini/gemini-2.5-flash
  target_model: gemini/gemini-3.1-flash-lite-preview
evaluators:
  structural:
    - type: length
      min_chars: 1
migration_policy:
  slices:
    overall:
      max_overall_regression_rate: 0.05
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="overall"):
        load_config(tmp_path / "evalshift.yaml")


def test_an_ordinary_slice_name_still_loads(tmp_path: Path) -> None:
    """Sanity: the guard rejects one literal, not per-slice budgets."""
    from evalshift_cli.config.loader import load_config

    (tmp_path / "evalshift.yaml").write_text(
        """
version: 1
project: acme/model-migration
prompts:
  - id: greet
    detection: manual
    content: "Hello {name}"
    variables: [name]
defaults:
  source_model: gemini/gemini-2.5-flash
  target_model: gemini/gemini-3.1-flash-lite-preview
evaluators:
  structural:
    - type: length
      min_chars: 1
migration_policy:
  slices:
    refunds:
      max_overall_regression_rate: 0.05
""",
        encoding="utf-8",
    )

    policy = load_config(tmp_path / "evalshift.yaml").migration_policy

    assert policy is not None
    assert set(policy.slices) == {"refunds"}
