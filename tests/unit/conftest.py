"""Fixtures shared by the insights suites.

The builders live in :mod:`tests.unit.insights_factories`; only the fixture
wiring is here, so a test that needs a variant (a failing verdict, a live-latency
run) imports the builder and overrides one keyword.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalshift.insights.facts import Facts, build_facts
from tests.unit.insights_factories import passing_run_kwargs, sample_run_kwargs


@pytest.fixture
def sample_run() -> dict[str, Any]:
    """The reference run from the spec: 21 examples, +102% cost, d = −2.51."""
    return sample_run_kwargs()


@pytest.fixture
def passing_run() -> dict[str, Any]:
    """The same shape with no negative deltas anywhere."""
    return passing_run_kwargs()


@pytest.fixture
def sample_facts(sample_run: dict[str, Any]) -> Facts:
    """The reference run's facts, already rendered."""
    return build_facts(**sample_run)
