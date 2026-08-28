"""EvalShift: a local-first CLI for safe LLM model migrations.

EvalShift runs your prompts on two LLM model versions against a golden suite of
inputs, scores the resulting outputs with structural, semantic, and LLM-as-judge
evaluators, and produces a single-file HTML report with statistically rigorous
regression analysis.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evalshift")
except PackageNotFoundError:  # running from a source tree with no installed metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
