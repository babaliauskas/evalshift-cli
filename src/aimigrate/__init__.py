"""AIMigrate: a local-first CLI for safe LLM model migrations.

AIMigrate runs your prompts on two LLM model versions against a golden suite of
inputs, scores the resulting outputs with structural, semantic, and LLM-as-judge
evaluators, and produces a single-file HTML report with statistically rigorous
regression analysis.
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
