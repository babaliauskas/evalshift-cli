"""Parser for ``detection: manual`` prompts.

Trivial: returns the inline ``content`` field verbatim. Exists mainly for
symmetry with :class:`PythonStringParser` so the orchestrator can dispatch
to a single :class:`PromptParser` interface.
"""

from __future__ import annotations

from pathlib import Path

from aimigrate.config.models import PromptDefinition
from aimigrate.parsers.base import PromptParseError, PromptTemplate


class ManualParser:
    """Returns the inline ``content`` from a manual prompt definition."""

    def parse(
        self,
        prompt: PromptDefinition,
        project_root: Path,
    ) -> PromptTemplate:
        if prompt.detection != "manual":
            raise PromptParseError(
                prompt_id=prompt.id,
                kind="invalid_definition",
                summary=f"ManualParser can't handle detection={prompt.detection!r}",
            )
        if prompt.content is None:
            # Pydantic validation should have already caught this, but be
            # defensive — failing late here would be confusing.
            raise PromptParseError(
                prompt_id=prompt.id,
                kind="invalid_definition",
                summary="manual prompt is missing 'content' (this is a bug)",
            )
        return PromptTemplate(
            id=prompt.id,
            content=prompt.content,
            declared_variables=list(prompt.variables),
        )


__all__ = ["ManualParser"]
