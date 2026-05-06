"""Hard-coded registry of LLM models AIMigrate knows about.

This module is the single source of truth for the model identifiers users
can put into ``aimigrate.yaml`` or pass on the CLI. We accept two forms:

1. **Canonical LiteLLM IDs** — e.g. ``gemini/gemini-2.5-flash``,
   ``anthropic/claude-sonnet-4-5``, ``openai/gpt-4o``. These are the
   identifiers AIMigrate dispatches to LiteLLM with no rewriting.
2. **Friendly aliases** — shorter names that map to a canonical ID.
   These let users write ``claude-4.5-sonnet`` instead of the longer
   provider-prefixed form, and they're the names that show up in
   reports.

The list below is intentionally small (the seven models from the PDF spec
plus a few currently-available ones used as their resolution targets).
We don't store $/Mtoken pricing here because LiteLLM already maintains
that data per model and its own ``litellm.completion_cost`` helper is the
authoritative source — duplicating it here would just go stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

Provider = Literal["anthropic", "openai", "google"]


class UnknownModelError(KeyError):
    """Raised by :func:`get_model` when no canonical id or alias matches."""

    def __init__(self, requested: str, *, suggestions: list[str] | None = None) -> None:
        self.requested = requested
        self.suggestions: list[str] = list(suggestions or [])
        message = f"unknown model: {requested!r}"
        if self.suggestions:
            message += f" (try one of: {', '.join(self.suggestions)})"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Static metadata for a model AIMigrate supports.

    Attributes:
        id: The canonical LiteLLM model identifier (provider-prefixed).
            This is what AIMigrate sends to ``litellm.acompletion``.
        provider: ``"anthropic"`` | ``"openai"`` | ``"google"``.
        display_name: Human-friendly name shown in reports.
        aliases: Other strings that resolve to this model. Aliases must
            be globally unique across the registry.
        default_temperature: Used when the caller doesn't override.
            Defaults to 0 because we want deterministic outputs in the
            evaluation hot path.
        default_max_tokens: Sane default for completion length. Users can
            override per-call.
    """

    id: str
    provider: Provider
    display_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_temperature: float = 0.0
    default_max_tokens: int = 1024


# Registry. New models go here. Each entry's `id` is the canonical
# LiteLLM identifier AIMigrate dispatches to.
_MODELS: Final[tuple[ModelMetadata, ...]] = (
    # ---- Anthropic -------------------------------------------------------
    # Sonnet 4.5 is the latest Anthropic Sonnet at the time of writing.
    # The PDF's forward-looking "claude-5-sonnet" alias resolves here for
    # now and will get its own row once Anthropic ships Claude 5.
    ModelMetadata(
        id="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        aliases=("claude-4.5-sonnet", "claude-sonnet-4-5", "claude-5-sonnet"),
    ),
    ModelMetadata(
        id="anthropic/claude-opus-4-5",
        provider="anthropic",
        display_name="Claude Opus 4.5",
        aliases=("claude-opus-4-5", "claude-5-opus"),
    ),
    # ---- OpenAI ----------------------------------------------------------
    ModelMetadata(
        id="openai/gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        aliases=("gpt-4o", "gpt-5"),  # gpt-5 mapped forward
    ),
    ModelMetadata(
        id="openai/gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        aliases=("gpt-4o-mini", "gpt-5-mini"),
    ),
    # ---- Google ----------------------------------------------------------
    ModelMetadata(
        id="gemini/gemini-2.5-pro",
        provider="google",
        display_name="Gemini 2.5 Pro",
        aliases=("gemini-2.5-pro",),
    ),
    ModelMetadata(
        id="gemini/gemini-2.5-flash",
        provider="google",
        display_name="Gemini 2.5 Flash",
        aliases=("gemini-2.5-flash",),
    ),
)


def _build_lookup() -> dict[str, ModelMetadata]:
    """Build the canonical-id-or-alias → ``ModelMetadata`` lookup.

    Asserts every alias is globally unique and never collides with a
    canonical id from another row. Run at import time so registry typos
    fail loudly during ``pytest --collect-only`` rather than at runtime.
    """
    lookup: dict[str, ModelMetadata] = {}
    for meta in _MODELS:
        # Canonical ids must be unique.
        if meta.id in lookup and lookup[meta.id] is not meta:
            raise RuntimeError(f"model registry corrupt: id {meta.id!r} maps to two different rows")
        lookup[meta.id] = meta
        for alias in meta.aliases:
            if alias in lookup and lookup[alias] is not meta:
                raise RuntimeError(
                    f"model registry corrupt: alias {alias!r} resolves ambiguously "
                    f"to both {lookup[alias].id!r} and {meta.id!r}"
                )
            lookup[alias] = meta
    return lookup


_LOOKUP: Final[dict[str, ModelMetadata]] = _build_lookup()


def get_model(id_or_alias: str) -> ModelMetadata:
    """Resolve a canonical id or alias to its :class:`ModelMetadata`.

    Args:
        id_or_alias: Either a canonical LiteLLM id
            (``"gemini/gemini-2.5-flash"``) or a friendly alias
            (``"gemini-2.5-flash"``).

    Returns:
        The matching :class:`ModelMetadata`.

    Raises:
        UnknownModelError: If no entry resolves. The error includes a
            short list of supported ids to nudge users toward a fix.
    """
    if id_or_alias in _LOOKUP:
        return _LOOKUP[id_or_alias]
    raise UnknownModelError(
        requested=id_or_alias,
        suggestions=sorted({m.id for m in _MODELS}),
    )


def list_supported() -> list[ModelMetadata]:
    """Return every supported model in registration order.

    Result is a fresh list so callers can't accidentally mutate the
    registry by appending to it.
    """
    seen: set[str] = set()
    out: list[ModelMetadata] = []
    for meta in _MODELS:
        if meta.id in seen:
            continue
        seen.add(meta.id)
        out.append(meta)
    return out


__all__ = [
    "ModelMetadata",
    "Provider",
    "UnknownModelError",
    "get_model",
    "list_supported",
]
