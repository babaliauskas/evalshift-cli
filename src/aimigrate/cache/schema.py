"""SQLAlchemy schema for the local LLM-response cache.

We use a single table — :class:`CachedCall` — keyed on the SHA-256 of
(model + canonicalised prompt + canonicalised inputs + temperature +
max_tokens). This keeps caching decisions transparent: identical inputs
hash to the same key, period. No fuzzy matching, no embeddings, no
similarity searches; the simplest thing that could possibly work, per
the PDF's guidance for the MVP.

The cache lives at ``~/.aimigrate/cache.db`` by default. Tests pass an
in-memory SQLite URL via the ``database_url`` parameter on the engine
factory; never reach for the real DB in unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_CACHE_PATH: Path = Path.home() / ".aimigrate" / "cache.db"


def _utcnow() -> datetime:
    """UTC ``now()`` (named for monkeypatching in TTL tests)."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for the cache schema."""


class CachedCall(Base):
    """One row per cached LLM response.

    Attributes:
        cache_key: SHA-256 hex digest of the canonicalised request. The
            primary key. See :func:`aimigrate.cache.store.cache_key`.
        model_id: The canonical model id used for the call (matches
            :attr:`aimigrate.models.registry.ModelMetadata.id`).
        prompt_text: The fully-rendered prompt sent to the model. Stored
            verbatim so we can audit cache hits.
        inputs_json: Canonical-JSON-encoded ``inputs`` mapping.
        response_text: The model's response text.
        input_tokens / output_tokens: From the provider response.
        cost_usd: Computed via ``litellm.completion_cost`` at write time.
        latency_ms: Wall time of the original live call. Useful for
            cost/perf reporting on cached results.
        created_at: When the cache row was written. Driver of TTL.
    """

    __tablename__ = "cached_calls"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_json: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )


def default_database_url(path: Path | None = None) -> str:
    """Return the SQLAlchemy URL for the on-disk cache DB.

    Args:
        path: Override the default location (mostly for tests). When
            ``None``, uses :data:`DEFAULT_CACHE_PATH`.
    """
    target = path if path is not None else DEFAULT_CACHE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{target}"


def create_engine(database_url: str | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the cache DB.

    Args:
        database_url: SQLAlchemy URL. Pass ``"sqlite+aiosqlite:///:memory:"``
            for tests. When ``None``, defaults to the on-disk cache.
    """
    url = database_url if database_url is not None else default_database_url()
    return create_async_engine(url, future=True)


__all__ = [
    "DEFAULT_CACHE_PATH",
    "Base",
    "CachedCall",
    "create_engine",
    "default_database_url",
]
