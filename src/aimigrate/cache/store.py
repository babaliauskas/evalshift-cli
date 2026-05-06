"""Async cache store backed by SQLite.

Two responsibilities:

* Compute deterministic cache keys (SHA-256 over canonicalised JSON of
  model + prompt + inputs + temperature + max_tokens).
* Get/put/clear cached LLM responses, with a configurable TTL.

The store is fully async because the orchestrator (Phase 4) will issue
many cache lookups concurrently from inside its asyncio loop. Sync
callers should run them via ``asyncio.run`` — which is exactly what the
``aimigrate cache clear`` CLI command does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from aimigrate.cache.schema import (
    Base,
    CachedCall,
    create_engine,
    default_database_url,
)

DEFAULT_TTL_DAYS: int = 7


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """The cache-side view of a previously-completed LLM call."""

    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    created_at: datetime


def cache_key(
    *,
    model_id: str,
    prompt_text: str,
    inputs: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
) -> str:
    """Compute the SHA-256 cache key for a call.

    Inputs are serialised with ``sort_keys=True`` so that ``{"a": 1, "b":
    2}`` and ``{"b": 2, "a": 1}`` produce the same key — dict ordering
    must not affect cache identity.
    """
    payload = json.dumps(
        {
            "model_id": model_id,
            "prompt_text": prompt_text,
            "inputs": inputs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CacheStore:
    """Async wrapper around the on-disk SQLite cache.

    Construct via :meth:`open` to get a fully-initialised store with the
    schema created. The class manages its own engine and sessionmaker;
    callers shouldn't reach into either.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        ttl: timedelta = timedelta(days=DEFAULT_TTL_DAYS),
    ) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        self._ttl = ttl

    @classmethod
    async def open(
        cls,
        database_url: str | None = None,
        *,
        ttl: timedelta = timedelta(days=DEFAULT_TTL_DAYS),
        path: Path | None = None,
    ) -> CacheStore:
        """Open the cache, creating the schema if necessary.

        Args:
            database_url: Explicit SQLAlchemy URL. Wins over ``path``.
            ttl: How long cached entries are considered fresh.
            path: Override the on-disk location used when
                ``database_url`` is ``None``.
        """
        url = database_url or default_database_url(path)
        engine = create_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return cls(engine, ttl=ttl)

    async def close(self) -> None:
        """Dispose of the underlying engine."""
        await self._engine.dispose()

    async def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key`` or ``None`` on miss/expiry."""
        cutoff = _utcnow() - self._ttl
        async with self._sessionmaker() as session:
            stmt = select(CachedCall).where(CachedCall.cache_key == key)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            # SQLite without timezone storage returns naive datetimes; treat
            # them as UTC for the comparison.
            created_at = _ensure_utc(row.created_at)
            if created_at < cutoff:
                return None
            return CachedResponse(
                response_text=row.response_text,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cost_usd=row.cost_usd,
                latency_ms=row.latency_ms,
                created_at=created_at,
            )

    async def put(
        self,
        key: str,
        *,
        model_id: str,
        prompt_text: str,
        inputs: Mapping[str, Any],
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
    ) -> None:
        """Insert (or replace) a cache entry."""
        async with self._sessionmaker() as session:
            # SQLite-friendly upsert via delete-then-insert.
            await session.execute(
                delete(CachedCall).where(CachedCall.cache_key == key),
            )
            session.add(
                CachedCall(
                    cache_key=key,
                    model_id=model_id,
                    prompt_text=prompt_text,
                    inputs_json=json.dumps(dict(inputs), sort_keys=True, default=str),
                    response_text=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    created_at=_utcnow(),
                ),
            )
            await session.commit()

    async def clear(self) -> int:
        """Delete every entry from the cache. Returns the number of rows removed."""
        async with self._sessionmaker() as session:
            # Count first so we can return a deterministic delete count
            # without depending on Result.rowcount, which isn't part of
            # SQLAlchemy's typed Result API.
            existing = len(
                (await session.execute(select(CachedCall.cache_key))).scalars().all(),
            )
            await session.execute(delete(CachedCall))
            await session.commit()
            return existing

    async def count(self) -> int:
        """Return the total number of rows in the cache (any TTL state)."""
        async with self._sessionmaker() as session:
            stmt = select(CachedCall.cache_key)
            return len((await session.execute(stmt)).scalars().all())


def _utcnow() -> datetime:
    """UTC ``now()`` (separate from schema's so tests can monkeypatch only one)."""
    return datetime.now(UTC)


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes (from SQLite) as UTC for comparisons."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


__all__ = [
    "DEFAULT_TTL_DAYS",
    "CacheStore",
    "CachedResponse",
    "cache_key",
]
