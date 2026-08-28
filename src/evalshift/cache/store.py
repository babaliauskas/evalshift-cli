"""Async cache store backed by SQLite.

Two responsibilities:

* Compute deterministic cache keys (SHA-256 over canonicalised JSON of
  model + prompt + inputs + temperature + max_tokens).
* Get/put/clear cached LLM responses, with a configurable TTL.

The store is fully async because the orchestrator (Phase 4) will issue
many cache lookups concurrently from inside its asyncio loop. Sync
callers should run them via ``asyncio.run`` — which is exactly what the
``evalshift cache clear`` CLI command does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from evalshift.cache.schema import (
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
    finish_reason: str | None = None


def cache_key(
    *,
    model_id: str,
    prompt_text: str,
    inputs: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
    history: Sequence[Mapping[str, str]] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    toolset_fingerprint: str | None = None,
) -> str:
    """Compute the SHA-256 cache key for a call.

    Inputs are serialised with ``sort_keys=True`` so that ``{"a": 1, "b":
    2}`` and ``{"b": 2, "a": 1}`` produce the same key — dict ordering
    must not affect cache identity.

    Args:
        history: Multi-turn conversation prefix (recorded turns dispatched
            ahead of ``prompt_text``). Included in the hashed payload only
            when not ``None``, so single-turn calls (``history=None``)
            produce byte-identical keys to before this parameter existed —
            existing cache entries stay valid. An empty list is still
            included (and hashes differently from ``None``) since it marks
            the call as message-mode.
        generation_config: Recorded per-example generation config applied at
            dispatch. Same inclusion rule as ``history``: hashed only when not
            ``None``, so config-less calls keep their pre-existing keys.
        toolset_fingerprint: Content-address of the toolset this call was
            dispatched with (``"sha256:<hex>"`` from
            :func:`evalshift.captures.toolset.fingerprint_tools`). Same
            inclusion rule as ``history``/``generation_config``: hashed only
            when not ``None``, so a call that never sends a ``tools``
            parameter to the provider at all keeps its pre-existing key. An
            example's toolset, whether spelled as an inline ``tools:`` list
            or a ``toolset_ref`` sidecar, resolves to the same fingerprint
            before it reaches this function — see
            :func:`evalshift.runner.orchestrator._fingerprint_toolset` — so
            the two spellings of one toolset never fork the cache, while two
            genuinely different toolsets always produce different keys.
    """
    payload: dict[str, Any] = {
        "model_id": model_id,
        "prompt_text": prompt_text,
        "inputs": inputs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if history is not None:
        payload["history"] = [dict(m) for m in history]
    if generation_config is not None:
        payload["generation_config"] = dict(generation_config)
    if toolset_fingerprint is not None:
        payload["toolset_fingerprint"] = toolset_fingerprint
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


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
            await _ensure_finish_reason_column(conn)
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
                finish_reason=row.finish_reason,
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
        finish_reason: str | None = None,
    ) -> None:
        """Insert (or replace) a cache entry.

        A single atomic ``INSERT ... ON CONFLICT DO UPDATE``: concurrent
        callers routinely miss the same key and race to write it back (the
        evaluate stage scores many pairs at once, and identical model
        outputs hash to identical keys). Delete-then-insert would raise a
        UNIQUE violation on the loser of that race; the payloads are
        identical, so last-writer-wins is the correct outcome.
        """
        values: dict[str, Any] = {
            "model_id": model_id,
            "prompt_text": prompt_text,
            "inputs_json": json.dumps(dict(inputs), sort_keys=True, default=str),
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "created_at": _utcnow(),
        }
        async with self._sessionmaker() as session:
            stmt = sqlite_insert(CachedCall).values(cache_key=key, **values)
            await session.execute(
                stmt.on_conflict_do_update(index_elements=["cache_key"], set_=values),
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


async def _ensure_finish_reason_column(conn: Any) -> None:
    """Additively backfill the ``finish_reason`` column on pre-existing DBs.

    ``create_all`` only creates missing *tables*, never alters existing ones,
    and the disposable 7-day cache has no migration framework. A cache DB
    created before this column existed would be missing it, so probe
    ``PRAGMA table_info`` and ``ALTER TABLE ... ADD COLUMN`` when absent. A
    fresh DB already has the column via ``create_all``, making this a no-op.
    """
    result = await conn.execute(text("PRAGMA table_info(cached_calls)"))
    columns = {row[1] for row in result.fetchall()}
    if "finish_reason" not in columns:
        await conn.execute(text("ALTER TABLE cached_calls ADD COLUMN finish_reason VARCHAR(32)"))


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
