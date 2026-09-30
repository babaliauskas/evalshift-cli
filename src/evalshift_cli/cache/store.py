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

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import delete, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import SingletonThreadPool, StaticPool

from evalshift_cli.cache.schema import (
    Base,
    CachedCall,
    create_engine,
    default_database_url,
)
from evalshift_cli.evaluators.tool_models import ToolTrace

DEFAULT_TTL_DAYS: int = 7


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """The cache-side view of a previously-completed LLM call.

    ``trace`` is set only for a tool-calling round: the parsed
    :class:`ToolTrace` the live call produced, restored field for field.
    Text-only responses (and every row written before tool calls were
    cached) carry ``None``.
    """

    response_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    created_at: datetime
    finish_reason: str | None = None
    trace: ToolTrace | None = None


def cache_key(
    *,
    model_id: str,
    prompt_text: str,
    inputs: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
    history: Sequence[Mapping[str, str]] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    tools_payload: Sequence[Mapping[str, Any]] | None = None,
    round_index: int | None = None,
    sample_index: int | None = None,
) -> str:
    """Compute the SHA-256 cache key for a call.

    Inputs are serialised with ``sort_keys=True`` so that ``{"a": 1, "b":
    2}`` and ``{"b": 2, "a": 1}`` produce the same key — dict ordering
    must not affect cache identity.

    Args:
        history: Multi-turn conversation prefix (recorded turns dispatched
            ahead of ``prompt_text``); the tool path passes the round's whole
            dispatched message list here instead (history, current turn and the
            teacher-forced recorded rounds), so every byte the provider sees is
            keyed. Included in the hashed payload only when not ``None``, so
            single-turn calls (``history=None``) produce byte-identical keys to
            before this parameter existed — existing cache entries stay valid.
            An empty list is still included (and hashes differently from
            ``None``) since it marks the call as message-mode.
        generation_config: Recorded per-example generation config applied at
            dispatch. Same inclusion rule as ``history``: hashed only when not
            ``None``, so config-less calls keep their pre-existing keys.
        tools_payload: The ``tools`` array exactly as sent to the provider,
            in order (:func:`evalshift_cli.models.client.serialize_tools`, the
            same helper dispatch uses). Same inclusion rule as
            ``history``/``generation_config``: hashed only when not ``None``,
            so a call that never sends a ``tools`` parameter keeps its
            pre-existing key. ``sort_keys`` orders each tool's own keys but
            never the list, so reordering the tools, or changing a name,
            description, schema or ``strict`` flag, changes the key — each of
            those changes what the provider receives. An inline ``tools:``
            list and a ``toolset_ref`` sidecar share entries exactly when they
            resolve to the same tools in the same order.
        round_index: 0-based round of a teacher-forced multi-round replay
            (see :meth:`evalshift_cli.suite.models.SuiteExample.rounds_to_replay`).
            Same inclusion rule as the three above: hashed only when not
            ``None``, so the text path (which never replays rounds and passes
            ``None``) keeps its pre-existing keys. ``0`` is a real round and
            hashes *differently* from ``None``. The tool path passes the real
            round for every round it dispatches, single-shot examples
            included, so a tool-calling key can never collide with a text
            one.
        sample_index: 0-based sample of a repeated-sampling run
            (``defaults.samples_per_example > 1``). Same inclusion rule as
            ``round_index``: hashed only when not ``None``, so every
            single-sample run keeps its pre-existing keys. The orchestrator
            passes ``None`` whenever ``samples_per_example == 1`` and the real
            index otherwise — with the cache on, the second sample of an
            example would otherwise be served from the first's cached
            response, and repeating the call is the whole point.
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
    if tools_payload is not None:
        payload["tools_payload"] = [dict(t) for t in tools_payload]
    if round_index is not None:
        payload["round_index"] = round_index
    if sample_index is not None:
        payload["sample_index"] = sample_index
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


class CacheStore:
    """Async wrapper around the on-disk SQLite cache.

    Construct via :meth:`open` to get a fully-initialised store with the
    schema created. The class manages its own engine and sessionmaker;
    callers shouldn't reach into either.

    Every operation is safe to call from concurrent tasks. On an engine
    whose pool hands out a single shared DBAPI connection (an in-memory
    ``sqlite+aiosqlite:///:memory:`` URL gets a ``StaticPool``), sessions
    are serialised: overlapping sessions there would share one transaction,
    and one session's rollback-on-return would discard another's
    uncommitted write. File-backed databases use a real connection pool and
    run unserialised.
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
        self._shared_connection_lock: asyncio.Lock | None = (
            asyncio.Lock() if _pool_shares_one_connection(engine) else None
        )

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
        try:
            await _ensure_schema(engine)
        except BaseException:
            # No store owns the engine yet, so nothing else would close it.
            await engine.dispose()
            raise
        return cls(engine, ttl=ttl)

    async def close(self) -> None:
        """Dispose of the underlying engine."""
        await self._engine.dispose()

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        """Open a session, holding the shared-connection lock when there is one.

        The lock spans the whole session, including its close: the pool's
        rollback-on-return runs on close, and on a shared connection that
        rollback is exactly what would discard another session's pending
        write.
        """
        guard: AbstractAsyncContextManager[None] = (
            self._shared_connection_lock
            if self._shared_connection_lock is not None
            else nullcontext()
        )
        async with guard, self._sessionmaker() as session:
            yield session

    async def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key`` or ``None`` on miss/expiry."""
        cutoff = _utcnow() - self._ttl
        async with self._session() as session:
            stmt = select(CachedCall).where(CachedCall.cache_key == key)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            # SQLite without timezone storage returns naive datetimes; treat
            # them as UTC for the comparison.
            created_at = _ensure_utc(row.created_at)
            if created_at < cutoff:
                return None
            trace: ToolTrace | None = None
            if row.trace_json is not None:
                try:
                    trace = ToolTrace.model_validate_json(row.trace_json)
                except ValidationError:
                    # Written by a version whose trace shape this one cannot
                    # read: re-dispatch rather than fail the run.
                    return None
            return CachedResponse(
                response_text=row.response_text,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cost_usd=row.cost_usd,
                latency_ms=row.latency_ms,
                created_at=created_at,
                finish_reason=row.finish_reason,
                trace=trace,
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
        trace: ToolTrace | None = None,
    ) -> None:
        """Insert (or replace) a cache entry.

        ``trace`` is the parsed tool trace of a tool-calling round, stored as
        JSON beside ``response_text`` and restored by :meth:`get`; leave it
        ``None`` for a text-only response.

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
            "trace_json": trace.model_dump_json() if trace is not None else None,
            "created_at": _utcnow(),
        }
        async with self._session() as session:
            stmt = sqlite_insert(CachedCall).values(cache_key=key, **values)
            await session.execute(
                stmt.on_conflict_do_update(index_elements=["cache_key"], set_=values),
            )
            await session.commit()

    async def clear(self) -> int:
        """Delete every entry from the cache. Returns the number of rows removed."""
        async with self._session() as session:
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
        async with self._session() as session:
            stmt = select(CachedCall.cache_key)
            return len((await session.execute(stmt)).scalars().all())


def _pool_shares_one_connection(engine: AsyncEngine) -> bool:
    """Whether every session on ``engine`` gets the same DBAPI connection.

    SQLAlchemy picks a single-connection pool for in-memory SQLite (a
    ``StaticPool`` under aiosqlite), so there is no transaction isolation
    between sessions at all. Overlapping sessions on it only ever worked by
    scheduling luck: SQLAlchemy 2.1 moved aiosqlite onto the generic asyncio
    cursor adapter (sqlalchemy#10415), which reorders the awaits so one
    session's rollback-on-return routinely lands between another's
    ``INSERT`` and ``COMMIT``.
    """
    return isinstance(engine.pool, StaticPool | SingletonThreadPool)


# Nullable columns added after the table first shipped, with the DDL that adds
# each one to a DB created before it existed. Append-only.
_ADDITIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("finish_reason", "VARCHAR(32)"),
    ("trace_json", "TEXT"),
)


async def _ensure_schema(engine: AsyncEngine) -> None:
    """Create the table and backfill :data:`_ADDITIVE_COLUMNS`, tolerating races.

    Several ``evalshift`` processes can open one cache DB at once (parallel CI
    jobs, a ``run`` next to an ``evaluate``). Each step is check-then-change,
    so another process can make the same change in between: ``create_all``
    then fails with "table … already exists" and a backfill with "duplicate
    column name". Both mean the schema the loser wanted is already there, so
    they are swallowed; any other error is raised. Each change runs in its own
    transaction so a lost race rolls back only that statement.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except OperationalError as exc:
        if not _lost_schema_race(exc, "already exists"):
            raise
    async with engine.connect() as conn:
        columns = await _existing_columns(conn)
    for name, ddl_type in _ADDITIVE_COLUMNS:
        if name in columns:
            continue
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"ALTER TABLE cached_calls ADD COLUMN {name} {ddl_type}"))
        except OperationalError as exc:
            if not _lost_schema_race(exc, "duplicate column name"):
                raise


async def _existing_columns(conn: Any) -> set[str]:
    """Column names of ``cached_calls`` as this connection sees them now.

    ``create_all`` only creates missing *tables*, never alters existing ones,
    and the disposable 7-day cache has no migration framework, so a DB created
    before a column in :data:`_ADDITIVE_COLUMNS` existed is missing it until
    :func:`_ensure_schema` adds it. Existing rows read back ``NULL`` there (not
    truncated, no tool trace), so they keep serving the text path.
    """
    result = await conn.execute(text("PRAGMA table_info(cached_calls)"))
    return {row[1] for row in result.fetchall()}


def _lost_schema_race(exc: OperationalError, message: str) -> bool:
    """Whether ``exc`` is SQLite reporting that a concurrent opener got there first."""
    return message in str(exc.orig)


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
