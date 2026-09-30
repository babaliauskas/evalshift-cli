"""Tests for :mod:`evalshift_cli.cache`.

Two layers:

* :func:`cache_key` is pure and tested directly — verifying SHA-256
  determinism and stability under dict-ordering changes.
* :class:`CacheStore` is exercised against an in-memory SQLite database
  for fast, hermetic round-trip tests of put/get/expiry/clear.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine
from typer.testing import CliRunner

from evalshift_cli.cache.schema import Base, create_engine
from evalshift_cli.cache.store import CacheStore, cache_key
from evalshift_cli.cli.main import app
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace

# Use an in-memory database for every test to keep them fast and hermetic.
IN_MEMORY_DB = "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# cache_key — pure function
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_same_inputs_same_key(self) -> None:
        a = cache_key(
            model_id="gemini/gemini-2.5-flash",
            prompt_text="hi",
            inputs={"x": 1},
            temperature=0.0,
            max_tokens=1024,
        )
        b = cache_key(
            model_id="gemini/gemini-2.5-flash",
            prompt_text="hi",
            inputs={"x": 1},
            temperature=0.0,
            max_tokens=1024,
        )
        assert a == b

    def test_different_inputs_different_keys(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={"x": 1},
            temperature=0.0,
            max_tokens=1024,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={"x": 2},
            temperature=0.0,
            max_tokens=1024,
        )
        assert a != b

    def test_dict_order_does_not_affect_key(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={"a": 1, "b": 2},
            temperature=0.0,
            max_tokens=1024,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={"b": 2, "a": 1},
            temperature=0.0,
            max_tokens=1024,
        )
        assert a == b

    def test_temperature_change_changes_key(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.5,
            max_tokens=1024,
        )
        assert a != b

    def test_model_change_changes_key(self) -> None:
        a = cache_key(
            model_id="gemini/gemini-2.5-flash",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        b = cache_key(
            model_id="gemini/gemini-2.5-pro",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        assert a != b

    def test_returns_64_char_hex(self) -> None:
        key = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_no_history_is_byte_identical_to_pre_history_payload(self) -> None:
        """Regression: omitting ``history`` must not change the hashed payload.

        Computes the expected key with the *old* payload shape (no
        ``"history"`` field at all) inline, so this test would fail if a
        future change starts hashing ``"history": None`` unconditionally.
        """
        import hashlib
        import json

        model_id = "gemini/gemini-2.5-flash"
        prompt_text = "hi"
        inputs = {"x": 1}
        temperature = 0.0
        max_tokens = 1024

        old_payload = json.dumps(
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
        expected = hashlib.sha256(old_payload.encode("utf-8")).hexdigest()

        actual = cache_key(
            model_id=model_id,
            prompt_text=prompt_text,
            inputs=inputs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assert actual == expected

        # Explicit history=None must match too.
        actual_explicit_none = cache_key(
            model_id=model_id,
            prompt_text=prompt_text,
            inputs=inputs,
            temperature=temperature,
            max_tokens=max_tokens,
            history=None,
        )
        assert actual_explicit_none == expected

    def test_history_changes_key(self) -> None:
        base = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        with_history = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=[{"role": "user", "content": "earlier turn"}],
        )
        assert base != with_history

    def test_empty_history_list_differs_from_none(self) -> None:
        none_key = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=None,
        )
        empty_key = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=[],
        )
        assert none_key != empty_key

    def test_different_histories_same_current_text_different_keys(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=[{"role": "user", "content": "turn A"}],
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=[{"role": "user", "content": "turn B"}],
        )
        assert a != b

    def test_same_history_same_args_same_key(self) -> None:
        history = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=history,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            history=[dict(m) for m in history],  # fresh copies, same content
        )
        assert a == b

    # -- tools_payload (the tool path keys the tools array exactly as sent) --

    def test_no_tools_payload_is_byte_identical_to_pre_toolset_payload(self) -> None:
        """Regression: omitting ``tools_payload`` must not change the hashed payload.

        Mirrors ``test_no_history_is_byte_identical_to_pre_history_payload``. A call
        dispatched via ``complete``/``complete_messages`` never sends a ``tools``
        parameter to the provider at all, so it must keep its pre-existing cache key
        byte-for-byte -- both when the argument is omitted and when it is passed
        explicitly as ``None``.
        """
        import hashlib
        import json

        model_id = "gemini/gemini-2.5-flash"
        prompt_text = "hi"
        inputs = {"x": 1}
        temperature = 0.0
        max_tokens = 1024

        old_payload = json.dumps(
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
        expected = hashlib.sha256(old_payload.encode("utf-8")).hexdigest()

        base = {
            "model_id": model_id,
            "prompt_text": prompt_text,
            "inputs": inputs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        assert cache_key(**base) == expected  # type: ignore[arg-type]
        assert cache_key(**base, tools_payload=None) == expected  # type: ignore[arg-type]

    def _tool_key(self, tools_payload: list[dict[str, Any]] | None) -> str:
        return cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            tools_payload=tools_payload,
        )

    _SEARCH: ClassVar[dict[str, Any]] = {
        "name": "search_orders",
        "description": "Look up orders.",
        "input_schema": {},
    }
    _REFUND: ClassVar[dict[str, Any]] = {
        "name": "issue_refund",
        "description": "Refund an order.",
        "input_schema": {},
    }

    def test_tools_payload_changes_key(self) -> None:
        assert self._tool_key([self._SEARCH]) != self._tool_key(None)

    def test_empty_tools_payload_differs_from_none(self) -> None:
        assert self._tool_key([]) != self._tool_key(None)

    def test_different_tools_produce_different_keys(self) -> None:
        assert self._tool_key([self._SEARCH]) != self._tool_key([self._REFUND])

    def test_same_tools_same_order_same_key(self) -> None:
        assert self._tool_key([self._SEARCH, self._REFUND]) == self._tool_key(
            [dict(self._SEARCH), dict(self._REFUND)]
        )

    def test_reordered_tools_produce_different_keys(self) -> None:
        """The provider receives the list in order, so order is part of the request."""
        assert self._tool_key([self._SEARCH, self._REFUND]) != self._tool_key(
            [self._REFUND, self._SEARCH]
        )

    def test_dict_key_order_inside_a_tool_does_not_matter(self) -> None:
        reordered = dict(reversed(list(self._SEARCH.items())))
        assert self._tool_key([self._SEARCH]) == self._tool_key([reordered])

    # -- round_index (teacher-forced multi-round replay) --

    def test_no_round_index_is_byte_identical_to_pre_round_payload(self) -> None:
        """Omitting ``round_index`` must not change the hashed payload.

        Same inclusion rule as ``history`` / ``generation_config`` /
        ``tools_payload``: hashed only when not ``None``, so every key
        minted before the round dimension existed stays valid.
        """
        import hashlib
        import json

        model_id = "gemini/gemini-2.5-flash"
        prompt_text = "hi"
        inputs = {"x": 1}
        temperature = 0.0
        max_tokens = 1024

        old_payload = json.dumps(
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
        expected = hashlib.sha256(old_payload.encode("utf-8")).hexdigest()

        assert (
            cache_key(
                model_id=model_id,
                prompt_text=prompt_text,
                inputs=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            == expected
        )
        assert (
            cache_key(
                model_id=model_id,
                prompt_text=prompt_text,
                inputs=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
                round_index=None,
            )
            == expected
        )

    def test_round_index_zero_differs_from_none(self) -> None:
        """``0`` is a real round, not "no round": it must hash differently to ``None``."""
        base = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            round_index=None,
        )
        round_zero = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            round_index=0,
        )
        assert base != round_zero

    def test_different_rounds_produce_different_keys(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            round_index=0,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            round_index=1,
        )
        assert a != b


# ---------------------------------------------------------------------------
# CacheStore — async round-trip
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store() -> AsyncIterator[CacheStore]:
    s = await CacheStore.open(database_url=IN_MEMORY_DB)
    try:
        yield s
    finally:
        await s.close()


def _put_kwargs() -> dict[str, object]:
    return {
        "model_id": "gemini/gemini-2.5-flash",
        "prompt_text": "Hi {name}",
        "inputs": {"name": "Alex"},
        "response_text": "Hello Alex!",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.0001,
        "latency_ms": 250,
    }


class TestCacheStore:
    async def test_get_miss_returns_none(self, store: CacheStore) -> None:
        assert await store.get("nonexistent") is None

    async def test_round_trip_put_get(self, store: CacheStore) -> None:
        await store.put("k", **_put_kwargs())
        got = await store.get("k")
        assert got is not None
        assert got.response_text == "Hello Alex!"
        assert got.input_tokens == 10
        assert got.output_tokens == 5
        assert got.cost_usd == pytest.approx(0.0001)
        # finish_reason defaults to None when not supplied.
        assert got.finish_reason is None

    async def test_round_trip_preserves_finish_reason(self, store: CacheStore) -> None:
        await store.put("k", **_put_kwargs(), finish_reason="length")
        got = await store.get("k")
        assert got is not None
        assert got.finish_reason == "length"

    async def test_open_backfills_finish_reason_column(self, tmp_path: Path) -> None:
        # A cache DB created before the finish_reason column existed must be
        # migrated additively on open() rather than crashing on read.
        db_path = tmp_path / "legacy.db"
        url = f"sqlite+aiosqlite:///{db_path}"
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TABLE cached_calls ("
                    "cache_key VARCHAR(64) PRIMARY KEY, model_id VARCHAR(128) NOT NULL, "
                    "prompt_text TEXT NOT NULL, inputs_json TEXT NOT NULL, "
                    "response_text TEXT NOT NULL, input_tokens INTEGER NOT NULL, "
                    "output_tokens INTEGER NOT NULL, cost_usd FLOAT NOT NULL, "
                    "latency_ms INTEGER NOT NULL, created_at DATETIME NOT NULL)"
                )
            )
        await engine.dispose()

        store = await CacheStore.open(database_url=url)
        try:
            await store.put("k", **_put_kwargs(), finish_reason="length")
            got = await store.get("k")
            assert got is not None
            assert got.finish_reason == "length"
        finally:
            await store.close()

    async def test_concurrent_puts_of_one_key_do_not_collide(self, store: CacheStore) -> None:
        # Two in-flight calls can miss the same key and both write it back.
        # The second writer must not blow up the caller with an integrity
        # error — the payloads are identical, so last-writer-wins is fine.
        import asyncio

        kw = _put_kwargs()
        await asyncio.gather(*(store.put("k", **kw) for _ in range(4)))  # type: ignore[arg-type]
        got = await store.get("k")
        assert got is not None
        assert got.response_text == "Hello Alex!"
        assert await store.count() == 1

    async def test_put_replaces_existing_entry(self, store: CacheStore) -> None:
        kw = _put_kwargs()
        await store.put("k", **kw)
        kw["response_text"] = "second response"
        await store.put("k", **kw)
        got = await store.get("k")
        assert got is not None
        assert got.response_text == "second response"
        assert await store.count() == 1

    async def test_expired_entry_returns_none(self, store: CacheStore) -> None:
        # TTL of zero forces immediate expiry on read.
        store_short_ttl = await CacheStore.open(
            database_url=IN_MEMORY_DB,
            ttl=timedelta(seconds=0),
        )
        await store_short_ttl.put("k", **_put_kwargs())
        # The row exists but is "older" than the TTL window (0s).
        assert await store_short_ttl.get("k") is None
        await store_short_ttl.close()

    async def test_clear_removes_every_row(self, store: CacheStore) -> None:
        await store.put("k1", **_put_kwargs())
        await store.put("k2", **_put_kwargs())
        assert await store.count() == 2
        removed = await store.clear()
        assert removed == 2
        assert await store.count() == 0


class TestCacheStoreSingleConnectionPool:
    """An in-memory SQLite URL gets a single-connection pool (``StaticPool``).

    Every session then shares *one* DBAPI connection, so two overlapping
    sessions share one transaction: one session's rollback-on-return (the
    pool's reset when a session closes) discards another session's
    uncommitted ``INSERT``. SQLAlchemy 2.1 moved aiosqlite onto the generic
    asyncio cursor adapter (sqlalchemy#10415), which reorders those awaits so
    the rollback routinely lands inside another put's window — writes were
    silently lost and every later lookup missed. The store must therefore run
    at most one session at a time on such an engine.
    """

    @staticmethod
    async def _memory_engine() -> AsyncEngine:
        engine = create_engine(IN_MEMORY_DB)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return engine

    async def test_concurrent_misses_on_distinct_keys_all_persist(self) -> None:
        # The orchestrator/evaluator pattern: many tasks miss, then write back.
        store = CacheStore(await self._memory_engine())
        keys = [f"k{i}" for i in range(16)]

        async def miss_then_put(key: str) -> None:
            assert await store.get(key) is None
            await store.put(key, **_put_kwargs())  # type: ignore[arg-type]

        try:
            await asyncio.gather(*(miss_then_put(k) for k in keys))
            assert await store.count() == len(keys)
            assert all([await store.get(k) is not None for k in keys])
        finally:
            await store.close()

    async def test_never_overlaps_sessions_on_the_shared_connection(self) -> None:
        # Pins the mechanism, independent of driver scheduling: with one
        # shared DBAPI connection, a second checkout while the first is still
        # out means two transactions are interleaved on it.
        engine = await self._memory_engine()
        store = CacheStore(engine)
        in_flight = 0
        peak = 0

        def on_checkout(*_: object) -> None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)

        def on_checkin(*_: object) -> None:
            nonlocal in_flight
            in_flight -= 1

        pool = engine.sync_engine.pool
        event.listen(pool, "checkout", on_checkout)
        event.listen(pool, "checkin", on_checkin)
        try:
            await asyncio.gather(
                *(store.put(f"k{i}", **_put_kwargs()) for i in range(4)),  # type: ignore[arg-type]
                *(store.get(f"k{i}") for i in range(4)),
                store.count(),
            )
            assert peak == 1
        finally:
            event.remove(pool, "checkout", on_checkout)
            event.remove(pool, "checkin", on_checkin)
            await store.close()


# ---------------------------------------------------------------------------
# `evalshift cache clear` CLI
# ---------------------------------------------------------------------------


runner = CliRunner()


class TestCacheClearCommand:
    def test_clear_runs_against_fresh_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Redirect the default cache path into tmp so we don't touch the
        # user's real ~/.evalshift/cache.db.
        monkeypatch.setattr(
            "evalshift_cli.cache.schema.DEFAULT_CACHE_PATH",
            tmp_path / "cache.db",
        )
        result = runner.invoke(app, ["cache", "clear"])
        assert result.exit_code == 0, result.stdout
        assert "cleared" in result.stdout

    def test_cache_help_lists_clear(self) -> None:
        result = runner.invoke(app, ["cache", "--help"])
        assert "clear" in result.stdout


class TestCacheKeySampleIndex:
    """``samples_per_example`` (Task 7.1): the sample index follows ``round_index``'s
    inclusion rule, so a single-sample run keeps every key it already had, while a
    repeated-sampling run forks one key per sample instead of serving every sample
    from the first cached response."""

    def _key(self, sample_index: int | None) -> str:
        return cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            sample_index=sample_index,
        )

    def test_none_keeps_the_existing_key(self) -> None:
        legacy = cache_key(
            model_id="m", prompt_text="hi", inputs={}, temperature=0.0, max_tokens=1024
        )
        assert self._key(None) == legacy

    def test_zero_differs_from_none(self) -> None:
        assert self._key(0) != self._key(None)

    def test_different_samples_produce_different_keys(self) -> None:
        assert self._key(0) != self._key(1)


# ---------------------------------------------------------------------------
# Tool-call responses — the value carries a parsed ToolTrace
# ---------------------------------------------------------------------------


def _rich_trace() -> ToolTrace:
    """A trace exercising every field a downstream consumer reads."""
    return ToolTrace(
        calls=[
            ToolCall(
                tool_name="search_orders",
                arguments={"customer": "Zoë", "limit": 5, "filters": {"open": True}},
                call_id="toolu_01ABC",
                sequence_index=0,
                round_index=0,
            ),
            ToolCall(
                tool_name="issue_refund",
                arguments={"order_id": "A-1", "amount": 12.5, "items": [1, 2]},
                call_id="call_xyz",
                parent_call_id="toolu_01ABC",
                sequence_index=1,
                round_index=1,
            ),
            ToolCall(
                tool_name="broken",
                arguments={"_parse_error": True},
                call_id=None,
                sequence_index=2,
                round_index=1,
            ),
        ],
        final_text="Refunded order A-1.",
        raised_refusal=True,
        refusal_text="I can only refund once.",
        round_count=2,
    )


def _tool_put_kwargs(trace: ToolTrace) -> dict[str, object]:
    kw = _put_kwargs()
    kw["response_text"] = trace.final_text or ""
    return kw


class TestCacheStoreToolTrace:
    async def test_round_trip_preserves_the_trace_exactly(self, store: CacheStore) -> None:
        trace = _rich_trace()
        await store.put("k", **_tool_put_kwargs(trace), finish_reason="tool_calls", trace=trace)  # type: ignore[arg-type]
        got = await store.get("k")
        assert got is not None
        assert got.trace == trace
        assert got.trace is not None
        assert got.trace.model_dump() == trace.model_dump()
        assert got.response_text == "Refunded order A-1."
        assert got.finish_reason == "tool_calls"
        assert (got.input_tokens, got.output_tokens, got.latency_ms) == (10, 5, 250)
        assert got.cost_usd == pytest.approx(0.0001)

    async def test_tool_only_trace_round_trips(self, store: CacheStore) -> None:
        trace = ToolTrace(
            calls=[ToolCall(tool_name="t", arguments={}, call_id="c1", sequence_index=0)],
            final_text=None,
        )
        await store.put("k", **_tool_put_kwargs(trace), trace=trace)  # type: ignore[arg-type]
        got = await store.get("k")
        assert got is not None
        assert got.trace == trace
        assert got.trace is not None
        assert got.trace.final_text is None

    async def test_text_rows_carry_no_trace(self, store: CacheStore) -> None:
        await store.put("k", **_put_kwargs())  # type: ignore[arg-type]
        got = await store.get("k")
        assert got is not None
        assert got.trace is None

    async def test_a_trace_that_no_longer_validates_is_a_miss(self, tmp_path: Path) -> None:
        # A row written by some other version whose trace shape this code
        # cannot read must be re-dispatched, not crash the run.
        url = f"sqlite+aiosqlite:///{tmp_path / 'c.db'}"
        store = await CacheStore.open(database_url=url)
        try:
            trace = _rich_trace()
            await store.put("k", **_tool_put_kwargs(trace), trace=trace)  # type: ignore[arg-type]
            async with store._engine.begin() as conn:
                await conn.execute(
                    text("UPDATE cached_calls SET trace_json = :j"),
                    {"j": '{"calls": "not a list"}'},
                )
            assert await store.get("k") is None
        finally:
            await store.close()


# DDL exactly as `CacheStore.open` created it on origin/main (af4e6fe), dumped
# from `sqlite_master` of a DB that code wrote — the shape every existing
# user's ~/.evalshift/cache.db has today.
_ORIGIN_MAIN_DDL = (
    "CREATE TABLE cached_calls (\n"
    "\tcache_key VARCHAR(64) NOT NULL, \n"
    "\tmodel_id VARCHAR(128) NOT NULL, \n"
    "\tprompt_text TEXT NOT NULL, \n"
    "\tinputs_json TEXT NOT NULL, \n"
    "\tresponse_text TEXT NOT NULL, \n"
    "\tinput_tokens INTEGER NOT NULL, \n"
    "\toutput_tokens INTEGER NOT NULL, \n"
    "\tcost_usd FLOAT NOT NULL, \n"
    "\tlatency_ms INTEGER NOT NULL, \n"
    "\tfinish_reason VARCHAR(32), \n"
    "\tcreated_at DATETIME NOT NULL, \n"
    "\tPRIMARY KEY (cache_key)\n"
    ")"
)


class TestCacheMigrationFromOriginMain:
    async def _legacy_db(self, tmp_path: Path) -> str:
        from datetime import UTC, datetime

        from sqlalchemy.ext.asyncio import create_async_engine

        url = f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}"
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text(_ORIGIN_MAIN_DDL))
            await conn.execute(
                text(
                    "INSERT INTO cached_calls (cache_key, model_id, prompt_text, inputs_json, "
                    "response_text, input_tokens, output_tokens, cost_usd, latency_ms, "
                    "finish_reason, created_at) VALUES ('old', 'm', 'p', '{}', 'old text', "
                    "1, 2, 0.5, 3, 'stop', :now)"
                ),
                {"now": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")},
            )
        await engine.dispose()
        return url

    async def test_existing_text_rows_still_hit(self, tmp_path: Path) -> None:
        store = await CacheStore.open(database_url=await self._legacy_db(tmp_path))
        try:
            got = await store.get("old")
            assert got is not None
            assert got.response_text == "old text"
            assert got.finish_reason == "stop"
            assert got.trace is None
        finally:
            await store.close()

    async def test_tool_rows_can_be_written_after_the_backfill(self, tmp_path: Path) -> None:
        url = await self._legacy_db(tmp_path)
        store = await CacheStore.open(database_url=url)
        try:
            trace = _rich_trace()
            await store.put("new", **_tool_put_kwargs(trace), trace=trace)  # type: ignore[arg-type]
            got = await store.get("new")
            assert got is not None
            assert got.trace == trace
        finally:
            await store.close()
        # Re-opening an already-migrated DB is a no-op, not a duplicate-column error.
        again = await CacheStore.open(database_url=url)
        try:
            assert await again.count() == 2
        finally:
            await again.close()


class TestToolRoundKeySensitivity:
    """Every component the tool path keys on moves the key; identical inputs hit.

    Mirrors what :func:`evalshift_cli.runner.orchestrator._execute_with_tools`
    passes per round: the dispatched message list as ``history``, the tools
    array exactly as sent, the generation config (tool_choice / parallel_tool_calls), the
    round index and the sample index.
    """

    _TOOLS: ClassVar[list[dict[str, Any]]] = [
        {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    ]

    def _kwargs(self) -> dict[str, Any]:
        return {
            "model_id": "anthropic/claude-sonnet-4-5",
            "prompt_text": "Hello Alex",
            "inputs": {"name": "Alex"},
            "temperature": 0.0,
            "max_tokens": 1024,
            "history": [
                {"role": "user", "content": "Hello Alex"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_r0_0",
                            "type": "function",
                            "function": {"name": "t", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_r0_0", "content": "{}"},
            ],
            "generation_config": {"tool_choice": "auto"},
            "tools_payload": [dict(t) for t in self._TOOLS],
            "round_index": 1,
            "sample_index": None,
        }

    def test_identical_inputs_same_key(self) -> None:
        assert cache_key(**self._kwargs()) == cache_key(**self._kwargs())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_id", "openai/gpt-4o"),
            ("prompt_text", "Hello Bea"),
            ("inputs", {"name": "Bea"}),
            ("temperature", 0.7),
            ("max_tokens", 2048),
            ("history", [{"role": "user", "content": "Hello Alex"}]),
            ("generation_config", {"tool_choice": "required"}),
            ("generation_config", {"tool_choice": "auto", "parallel_tool_calls": False}),
            ("round_index", 2),
            ("sample_index", 1),
        ],
    )
    def test_each_component_changes_the_key(self, field: str, value: Any) -> None:
        changed = self._kwargs()
        changed[field] = value
        assert cache_key(**changed) != cache_key(**self._kwargs())

    def test_a_changed_fixture_result_changes_the_key(self) -> None:
        changed = self._kwargs()
        changed["history"] = [dict(m) for m in changed["history"]]
        changed["history"][2]["content"] = '{"hits": 1}'
        assert cache_key(**changed) != cache_key(**self._kwargs())

    def test_tool_strictness_changes_the_key(self) -> None:
        changed = self._kwargs()
        changed["tools_payload"] = [{**self._TOOLS[0], "strict": True}]
        assert cache_key(**changed) != cache_key(**self._kwargs())

    def test_a_changed_tool_schema_changes_the_key(self) -> None:
        changed = self._kwargs()
        changed["tools_payload"] = [
            {**self._TOOLS[0], "input_schema": {"type": "object", "required": ["q"]}}
        ]
        assert cache_key(**changed) != cache_key(**self._kwargs())

    def test_reordering_the_tools_changes_the_key(self) -> None:
        second = {"name": "u", "description": "e", "input_schema": {"type": "object"}}
        a = self._kwargs()
        a["tools_payload"] = [self._TOOLS[0], second]
        b = self._kwargs()
        b["tools_payload"] = [second, self._TOOLS[0]]
        assert cache_key(**a) != cache_key(**b)


def test_the_suite_never_opens_the_users_real_cache() -> None:
    # Several command-level tests open the default on-disk cache. Without the
    # autouse redirect in tests/conftest.py they read and wrote the developer's
    # ~/.evalshift/cache.db, so a cached row from an earlier test run could
    # serve a later one.
    from evalshift_cli.cache import schema

    assert Path.home() / ".evalshift" / "cache.db" != schema.DEFAULT_CACHE_PATH
