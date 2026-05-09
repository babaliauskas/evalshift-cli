"""Tests for :mod:`evalshift.cache`.

Two layers:

* :func:`cache_key` is pure and tested directly — verifying SHA-256
  determinism and stability under dict-ordering changes.
* :class:`CacheStore` is exercised against an in-memory SQLite database
  for fast, hermetic round-trip tests of put/get/expiry/clear.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from typer.testing import CliRunner

from evalshift.cache.store import CacheStore, cache_key
from evalshift.cli.main import app

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
            "evalshift.cache.schema.DEFAULT_CACHE_PATH",
            tmp_path / "cache.db",
        )
        result = runner.invoke(app, ["cache", "clear"])
        assert result.exit_code == 0, result.stdout
        assert "cleared" in result.stdout

    def test_cache_help_lists_clear(self) -> None:
        result = runner.invoke(app, ["cache", "--help"])
        assert "clear" in result.stdout
