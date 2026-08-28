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
from evalshift.captures.toolset import fingerprint_tools
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

    # -- toolset_fingerprint (Task 8: per-example toolsets reach the cache key) --

    def test_no_toolset_fingerprint_is_byte_identical_to_pre_toolset_payload(self) -> None:
        """Regression: omitting ``toolset_fingerprint`` must not change the hashed payload.

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

        actual = cache_key(
            model_id=model_id,
            prompt_text=prompt_text,
            inputs=inputs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assert actual == expected

        actual_explicit_none = cache_key(
            model_id=model_id,
            prompt_text=prompt_text,
            inputs=inputs,
            temperature=temperature,
            max_tokens=max_tokens,
            toolset_fingerprint=None,
        )
        assert actual_explicit_none == expected

    def test_toolset_fingerprint_changes_key(self) -> None:
        base = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
        )
        with_toolset = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint="sha256:" + "a" * 64,
        )
        assert base != with_toolset

    def test_different_toolsets_produce_different_keys(self) -> None:
        """The cache key differs across differing toolsets (Task 8 requirement)."""
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint="sha256:" + "a" * 64,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint="sha256:" + "b" * 64,
        )
        assert a != b

    def test_same_toolset_fingerprint_same_key(self) -> None:
        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint="sha256:" + "c" * 64,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint="sha256:" + "c" * 64,
        )
        assert a == b

    def test_inline_and_ref_resolved_toolset_fingerprint_the_same_key(self) -> None:
        """An inline toolset and a ``toolset_ref`` to the same tools key identically.

        Simulates the two spellings ``SuiteExample`` allows for one toolset:
        ``fp_inline`` fingerprints a hand-authored ``tools:`` list directly;
        ``fp_from_sidecar`` fingerprints the *same* tools as they would come back
        off a promoted sidecar -- a freshly-built, differently-ordered list of
        equivalent dicts (``fingerprint_tools`` sorts by name, so list order must
        not matter). Both must fingerprint identically, and two ``cache_key()``
        calls built from each must collide.
        """
        inline_tools = [
            {"name": "search_orders", "description": "Look up orders.", "input_schema": {}},
            {"name": "issue_refund", "description": "Refund an order.", "input_schema": {}},
        ]
        sidecar_tools = list(reversed(inline_tools))  # same tools, different order

        fp_inline = fingerprint_tools(inline_tools)
        fp_from_sidecar = fingerprint_tools(sidecar_tools)
        assert fp_inline == fp_from_sidecar

        a = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint=fp_inline,
        )
        b = cache_key(
            model_id="m",
            prompt_text="hi",
            inputs={},
            temperature=0.0,
            max_tokens=1024,
            toolset_fingerprint=fp_from_sidecar,
        )
        assert a == b


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
