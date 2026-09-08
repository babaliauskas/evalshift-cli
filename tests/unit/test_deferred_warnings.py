"""Tests for deferring console warnings during the ``all`` pipeline.

Warnings raised while the pipeline renders — LiteLLM deprecation notices,
insights-retry notes — used to land wherever the emitting call happened to
be: above the live block, glued between stages, mid-verdict. ``evalshift
all`` now runs inside :func:`deferred_console_warnings`, which buffers
WARNING records so the command can print them as one section under the
pipeline block. Errors are never deferred: anything above WARNING still
reaches stderr the moment it happens.
"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from evalshift.models.client import deferred_console_warnings

_LITELLM_LOG = logging.getLogger("LiteLLM")

# The dedupe filter on the LiteLLM logger keeps one process-global ``_seen``
# set, so every test logs a message of its own — a string reused across tests
# would be swallowed as a repeat depending on execution order.


class TestBuffering:
    def test_litellm_warning_is_buffered_not_printed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)

        with deferred_console_warnings() as deferred:
            _LITELLM_LOG.warning("deferred-test: temperature deprecated for gemini-9")

        assert stderr.getvalue() == ""
        messages = [r.getMessage() for r in deferred]
        assert messages == ["deferred-test: temperature deprecated for gemini-9"]

    def test_evalshift_warning_is_buffered_not_printed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Our own loggers (e.g. insights) have no handler — lastResort wrote
        them to stderr mid-block. They must land in the buffer instead."""
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)

        with deferred_console_warnings() as deferred:
            logging.getLogger("evalshift.insights.generator").warning(
                "deferred-test: insights generation attempt 1 rejected",
            )

        assert stderr.getvalue() == ""
        messages = [r.getMessage() for r in deferred]
        assert messages == ["deferred-test: insights generation attempt 1 rejected"]

    def test_buffer_preserves_arrival_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        with deferred_console_warnings() as deferred:
            _LITELLM_LOG.warning("deferred-test: first")
            logging.getLogger("evalshift.runner").warning("deferred-test: second")

        assert [r.getMessage() for r in deferred] == [
            "deferred-test: first",
            "deferred-test: second",
        ]


class TestErrorsAreNeverDeferred:
    def test_litellm_error_reaches_stderr_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)

        with deferred_console_warnings() as deferred:
            _LITELLM_LOG.error("deferred-test: provider exploded")
            assert "deferred-test: provider exploded" in stderr.getvalue()

        assert deferred == []

    def test_evalshift_error_reaches_stderr_immediately(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)

        with deferred_console_warnings() as deferred:
            logging.getLogger("evalshift.models.client").error(
                "deferred-test: retry budget exhausted",
            )
            assert "deferred-test: retry budget exhausted" in stderr.getvalue()

        assert deferred == []


class TestRestoration:
    def test_litellm_console_handler_is_restored_after_the_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        handlers_before = list(_LITELLM_LOG.handlers)

        with deferred_console_warnings():
            pass

        assert list(_LITELLM_LOG.handlers) == handlers_before

        _LITELLM_LOG.warning("deferred-test: emitted after restore")
        assert "deferred-test: emitted after restore" in stderr.getvalue()

    def test_restores_when_the_block_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        litellm_handlers_before = list(_LITELLM_LOG.handlers)

        with pytest.raises(RuntimeError), deferred_console_warnings():
            raise RuntimeError("stage failed")

        assert list(root.handlers) == handlers_before
        assert list(_LITELLM_LOG.handlers) == litellm_handlers_before

        _LITELLM_LOG.warning("deferred-test: emitted after a failed block")
        assert "deferred-test: emitted after a failed block" in stderr.getvalue()


class TestLevelRoutedHandler:
    def test_a_sub_warning_record_does_not_break_deferral(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An INFO record must not cost us the handler.

        litellm >= 1.100 routes records below WARNING to ``sys.stdout`` by
        re-pointing the handler's stream. When only the stderr pair counted as
        a console stream, one INFO record left the handler unrecognised, so
        :func:`deferred_console_warnings` detached nothing and the warning
        printed mid-pipeline as well as landing in the buffer.
        """
        stderr = io.StringIO()
        stdout = io.StringIO()
        monkeypatch.setattr(sys, "stderr", stderr)
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(_LITELLM_LOG, "level", logging.DEBUG)

        _LITELLM_LOG.info("deferred-test: routes the handler at stdout")

        with deferred_console_warnings() as deferred:
            _LITELLM_LOG.warning("deferred-test: must not reach the console")

        assert any("deferred-test: must not reach the console" in r.getMessage() for r in deferred)
        assert "deferred-test: must not reach the console" not in stderr.getvalue()
        assert "deferred-test: must not reach the console" not in stdout.getvalue()
