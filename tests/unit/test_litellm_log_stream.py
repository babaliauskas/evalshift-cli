"""Tests for late-binding LiteLLM's log stream.

LiteLLM attaches a ``StreamHandler`` at import time, capturing the real
``sys.stderr`` object. Anything that redirects ``sys.stderr`` afterwards —
notably the redirect ``rich.live.Live`` installs to keep a live region
intact — is therefore bypassed, and a warning arriving mid-frame scribbles
over the ``evalshift all`` pipeline block, leaving a duplicate of it behind.
Resolving the stream per write hands those lines back to Rich.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.live import Live
from rich.text import Text

from evalshift.models.client import _configure_litellm, _LateBoundStderr


class TestLateBoundStderr:
    def test_write_targets_the_current_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proxy = _LateBoundStderr()
        first = io.StringIO()
        monkeypatch.setattr(sys, "stderr", first)
        proxy.write("early\n")

        second = io.StringIO()
        monkeypatch.setattr(sys, "stderr", second)
        proxy.write("late\n")

        assert first.getvalue() == "early\n"
        assert second.getvalue() == "late\n"

    def test_flush_targets_the_current_stderr(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flushed: list[str] = []

        class _Recording(io.StringIO):
            def flush(self) -> None:
                flushed.append("yes")

        monkeypatch.setattr(sys, "stderr", _Recording())
        _LateBoundStderr().flush()
        assert flushed == ["yes"]


class TestInstallation:
    def test_the_litellm_stream_handler_is_late_bound(self) -> None:
        import evalshift.models.client  # noqa: F401  (import for its side effect)

        handlers = logging.getLogger("LiteLLM").handlers
        streams = [h.stream for h in handlers if isinstance(h, logging.StreamHandler)]
        assert streams, "LiteLLM is expected to install a StreamHandler"
        assert all(isinstance(s, _LateBoundStderr) for s in streams)

    def test_installing_twice_does_not_nest_proxies(self) -> None:
        litellm_log = logging.getLogger("LiteLLM")
        before = list(litellm_log.handlers)
        _configure_litellm()
        _configure_litellm()
        assert list(litellm_log.handlers) == before

    def test_a_file_handler_is_left_alone(
        self,
        tmp_path: Path,
    ) -> None:
        """Only console handlers are proxied — a log file must keep receiving."""
        litellm_log = logging.getLogger("LiteLLM")
        path = tmp_path / "litellm.log"
        file_handler = logging.FileHandler(path)
        litellm_log.addHandler(file_handler)
        try:
            _configure_litellm()
            assert not isinstance(file_handler.stream, _LateBoundStderr)
        finally:
            litellm_log.removeHandler(file_handler)
            file_handler.close()


class TestUnderALiveRegion:
    def test_a_warning_is_printed_through_the_live_console(self) -> None:
        """The end-to-end property: no raw write while a live region is open."""
        raw = io.StringIO()
        console_file = io.StringIO()
        console = Console(file=console_file, force_terminal=True, width=80)

        original_stderr = sys.stderr
        sys.stderr = raw
        try:
            with Live(Text("pipeline"), console=console, refresh_per_second=10):
                logging.getLogger("LiteLLM").warning("deprecation notice")
        finally:
            sys.stderr = original_stderr

        assert "deprecation notice" in console_file.getvalue()
        assert "deprecation notice" not in raw.getvalue()
