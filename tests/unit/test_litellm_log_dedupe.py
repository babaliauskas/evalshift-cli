"""Tests for the LiteLLM warning dedupe filter.

LiteLLM emits its Gemini 3 ``temperature`` deprecation warning once per call.
A single run produced roughly 290 identical lines. The filter keeps the first
occurrence and drops repeats, so a new deprecation still surfaces exactly once
instead of being silenced wholesale.
"""

from __future__ import annotations

import logging

from evalshift.models.client import _DedupeWarningsFilter


def _record(message: str, *, level: int = logging.WARNING) -> logging.LogRecord:
    return logging.LogRecord(
        name="LiteLLM",
        level=level,
        pathname="vertex_and_google_ai_studio_gemini.py",
        lineno=1095,
        msg=message,
        args=(),
        exc_info=None,
    )


_DEPRECATION = (
    "DeprecationWarning: `temperature`, `top_p`, and `top_k` continue to "
    "function for Gemini 3+ (gemini-3.5-flash-lite) but are planned for removal."
)


class TestDedupe:
    def test_first_warning_passes(self) -> None:
        assert _DedupeWarningsFilter().filter(_record(_DEPRECATION)) is True

    def test_repeat_warning_is_dropped(self) -> None:
        log_filter = _DedupeWarningsFilter()
        log_filter.filter(_record(_DEPRECATION))
        assert log_filter.filter(_record(_DEPRECATION)) is False

    def test_a_different_warning_still_passes(self) -> None:
        """Dedupe, not blanket suppression — a new deprecation must be seen."""
        log_filter = _DedupeWarningsFilter()
        log_filter.filter(_record(_DEPRECATION))
        assert log_filter.filter(_record("something else entirely")) is True

    def test_dedupes_on_the_formatted_message_not_the_template(self) -> None:
        """One template, two models, is two distinct warnings — both matter.

        LiteLLM names the model inside the deprecation string, so comparing
        raw templates would report only the first Gemini 3 model in a run.
        """
        log_filter = _DedupeWarningsFilter()

        first = _record("deprecated for %s")
        first.args = ("gemini-3.1-flash-lite-preview",)
        second = _record("deprecated for %s")
        second.args = ("gemini-3.5-flash-lite",)
        repeat = _record("deprecated for %s")
        repeat.args = ("gemini-3.1-flash-lite-preview",)

        assert log_filter.filter(first) is True
        assert log_filter.filter(second) is True
        assert log_filter.filter(repeat) is False


class TestLevelScope:
    """Only WARNING is deduped. Nothing that signals a failure is swallowed."""

    def test_repeated_errors_all_pass(self) -> None:
        log_filter = _DedupeWarningsFilter()
        first = _record("upstream exploded", level=logging.ERROR)
        second = _record("upstream exploded", level=logging.ERROR)
        assert log_filter.filter(first) is True
        assert log_filter.filter(second) is True

    def test_repeated_info_passes(self) -> None:
        log_filter = _DedupeWarningsFilter()
        log_filter.filter(_record("chatty", level=logging.INFO))
        assert log_filter.filter(_record("chatty", level=logging.INFO)) is True


class TestInstallation:
    def test_filter_is_attached_to_the_litellm_logger(self) -> None:
        """Importing the client is what installs it — no opt-in step."""
        import evalshift.models.client  # noqa: F401  (import for its side effect)

        attached = logging.getLogger("LiteLLM").filters
        assert any(isinstance(f, _DedupeWarningsFilter) for f in attached)

    def test_installing_twice_does_not_stack_filters(self) -> None:
        from evalshift.models.client import _configure_litellm

        _configure_litellm()
        _configure_litellm()
        attached = logging.getLogger("LiteLLM").filters
        assert sum(isinstance(f, _DedupeWarningsFilter) for f in attached) == 1
