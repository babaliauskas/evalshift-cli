"""Tests for :mod:`evalshift.models.client`.

We test the client by monkeypatching ``litellm.acompletion`` and
``litellm.completion_cost``. There are no live API calls in this suite —
that lives in ``scripts/smoke_live.py`` which the user runs by hand.

What we care about here:

* Successful calls produce a :class:`CompletionResult` with the right
  text, tokens, and cost.
* Provider exceptions are mapped into :class:`RateLimitError` /
  :class:`AuthError` / :class:`ModelError`.
* Transient errors are retried (and the retry sleeps respect the policy
  cap), permanent errors (auth) are not.
* The canonical model id is what gets dispatched even when an alias is
  passed in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from evalshift.evaluators.tool_models import ToolSpec
from evalshift.models import client as client_module
from evalshift.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    ModelError,
    RateLimitError,
    RetryPolicy,
    ToolCompletionResult,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        text: str,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        finish_reason: str | None = None,
    ) -> None:
        self.choices = [_FakeChoice(text, finish_reason)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _patch_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[..., Any],
) -> dict[str, Any]:
    """Replace ``litellm.acompletion`` with ``handler`` and capture call kwargs."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        result = handler(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        client_module.litellm,
        "completion_cost",
        lambda completion_response=None, **_: 0.00042,
    )
    # Make backoff sleeps a no-op so retry tests run instantly.
    monkeypatch.setattr(client_module.asyncio, "sleep", _noop_sleep)
    return captured


async def _noop_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Tool-aware fakes
# ---------------------------------------------------------------------------


def _patch_tools_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    response_dict: dict[str, Any] | Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Replace ``litellm.acompletion`` so it returns a dict-shaped response.

    The tool-aware code path coerces SDK objects to dicts via
    ``response.model_dump()``; for the test we just hand back a dict
    directly. Returns the captured kwargs so tests can assert what was
    sent to LiteLLM.
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        if callable(response_dict):
            return response_dict(**kwargs)
        return response_dict

    monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        client_module.litellm,
        "completion_cost",
        lambda completion_response=None, **_: 0.0,
    )
    monkeypatch.setattr(client_module.asyncio, "sleep", _noop_sleep)
    return captured


_OPENAI_SINGLE_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "search_db",
                            "arguments": '{"query": "ACME"}',
                        },
                    },
                ],
            },
        },
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_delay_zero_attempts_under_cap(self) -> None:
        policy = RetryPolicy(max_attempts=3, base_seconds=2.0, cap_seconds=10.0)
        # First attempt's upper bound should be `base * 2^0 = 2`.
        for _ in range(50):
            d = policy.delay(1)
            assert 0 <= d <= 2.0

    def test_delay_caps_at_cap_seconds(self) -> None:
        policy = RetryPolicy(max_attempts=10, base_seconds=2.0, cap_seconds=4.0)
        # By attempt 5 the exponential blows past cap=4; delay must respect it.
        for _ in range(50):
            d = policy.delay(5)
            assert 0 <= d <= 4.0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCompleteHappyPath:
    async def test_returns_completion_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse("Hello Alex!", prompt_tokens=11, completion_tokens=6),
        )
        client = ModelClient()
        result = await client.complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
        )
        assert isinstance(result, CompletionResult)
        assert result.text == "Hello Alex!"
        assert result.input_tokens == 11
        assert result.output_tokens == 6
        assert result.cost_usd == pytest.approx(0.00042)
        assert result.latency_ms >= 0
        # Canonical id must reach LiteLLM.
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_alias_resolved_to_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        client = ModelClient()
        result = await client.complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.model_id == "gemini/gemini-2.5-flash"
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_default_temperature_and_max_tokens_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
        )
        assert captured["kwargs"]["temperature"] == 0.0
        assert captured["kwargs"]["max_tokens"] == 4096

    async def test_drop_params_enabled_for_reasoning_model_compat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Reasoning-tier models (e.g. gpt-5.6-*) reject temperature != 1.
        # ``drop_params`` tells LiteLLM to silently drop unsupported params
        # instead of erroring, so such models stay usable as judges.
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert captured["kwargs"]["drop_params"] is True

    async def test_finish_reason_stop_is_captured_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok", finish_reason="stop"))
        result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason == "stop"
        assert result.truncated is False

    async def test_finish_reason_length_marks_truncated_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch, lambda **_: _FakeResponse("partial", finish_reason="length")
        )
        with caplog.at_level("WARNING"):
            result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason == "length"
        assert result.truncated is True
        assert any("truncated" in r.message for r in caplog.records)

    async def test_finish_reason_absent_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A response object whose choice lacks finish_reason → None, not a crash.
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason is None
        assert result.truncated is False

    async def test_explicit_max_tokens_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            max_tokens=256,
        )
        assert captured["kwargs"]["max_tokens"] == 256

    async def test_explicit_temperature_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            extra={"response_format": {"type": "json_object"}},
        )
        assert captured["kwargs"]["response_format"] == {"type": "json_object"}

    async def test_unknown_cost_does_not_fail_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))

        def boom(**_kwargs: Any) -> float:
            raise RuntimeError("no pricing for this model")

        monkeypatch.setattr(client_module.litellm, "completion_cost", boom)
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class _ProviderRateLimitError(Exception):
    pass


class _ProviderAuthError(Exception):
    pass


# EvalShift's error mapper looks at the *runtime* class name, so we
# rename these stand-ins to match what real providers raise without
# importing each vendor SDK into the test suite.
_ProviderRateLimitError.__name__ = "RateLimitError"
_ProviderAuthError.__name__ = "AuthenticationError"


class TestErrorMapping:
    async def test_rate_limit_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _ProviderRateLimitError("slow down")
            return _FakeResponse("ok-after-retry")

        _patch_acompletion(monkeypatch, handler)
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == "ok-after-retry"
        assert attempts["n"] == 2

    async def test_rate_limit_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(**_kwargs: Any) -> Any:
            raise _ProviderRateLimitError("slow down")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(RateLimitError):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")

    async def test_auth_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            raise _ProviderAuthError("bad key")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=5))
        with pytest.raises(AuthError):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        # Auth errors are deterministic — burning retries on them wastes time.
        assert attempts["n"] == 1

    async def test_unknown_error_becomes_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(**_kwargs: Any) -> Any:
            raise ValueError("what is happening")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="what is happening"):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")


# ---------------------------------------------------------------------------
# Response extraction edge cases
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    async def test_response_with_dict_message_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Some providers (and the LiteLLM mock harness) return choices
        # whose `.message` is a dict rather than an object.
        class _DictMessageChoice:
            def __init__(self, content: str) -> None:
                self.message = {"content": content}

        class _DictMessageResponse:
            def __init__(self, content: str) -> None:
                self.choices = [_DictMessageChoice(content)]
                self.usage = _FakeUsage(1, 1)

        _patch_acompletion(monkeypatch, lambda **_: _DictMessageResponse("dict msg"))
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == "dict msg"

    async def test_malformed_response_raises_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Empty:
            def __init__(self) -> None:
                self.choices: list[Any] = []

        _patch_acompletion(monkeypatch, lambda **_: _Empty())
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="could not extract text"):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")


# ---------------------------------------------------------------------------
# complete_with_tools
# ---------------------------------------------------------------------------


_DEMO_TOOL = ToolSpec(
    name="search_db",
    description="Search the customer DB",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


class TestCompleteWithTools:
    async def test_returns_tool_completion_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert isinstance(result, ToolCompletionResult)
        assert result.model_id == "openai/gpt-4o"
        assert result.trace.call_count == 1
        assert result.trace.calls[0].tool_name == "search_db"
        assert result.trace.calls[0].arguments == {"query": "ACME"}
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        # Captured kwargs include the tool payload in OpenAI shape.
        kwargs = captured["kwargs"]
        assert "tools" in kwargs
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "search_db"

    async def test_tool_finish_reason_length_marks_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = {
            **_OPENAI_SINGLE_RESPONSE,
            "choices": [dict(_OPENAI_SINGLE_RESPONSE["choices"][0])],
        }
        response["choices"][0]["finish_reason"] = "length"
        _patch_tools_acompletion(monkeypatch, response)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.finish_reason == "length"
        assert result.truncated is True

    async def test_tool_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            max_tokens=321,
        )
        assert captured["kwargs"]["max_tokens"] == 321

    async def test_anthropic_serialises_tool_in_anthropic_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Anthropic-shape response (content list with tool_use block).
        anthropic_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_x",
                                "name": "search_db",
                                "input": {"query": "ACME"},
                            },
                        ],
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        captured = _patch_tools_acompletion(monkeypatch, anthropic_response)
        result = await ModelClient().complete_with_tools(
            model="claude-4.5-sonnet",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 1
        # Tool payload should be the Anthropic shape (no `function` wrapper).
        sent_tool = captured["kwargs"]["tools"][0]
        assert "function" not in sent_tool
        assert sent_tool["name"] == "search_db"
        assert "input_schema" in sent_tool

    async def test_alias_resolves_to_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use a Gemini-shape response (delegates to OpenAI parser).
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="gemini-2.5-flash",  # alias
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.model_id == "gemini/gemini-2.5-flash"
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_explicit_temperature_and_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            extra={"tool_choice": "auto"},
        )
        assert captured["kwargs"]["tool_choice"] == "auto"

    async def test_text_only_response_yields_empty_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Sure thing!"}},
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        _patch_tools_acompletion(monkeypatch, text_response)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 0
        assert result.trace.final_text == "Sure thing!"

    async def test_parse_failure_wrapped_as_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bad shape that won't pass the parser.
        _patch_tools_acompletion(monkeypatch, {"unexpected": "shape"})
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="failed to parse tool response"):
            await client.complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])

    async def test_rate_limit_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:

                class _RLError(Exception):
                    pass

                _RLError.__name__ = "RateLimitError"
                raise _RLError("slow down")
            return _OPENAI_SINGLE_RESPONSE

        _patch_tools_acompletion(monkeypatch, handler)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL]
        )
        assert result.trace.call_count == 1
        assert attempts["n"] == 2

    async def test_auth_error_short_circuits_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1

            class _AuthenticationError(Exception):
                pass

            _AuthenticationError.__name__ = "AuthenticationError"
            raise _AuthenticationError("bad key")

        _patch_tools_acompletion(monkeypatch, handler)
        with pytest.raises(AuthError):
            await ModelClient().complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# complete_messages / complete_messages_with_tools
# ---------------------------------------------------------------------------


_HISTORY_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "Sunny."},
    {"role": "user", "content": "And tomorrow?"},
]


class TestCompleteMessages:
    async def test_passes_messages_through_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("Sunny too."))
        result = await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
        )
        assert isinstance(result, CompletionResult)
        assert result.text == "Sunny too."
        # The exact list object must reach litellm unmodified.
        assert captured["kwargs"]["messages"] is _HISTORY_MESSAGES

    async def test_delegation_equivalence_with_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_a = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="x")
        kwargs_a = captured_a["kwargs"]

        captured_b = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "x"}],
        )
        kwargs_b = captured_b["kwargs"]

        assert kwargs_a == kwargs_b

    async def test_temperature_and_max_tokens_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
            extra={"response_format": {"type": "json_object"}},
        )
        assert captured["kwargs"]["response_format"] == {"type": "json_object"}

    async def test_retries_on_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _ProviderRateLimitError("slow down")
            return _FakeResponse("ok-after-retry")

        _patch_acompletion(monkeypatch, handler)
        result = await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash", messages=_HISTORY_MESSAGES
        )
        assert result.text == "ok-after-retry"
        assert attempts["n"] == 2


class TestCompleteMessagesWithTools:
    async def test_tools_serialisation_matches_complete_with_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_a = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])
        tools_a = captured_a["kwargs"]["tools"]

        captured_b = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
        )
        tools_b = captured_b["kwargs"]["tools"]

        assert tools_a == tools_b

    async def test_passes_messages_through_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=_HISTORY_MESSAGES,
            tools=[_DEMO_TOOL],
        )
        assert isinstance(result, ToolCompletionResult)
        assert captured["kwargs"]["messages"] is _HISTORY_MESSAGES

    async def test_anthropic_serialises_tool_in_anthropic_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anthropic_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_x",
                                "name": "search_db",
                                "input": {"query": "ACME"},
                            },
                        ],
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        captured = _patch_tools_acompletion(monkeypatch, anthropic_response)
        result = await ModelClient().complete_messages_with_tools(
            model="claude-4.5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 1
        sent_tool = captured["kwargs"]["tools"][0]
        assert "function" not in sent_tool
        assert sent_tool["name"] == "search_db"

    async def test_max_tokens_and_extra_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
            max_tokens=321,
            extra={"tool_choice": "auto"},
        )
        assert captured["kwargs"]["max_tokens"] == 321
        assert captured["kwargs"]["tool_choice"] == "auto"

    async def test_auth_error_short_circuits_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1

            class _AuthenticationError(Exception):
                pass

            _AuthenticationError.__name__ = "AuthenticationError"
            raise _AuthenticationError("bad key")

        _patch_tools_acompletion(monkeypatch, handler)
        with pytest.raises(AuthError):
            await ModelClient().complete_messages_with_tools(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                tools=[_DEMO_TOOL],
            )
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# Empty-output warning
# ---------------------------------------------------------------------------


class TestEmptyOutputWarning:
    async def test_warns_on_empty_text_with_output_tokens_and_stop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=42, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == ""
        assert any(
            "returned no text" in r.message and "42 output tokens" in r.message
            for r in caplog.records
        )

    async def test_no_warning_when_zero_output_tokens(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=0, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_no_warning_when_finish_reason_not_stop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=42, finish_reason="length"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_no_warning_when_text_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "hello", prompt_tokens=10, completion_tokens=42, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_warning_names_the_model(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=5, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert any(
            "gemini/gemini-2.5-flash" in r.message and "returned no text" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Temperature value rejection (reasoning-tier models)
# ---------------------------------------------------------------------------

# Name-based to match _map_exception's idiom: production raises
# litellm.BadRequestError; the client only inspects the type NAME.
_BadRequestError = type("BadRequestError", (Exception,), {})

_TEMP_400_MSG = (
    "OpenAIException - Unsupported value: 'temperature' does not support 0.0 "
    "with this model. Only the default (1) value is supported."
)


class TestTemperatureValueRejection:
    async def test_adapts_resends_without_temperature_and_records_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("adapted ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with caplog.at_level("WARNING"):
            result = await client.complete(model="gpt-4o", prompt="hi")

        assert result.text == "adapted ok"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})
        warnings = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(warnings) == 1

    async def test_adaptation_does_not_consume_a_retry_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # max_attempts=2. Sequence: temp-400 (adaptation), transient timeout
        # (attempt 1), success (attempt 2). Only passes if the adaptation
        # left the full retry budget intact.
        calls: list[dict[str, Any]] = []
        timeout_error = type("Timeout", (Exception,), {})

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            if len(calls) == 2:
                raise timeout_error("transient")
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        result = await client.complete(model="gpt-4o", prompt="hi")
        assert result.text == "ok"
        assert len(calls) == 3

    async def test_later_calls_omit_temperature_preemptively(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient()
        await client.complete(model="gpt-4o", prompt="first")
        await client.complete(model="gpt-4o", prompt="second")
        # First call: two dispatches (reject + adapted); second call: one.
        assert len(calls) == 3
        assert "temperature" not in calls[2]

    async def test_non_temperature_400_keeps_existing_retry_then_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError("Unsupported value: 'tool_choice'")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        assert client.temperature_rejected_models == frozenset()

    async def test_temperature_400_without_temperature_in_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A model that keeps 400ing about temperature even after the
        # parameter is gone must surface as ModelError, not loop: after the
        # one adaptation kwargs no longer carry temperature, so the
        # predicate's kwargs leg fails and the normal path takes over.
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError(_TEMP_400_MSG)

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        # The adaptation itself still fired once and recorded the model.
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})
