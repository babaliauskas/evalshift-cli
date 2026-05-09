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
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.choices = [_FakeChoice(text)]
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
        assert captured["kwargs"]["max_tokens"] == 1024

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
