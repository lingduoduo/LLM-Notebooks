"""Regression tests for reasoning-model request shaping.

OpenAI's chat completions endpoint rejects two parameters for the GPT-5 / o-series
reasoning family that every older model accepted:

    Unsupported parameter: 'max_tokens' is not supported with this model.
    Use 'max_completion_tokens' instead.

    Unsupported value: 'temperature' does not support 0.2 with this model.
    Only the default (1) value is supported.

Both are hard 400s, so a request carrying them never reaches the model. Since the
default model here is gpt-5.6-luna, sending `max_tokens` turned every sub-agent
run and every intelligence tool call into an API error instead of a result.

The OpenRouter fallback route normalizes `max_tokens` itself, so the rename must
NOT be applied there.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_fallback import (  # noqa: E402
    is_reasoning_model,
    reasoning_safe_temperature,
    token_limit_kwargs,
    token_limit_parameter,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class TestReasoningModelDetection:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5.6-luna", "gpt-5", "GPT-5.6", "o1-preview", "o3-mini", "o4-mini",
         "openai/gpt-5.6-luna"],
    )
    def test_reasoning_models_detected(self, model):
        assert is_reasoning_model(model) is True

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "openai/gpt-4o", None, ""])
    def test_non_reasoning_models_not_detected(self, model):
        assert is_reasoning_model(model) is False


class TestTokenLimitParameter:
    @pytest.mark.parametrize("model", ["gpt-5.6-luna", "o3-mini"])
    def test_reasoning_models_use_max_completion_tokens(self, model):
        assert token_limit_parameter(model) == "max_completion_tokens"

    @pytest.mark.parametrize("model", ["gpt-4o", None])
    def test_other_models_keep_max_tokens(self, model):
        assert token_limit_parameter(model) == "max_tokens"

    def test_openrouter_route_keeps_max_tokens(self):
        """OpenRouter normalizes max_tokens; the rename is direct-OpenAI only."""
        assert token_limit_parameter("openai/gpt-5.6-luna", OPENROUTER_URL) == "max_tokens"

    def test_kwargs_carry_the_limit_under_the_right_name(self):
        assert token_limit_kwargs("gpt-5.6-luna", 800) == {"max_completion_tokens": 800}
        assert token_limit_kwargs("gpt-4o", 800) == {"max_tokens": 800}
        assert token_limit_kwargs("gpt-5.6-luna", 800, OPENROUTER_URL) == {"max_tokens": 800}


class TestReasoningSafeTemperature:
    def test_reasoning_models_pinned_to_one(self):
        assert reasoning_safe_temperature("gpt-5.6-luna", 0.2) == 1
        assert reasoning_safe_temperature("o3-mini", 0.7) == 1

    def test_other_models_keep_requested_temperature(self):
        assert reasoning_safe_temperature("gpt-4o", 0.2) == 0.2


class _FakeCompletions:
    """Records the kwargs of each create() call and returns a canned reply."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = '{"status": "done", "result": "ok", "missing": ""}'

        class _Choice:
            message = _Msg()

        class _Usage:
            prompt_tokens = 10
            total_tokens = 20

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        return _Resp()


class _FakeClient:
    def __init__(self):
        self.chat = type("chat", (), {"completions": _FakeCompletions()})()


class TestSubagentRequestShape:
    """The sub-agent turn must send a request the default model accepts."""

    def test_run_turn_sends_reasoning_safe_parameters(self, monkeypatch):
        import subagent_tools as sa

        fake = _FakeClient()
        monkeypatch.setattr(sa, "DEFAULT_MODEL", "gpt-5.6-luna")
        monkeypatch.setattr(sa, "DEFAULT_BASE_URL", None)
        monkeypatch.setattr(sa, "_offline", lambda: False)
        monkeypatch.setattr(sa, "_get_client", lambda: fake)

        record = {"messages": [{"role": "user", "content": "hi"}]}
        sa._run_turn(record)

        sent = fake.chat.completions.calls[0]
        assert "max_completion_tokens" in sent
        assert "max_tokens" not in sent
        assert sent["temperature"] == 1

    def test_run_turn_keeps_tuned_values_for_classic_models(self, monkeypatch):
        import subagent_tools as sa

        fake = _FakeClient()
        monkeypatch.setattr(sa, "DEFAULT_MODEL", "gpt-4o")
        monkeypatch.setattr(sa, "DEFAULT_BASE_URL", None)
        monkeypatch.setattr(sa, "_offline", lambda: False)
        monkeypatch.setattr(sa, "_get_client", lambda: fake)

        record = {"messages": [{"role": "user", "content": "hi"}]}
        sa._run_turn(record)

        sent = fake.chat.completions.calls[0]
        assert sent["max_tokens"] == 800
        assert "max_completion_tokens" not in sent
        assert sent["temperature"] == 0.3

    def test_llm_generated_context_sends_reasoning_safe_parameters(self, monkeypatch):
        import subagent_tools as sa

        fake = _FakeClient()
        monkeypatch.setattr(sa, "DEFAULT_MODEL", "gpt-5.6-luna")
        monkeypatch.setattr(sa, "DEFAULT_BASE_URL", None)
        monkeypatch.setattr(sa, "_offline", lambda: False)
        monkeypatch.setattr(sa, "_get_client", lambda: fake)

        sa._prepare_llm_generated_context("summarize", {"a": 1}, None)

        sent = fake.chat.completions.calls[0]
        assert "max_completion_tokens" in sent
        assert "max_tokens" not in sent
        assert sent["temperature"] == 1


class TestIntelligenceToolsRequestShape:
    """Every intelligence tool call must be shaped for the configured model."""

    @pytest.mark.parametrize(
        "coro_name, args",
        [
            ("generate_python_code", ("sort a list",)),
            ("complex_problem_reasoning", ("why is the sky blue",)),
            ("guard_reasoning_process", ("rm -rf /", {"cwd": "/"})),
        ],
    )
    def test_reasoning_safe_parameters(self, monkeypatch, coro_name, args):
        import asyncio

        import intelligence_tools as it

        fake = _FakeClient()
        monkeypatch.setattr(
            it, "_client_and_model", lambda: (fake, "gpt-5.6-luna", None)
        )

        asyncio.run(getattr(it, coro_name)(*args))

        sent = fake.chat.completions.calls[0]
        assert "max_completion_tokens" in sent
        assert "max_tokens" not in sent
        assert sent["temperature"] == 1
