"""Tests for the OpenAI model configuration and per-model request shaping.

The GPT-5 family and the o-series reject `max_tokens` (they want
`max_completion_tokens`) and several of them reject any non-default
`temperature`, while the older chat models reject `max_completion_tokens`.
agent.py has to pick the right parameters from the model name alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import agent as agent_module
from agent import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    completion_limits,
    is_reasoning_model,
    resolve_api_key,
    supports_chat_tools,
)


def test_reasoning_models_detected():
    for model in ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5", "gpt-5.2-pro",
                  "o1", "o3-mini", "o4-mini"]:
        assert is_reasoning_model(model), model


def test_non_reasoning_models_detected():
    for model in ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", None, ""]:
        assert not is_reasoning_model(model), model


def test_reasoning_models_omit_temperature_and_use_completion_tokens():
    """gpt-5 / o-series reject non-default temperature, so it is left off."""
    kwargs = completion_limits("gpt-5.2", 0.7, 8192)
    assert kwargs == {"max_completion_tokens": 8192}


def test_chat_models_keep_max_tokens_and_requested_temperature():
    kwargs = completion_limits("gpt-4.1", 0.7, 8192)
    assert kwargs == {"temperature": 0.7, "max_tokens": 8192}


def test_default_model_is_openai_and_supports_tools():
    """This agent only works with models that allow tools on chat completions."""
    assert DEFAULT_MODEL.startswith("gpt-")
    assert supports_chat_tools(DEFAULT_MODEL)
    assert DEFAULT_BASE_URL == "https://api.openai.com/v1"


def test_gpt_5_6_family_flagged_as_toolless_on_chat_completions():
    for model in ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol"]:
        assert not supports_chat_tools(model), model
    for model in ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "o4-mini"]:
        assert supports_chat_tools(model), model


def test_resolve_api_key_reads_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    assert resolve_api_key() == "sk-test-key"


def test_resolve_api_key_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key() is None


def test_agent_uses_env_model_and_base_url(monkeypatch):
    """The agent honours OPENAI_MODEL / OPENAI_BASE_URL without an explicit arg."""
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    agent = agent_module.EventTriggeredAgent(api_key="sk-test-key")

    assert agent.model == "gpt-4.1"
    assert agent.provider == "openai"
    assert captured == {"api_key": "sk-test-key", "base_url": "https://example.test/v1"}


def test_agent_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(agent_module, "OpenAI", lambda api_key, base_url: None)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    agent = agent_module.EventTriggeredAgent(api_key="sk-test-key")

    assert agent.model == DEFAULT_MODEL
