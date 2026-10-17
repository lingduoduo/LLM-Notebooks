"""The ByteDance Ark (Doubao) backend is gone and OpenAI is the default."""
import inspect
import sys

import pytest

import agent
import run_experiment_9_9


def test_supported_providers_exclude_ark():
    assert agent.SUPPORTED_PROVIDERS == ("openai", "openrouter")


def test_agent_defaults_to_openai_provider():
    default = inspect.signature(agent.OpenAILongitudinalAgent.__init__).parameters["provider"].default
    assert default == "openai"


def test_agent_rejects_ark_even_with_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "x")
    with pytest.raises(ValueError, match="unsupported provider: ark"):
        agent.OpenAILongitudinalAgent(provider="ark")


def test_agent_uses_official_openai_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = agent.OpenAILongitudinalAgent()
    assert llm.provider == "openai"
    assert llm.model == "gpt-4o-mini"
    assert llm.key_env == "OPENAI_API_KEY"
    assert llm.endpoint == "https://api.openai.com/v1/chat/completions"


def test_agent_without_openai_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        agent.OpenAILongitudinalAgent()


def test_cli_rejects_ark(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_experiment_9_9.py", "--provider", "ark"])
    with pytest.raises(SystemExit) as exc:
        run_experiment_9_9.main()
    assert exc.value.code == 2


def test_cli_defaults_are_openai(monkeypatch):
    captured = {}

    def fake_one_run(provider, model, arm, seed):
        captured["args"] = (provider, model)
        raise RuntimeError("stop before any network call")

    monkeypatch.setattr(run_experiment_9_9, "one_run", fake_one_run)
    monkeypatch.setattr(sys, "argv", ["run_experiment_9_9.py", "--workers", "1"])
    with pytest.raises(RuntimeError, match="stop before any network call"):
        run_experiment_9_9.main()
    assert captured["args"] == ("openai", "gpt-4o-mini")
