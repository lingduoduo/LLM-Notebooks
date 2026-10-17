"""Moonshot/Kimi and ByteDance Ark/Doubao backends are gone; OpenAI is the default."""
import pytest

import agent

REMOVED = ("moonshot", "ark")


def _clear_env(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                 "MOONSHOT_API_KEY", "ARK_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_only_openai_provider_is_registered():
    assert set(agent._PROVIDERS) == {"openai"}
    assert agent._PROVIDERS["openai"] == ("OPENAI_API_KEY", None, "gpt-5.6-luna")


def test_default_build_client_uses_official_openai(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client, model = agent.build_client()
    assert model == "gpt-5.6-luna"
    assert str(client.base_url).startswith("https://api.openai.com/")


@pytest.mark.parametrize("provider", REMOVED)
def test_removed_provider_is_rejected_not_silently_rerouted(monkeypatch, provider):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MOONSHOT_API_KEY", "x")
    monkeypatch.setenv("ARK_API_KEY", "x")
    with pytest.raises(RuntimeError, match=f"unsupported LLM_PROVIDER='{provider}'"):
        agent.build_client()


@pytest.mark.parametrize("model", ["kimi-k3", "doubao-seed-1-6-250615", "qwen-max", "deepseek-chat"])
def test_openrouter_fallback_never_maps_to_a_chinese_model(model):
    assert agent._to_openrouter_model(model) == "openai/gpt-5.6-luna"
