"""SiliconFlow (Qwen default) is no longer a provider; OpenAI is the default.

Offline: agents are constructed with a fake key, no API calls are made.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    PROVIDER_DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    Config,
    openrouter_model_id,
)

ROOT = Path(__file__).resolve().parent
REMOVED = ("siliconflow", "doubao", "kimi", "moonshot", "deepseek", "dashscope", "qwen")
CHINESE_TERMS = REMOVED + ("volces", "volcengine", "stepfun", "minimax", "zhipu", "bigmodel")


def test_config_has_no_siliconflow_settings():
    assert not hasattr(Config, "SILICONFLOW_API_KEY")
    assert not hasattr(Config, "SILICONFLOW_BASE_URL")
    assert "siliconflow" not in PROVIDER_DEFAULT_MODELS
    assert Config.get_api_key("siliconflow") is None
    assert SUPPORTED_PROVIDERS == ("openai", "openrouter")


def test_validate_rejects_siliconflow(capsys):
    assert Config.validate("siliconflow") is False
    assert "SILICONFLOW" not in capsys.readouterr().out


def test_default_provider_and_model_are_openai():
    assert PROVIDER_DEFAULT_MODELS == {"openai": DEFAULT_OPENAI_MODEL}
    assert DEFAULT_OPENAI_MODEL.startswith("gpt-")


@pytest.mark.parametrize("model", ["Qwen/Qwen3-235B", "deepseek-chat", "kimi-k2", None, ""])
def test_openrouter_fallback_never_maps_to_chinese_model(monkeypatch, model):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    mapped = openrouter_model_id(model)
    if model and "/" in model:
        return  # explicit OpenRouter ids pass through by design
    assert mapped == "openai/" + DEFAULT_OPENAI_MODEL


@pytest.mark.parametrize(
    "module, cls",
    [("agent", "UserMemoryAgent"), ("conversational_agent", "ConversationalAgent")],
)
@pytest.mark.parametrize("provider", REMOVED)
def test_agents_reject_removed_providers(monkeypatch, module, cls, provider):
    # Even with an OpenRouter key present, removed providers must not be
    # silently re-routed through the fallback.
    monkeypatch.setattr(Config, "OPENROUTER_API_KEY", "sk-or-test")
    agent_cls = getattr(__import__(module), cls)
    with pytest.raises(ValueError, match=f"Unsupported provider: {provider}"):
        agent_cls("u1", provider=provider)


@pytest.mark.parametrize(
    "module, cls",
    [("agent", "UserMemoryAgent"), ("conversational_agent", "ConversationalAgent")],
)
def test_agents_default_to_openai(monkeypatch, tmp_path, module, cls):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Config, "PROVIDER", "openai")
    monkeypatch.setattr(Config, "MODEL_NAME", "")
    monkeypatch.setattr(Config, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    agent = getattr(__import__(module), cls)("u1", api_key="sk-test", verbose=False)
    assert agent.provider == "openai"
    assert agent.model == DEFAULT_OPENAI_MODEL
    assert str(agent.client.base_url).startswith("https://api.openai.com/v1")


@pytest.mark.parametrize("argv_prefix", [["extract", "--conversation", "hi"]])
def test_memory_cli_rejects_siliconflow(argv_prefix):
    from memory_cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv_prefix + ["--provider", "siliconflow"])


@pytest.mark.parametrize(
    "name",
    ["config.py", "agent.py", "conversational_agent.py", "background_memory_processor.py",
     "main.py", "memory_cli.py", "env.example"],
)
def test_sources_have_no_chinese_provider_references(name):
    text = (ROOT / name).read_text(encoding="utf-8").lower()
    for term in CHINESE_TERMS:
        assert term not in text, f"{name} still mentions {term}"
