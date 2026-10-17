"""DashScope/Qwen (and other Chinese hosted APIs) are no longer providers.

config.py must default to the official OpenAI API and reject removed providers.
Offline: config is reloaded with a no-op load_dotenv, no API calls are made.
"""

import importlib
from pathlib import Path

import dotenv
import pytest

import config

ROOT = Path(__file__).resolve().parents[1]
ENV_VARS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENROUTER_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_MODEL",
    "DASHSCOPE_BASE_URL",
)


@pytest.fixture
def reload_config(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def _reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


def test_default_provider_is_openai(reload_config):
    cfg = reload_config(OPENAI_API_KEY="sk-test")
    assert cfg.LLM_PROVIDER == "openai"
    assert cfg.OPENAI_API_KEY == "sk-test"
    assert cfg.OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert cfg.OPENAI_MODEL == "gpt-5.6-luna"


@pytest.mark.parametrize("provider", ["dashscope", "qwen", "bailian", "kimi", "deepseek"])
def test_removed_providers_are_rejected(reload_config, provider):
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        reload_config(LLM_PROVIDER=provider, DASHSCOPE_API_KEY="sk-test")


def test_dashscope_key_is_ignored(reload_config):
    cfg = reload_config(DASHSCOPE_API_KEY="sk-dash")
    assert cfg.OPENAI_API_KEY is None
    assert "dashscope" not in cfg.OPENAI_BASE_URL


@pytest.mark.parametrize("name", ["config.py", "env.example", "README.md"])
def test_no_dashscope_references(name):
    text = (ROOT / name).read_text(encoding="utf-8").lower()
    for term in ("dashscope", "qwen", "bailian", "moonshot", "kimi", "deepseek"):
        assert term not in text, f"{name} still mentions {term}"
