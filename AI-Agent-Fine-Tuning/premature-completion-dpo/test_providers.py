"""Offline tests: hosted Chinese-model API backends are removed; OpenAI is the default."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import llm_client

ROOT = Path(__file__).resolve().parent
BANNED = re.compile(r"volces|volcengine|ark_api|doubao|moonshot|dashscope|siliconflow", re.I)


def test_provider_table_has_no_ark():
    assert set(llm_client.PROVIDER_DEFAULTS) == {"openai", "openrouter"}
    for env_name, base_url, model in llm_client.PROVIDER_DEFAULTS.values():
        assert not BANNED.search(f"{env_name} {base_url} {model}")


def test_openai_is_default_backend():
    env_name, base_url, model = llm_client.PROVIDER_DEFAULTS["openai"]
    assert env_name == "OPENAI_API_KEY"
    assert base_url is None
    assert llm_client.default_model("openai") == "gpt-4o-mini"


@pytest.mark.parametrize("provider", ["ark", "doubao", "moonshot", "deepseek"])
def test_removed_providers_rejected(provider, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "dummy")
    with pytest.raises(ValueError):
        llm_client.make_client(provider)
    with pytest.raises(ValueError):
        llm_client.default_model(provider)


def test_make_client_openai_offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _client, backend = llm_client.make_client("openai")
    assert backend["endpoint"] == "https://api.openai.com/v1/chat/completions"
    assert backend["credential_env"] == "OPENAI_API_KEY"


@pytest.mark.parametrize("script", ["build_preference_data.py", "evaluate.py"])
def test_cli_choices_and_default(script):
    src = (ROOT / script).read_text(encoding="utf-8")
    assert 'parser.add_argument("--provider", default="openai", choices=["openai", "openrouter"])' in src
    assert not BANNED.search(src)


@pytest.mark.parametrize("name", ["README.md", "env.example", "demo.py", "llm_client.py"])
def test_docs_and_config_free_of_removed_providers(name):
    assert not BANNED.search((ROOT / name).read_text(encoding="utf-8"))
