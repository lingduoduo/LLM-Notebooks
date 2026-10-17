"""Offline checks that only OpenAI-compatible non-Chinese-provider backends remain."""

from pathlib import Path

import pytest

import llm
import run_official_experiment


ROOT = Path(__file__).resolve().parent
REMOVED_ENV = ("ARK_API_KEY", "ARK_MODEL", "MOONSHOT_API_KEY", "MOONSHOT_MODEL")
ALL_ENV = REMOVED_ENV + ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENROUTER_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_removed_provider_keys_do_not_create_backends(clean_env):
    clean_env.setenv("ARK_API_KEY", "dummy")
    clean_env.setenv("MOONSHOT_API_KEY", "dummy")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm._backends()


def test_openai_is_default_backend(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    backends = llm._backends()
    assert [provider for _, _, provider in backends] == ["openai"]
    client, model, _ = backends[0]
    assert model == "gpt-4.1-mini"
    assert "api.openai.com" in str(client.base_url)


def test_no_chinese_provider_endpoints_in_runtime_sources():
    banned = ("volces.com", "moonshot.cn", "ARK_API_KEY", "MOONSHOT_API_KEY", "doubao", "kimi")
    for name in ("llm.py", "run_official_experiment.py", "env.example"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} still referenced in {name}"
    assert "ARK_API_KEY" not in run_official_experiment.SECRET_ENV_NAMES
    assert "MOONSHOT_API_KEY" not in run_official_experiment.SECRET_ENV_NAMES
