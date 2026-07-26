import pytest

import config as config_module
from config import Config, LLMConfig
from main import setup_environment


@pytest.fixture(autouse=True)
def skip_workspace_dotenv_except_when_a_test_requests_it(monkeypatch):
    """Keep unit tests independent from a developer's local .env file."""
    monkeypatch.setattr(config_module, "_environment_loaded", True, raising=False)


def test_llm_config_defaults_to_openai_terra(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = Config()
    assert config.llm.model == "gpt-5.6-terra"
    assert not hasattr(config.llm, "provider")


def test_client_config_uses_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client_config, model = LLMConfig().get_client_config()
    assert client_config == {"api_key": "env-key"}
    assert model == "gpt-5.6-terra"


def test_explicit_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client_config, _ = LLMConfig(api_key="explicit-key").get_client_config()
    assert client_config == {"api_key": "explicit-key"}


def test_model_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    assert Config.from_env().llm.model == "gpt-4.1-mini"


def test_openai_generation_settings_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("LLM_MAX_TOKENS", "150")

    llm = Config.from_env().llm

    assert llm.temperature == 0.3
    assert llm.max_tokens == 150


def test_from_env_loads_dotenv_once_at_configuration_boundary(monkeypatch, tmp_path):
    """A documented .env config must reach every Config.from_env caller."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "LLM_MODEL=gpt-4.1-mini\n"
        "LLM_TEMPERATURE=0.3\n"
        "LLM_MAX_TOKENS=150\n",
        encoding="utf-8",
    )
    for variable in ("OPENAI_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(variable, raising=False)

    load_calls = []

    def load_test_dotenv():
        load_calls.append(dotenv_path)
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path)

    monkeypatch.setattr(config_module, "load_dotenv", load_test_dotenv, raising=False)
    monkeypatch.setattr(config_module, "_environment_loaded", False, raising=False)

    first = Config.from_env()
    second = Config.from_env()

    assert first.llm.get_api_key() == "dotenv-key"
    assert first.llm.model == "gpt-4.1-mini"
    assert first.llm.temperature == 0.3
    assert first.llm.max_tokens == 150
    assert second.llm.model == "gpt-4.1-mini"
    assert load_calls == [dotenv_path]

    for variable in ("OPENAI_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS"):
        config_module.os.environ.pop(variable, None)


def test_missing_openai_api_key_has_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        LLMConfig().get_client_config()


def test_legacy_provider_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    config = Config.from_env()
    assert not hasattr(config.llm, "provider")


def test_startup_requires_only_openai_api_key(monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert setup_environment() is False
    assert "OPENAI_API_KEY" in caplog.text
    assert "MOONSHOT_API_KEY" not in caplog.text
    assert "ARK_API_KEY" not in caplog.text
    assert "SILICONFLOW_API_KEY" not in caplog.text
