"""Doubao is no longer a provider: config, agents and CLIs must reject it."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROVIDER_DEFAULT_MODELS, Config


def test_config_has_no_doubao_settings():
    assert not hasattr(Config, "DOUBAO_API_KEY")
    assert not hasattr(Config, "DOUBAO_BASE_URL")
    assert "doubao" not in PROVIDER_DEFAULT_MODELS
    assert Config.get_api_key("doubao") is None


def test_validate_rejects_doubao(capsys):
    assert Config.validate("doubao") is False
    assert "DOUBAO" not in capsys.readouterr().out


def test_openai_remains_available(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test")
    assert Config.get_api_key("openai") == "sk-test"


@pytest.mark.parametrize(
    "module, cls",
    [("agent", "UserMemoryAgent"), ("conversational_agent", "ConversationalAgent")],
)
def test_agents_reject_doubao(module, cls):
    agent_cls = getattr(__import__(module), cls)
    with pytest.raises(ValueError, match="Unsupported provider: doubao"):
        agent_cls("u1", api_key="sk-test", provider="doubao")


def test_memory_cli_rejects_doubao_provider():
    from memory_cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["extract", "--conversation", "hi", "--provider", "doubao"]
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
