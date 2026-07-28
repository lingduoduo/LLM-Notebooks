"""Regression test: malformed numeric env vars must not crash config import.

TEMPERATURE / MAX_TOKENS / MAX_OUTPUT_LENGTH were parsed with bare
float()/int() at import time, so e.g. MAX_TOKENS=abc crashed every tool with
ValueError. They now fall back to defaults with a warning.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

import config as cfg


def _load_config_copy():
    """Import config.py under a private module name.

    Reloading ``config`` in place would swap ``sys.modules['config']`` for a new
    module object while every already-imported tool still holds a reference to
    the old ``Config`` class, so later tests would patch one class and exercise
    another. Loading under a throwaway name keeps import-time behavior testable
    without disturbing the shared module.
    """
    path = Path(__file__).with_name("config.py")
    spec = importlib.util.spec_from_file_location("_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_int_falls_back_on_malformed(monkeypatch, capsys):
    monkeypatch.setenv("MAX_TOKENS", "abc")
    assert cfg._env_int("MAX_TOKENS", 4096) == 4096
    assert "invalid MAX_TOKENS" in capsys.readouterr().err


def test_env_int_parses_valid_value(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "123")
    assert cfg._env_int("MAX_TOKENS", 4096) == 123


def test_env_float_falls_back_on_malformed(monkeypatch, capsys):
    monkeypatch.setenv("TEMPERATURE", "hot")
    assert cfg._env_float("TEMPERATURE", 0.7) == 0.7
    assert "invalid TEMPERATURE" in capsys.readouterr().err


def test_env_float_parses_valid_value(monkeypatch):
    monkeypatch.setenv("TEMPERATURE", "0.2")
    assert cfg._env_float("TEMPERATURE", 0.7) == 0.2


def test_module_import_survives_malformed_env(monkeypatch):
    """Import-time class attributes must not raise on malformed env values."""
    monkeypatch.setenv("MAX_OUTPUT_LENGTH", "lots")
    fresh = _load_config_copy()
    assert fresh.Config.MAX_OUTPUT_LENGTH == 1000


def test_loading_a_config_copy_leaves_the_shared_module_intact():
    """Import-time tests must not orphan the Config every tool already bound."""
    before = sys.modules["config"]
    _load_config_copy()
    assert sys.modules["config"] is before


class TestOpenAIConfiguration:
    """The provider router was replaced by one direct OpenAI configuration."""

    def test_uses_direct_openai_and_the_default_model(self, monkeypatch):
        monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(cfg.Config, "MODEL", "gpt-5.6")

        assert cfg.Config.get_llm_config() == {
            "provider": "openai",
            "api_key": "test-openai-key",
            "model": "gpt-5.6",
        }

    def test_model_can_be_overridden(self, monkeypatch):
        monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(cfg.Config, "MODEL", "gpt-5.6-terra")

        assert cfg.Config.get_llm_config()["model"] == "gpt-5.6-terra"

    def test_default_model_is_gpt_5_6(self, monkeypatch):
        monkeypatch.delenv("MODEL", raising=False)
        fresh = _load_config_copy()

        assert fresh.Config.MODEL == "gpt-5.6"

    def test_api_key_is_required(self, monkeypatch):
        monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", None)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            cfg.Config.get_llm_config()

    def test_validate_rejects_a_missing_key(self, monkeypatch):
        monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", None)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            cfg.Config.validate()

    @pytest.mark.parametrize(
        "removed",
        ["PROVIDER", "SILICONFLOW_API_KEY", "DOUBAO_API_KEY", "KIMI_API_KEY",
         "MOONSHOT_API_KEY", "OPENROUTER_API_KEY"],
    )
    def test_third_party_provider_settings_are_gone(self, removed):
        assert not hasattr(cfg.Config, removed)

    @pytest.mark.parametrize("removed", ["get_api_key", "effective_provider"])
    def test_provider_routing_helpers_are_gone(self, removed):
        assert not hasattr(cfg.Config, removed)
