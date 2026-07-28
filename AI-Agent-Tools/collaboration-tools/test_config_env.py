"""Regression test: malformed numeric env vars must not crash config load.

BROWSER_TIMEOUT / SMTP_PORT / HITL_TIMEOUT_SECONDS (config.py) and
OPENAI_TIMEOUT / OPENAI_MAX_RETRIES (subagent_tools.py) were parsed with bare
int()/float(); malformed values crashed with ValueError at import/startup.
They now fall back to defaults with a warning.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Ensure the real local modules are imported.
for _mod in ("config", "subagent_tools", "llm_fallback"):
    sys.modules.pop(_mod, None)

import config as cfg
import subagent_tools as sa


def test_env_int_falls_back_on_malformed(monkeypatch, capsys):
    monkeypatch.setenv("SMTP_PORT", "smtp")
    assert cfg._env_int("SMTP_PORT", 587) == 587
    assert "invalid SMTP_PORT" in capsys.readouterr().err


def test_load_config_survives_malformed_env(monkeypatch):
    monkeypatch.setenv("BROWSER_TIMEOUT", "soon")
    monkeypatch.setenv("SMTP_PORT", "smtp")
    monkeypatch.setenv("HITL_TIMEOUT_SECONDS", "never")
    c = cfg.load_config()
    assert c.browser.timeout == 30000
    assert c.email.smtp_port == 587
    assert c.hitl.timeout_seconds == 3600


def test_load_config_parses_valid_env(monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "2525")
    assert cfg.load_config().email.smtp_port == 2525


def test_subagent_env_or_default_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT", "abc")
    assert sa._env_or_default("OPENAI_TIMEOUT", 60.0, float) == 60.0


def test_subagent_env_or_default_parses_valid(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")
    assert sa._env_or_default("OPENAI_MAX_RETRIES", 2, int) == 5


class TestPlaceholderCredentials:
    """Copying env.example to .env leaves placeholders that look configured.

    A placeholder is worse than an empty value: the tools skip their "not
    configured" branch and fire real requests at fake endpoints (404s from the
    sample Slack/Discord webhooks, SendGrid auth failures).
    """

    def test_unfilled_placeholders_are_treated_as_unset(self):
        import config as c

        for value in (
            "your-sendgrid-api-key",
            "your-telegram-bot-token",
            "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            "https://discord.com/api/webhooks/YOUR/WEBHOOK/URL",
            "your-email@gmail.com",
            "admin@example.com",
            "",
            "   ",
        ):
            assert c._is_placeholder(value) is True, value

    def test_real_credentials_are_kept(self):
        import config as c

        for value in (
            "SG.aBc123RealLookingKey",
            "https://hooks.slack.com/services/T0/B0/abcdef123456",
            "1234567890:AAExampleRealBotToken",
            "ops@mycompany.io",
            "http://localhost:8080/hitl",
        ):
            assert c._is_placeholder(value) is False, value

    def test_env_secret_returns_none_for_placeholder(self, monkeypatch):
        import config as c

        monkeypatch.setenv("SOME_TOKEN", "your-telegram-bot-token")
        assert c._env_secret("SOME_TOKEN") is None

        monkeypatch.setenv("SOME_TOKEN", "1234:realtoken")
        assert c._env_secret("SOME_TOKEN") == "1234:realtoken"

        monkeypatch.delenv("SOME_TOKEN")
        assert c._env_secret("SOME_TOKEN") is None
