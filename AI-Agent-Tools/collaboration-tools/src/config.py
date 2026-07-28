"""Configuration management for Collaboration Tools MCP Server."""

import os
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Placeholder markers used throughout env.example. Copying that file to .env is
# the documented setup step, so unfilled entries routinely reach the tools --
# and a placeholder is worse than an empty value: it looks configured, so the
# code skips its "not configured" branch and fires a real request at a fake
# endpoint (404s from the sample Slack/Discord webhooks, SendGrid auth errors).
# Treating them as unset restores the honest "not configured" result.
_PLACEHOLDER_MARKERS = ("your-", "your_", "your/", "@example.com")


def _is_placeholder(value: str) -> bool:
    """Whether a value is still an unfilled env.example placeholder."""
    low = value.strip().lower()
    if not low:
        return True
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _env_secret(name: str) -> Optional[str]:
    """Read a credential env var, treating unfilled placeholders as unset."""
    raw = os.getenv(name)
    if raw is None:
        return None
    if _is_placeholder(raw):
        print(f"Note: {name} is still the env.example placeholder; treating it as unset",
              file=sys.stderr)
        return None
    return raw


def _env_int(name: str, default: int) -> int:
    """Read an integer env var; fall back to default (with a warning) if malformed."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: invalid {name}={raw!r} (must be an integer); using default {default}",
              file=sys.stderr)
        return default


class BrowserConfig(BaseModel):
    """Browser automation configuration."""
    headless: bool = Field(default=False)
    user_data_dir: str = Field(default="~/.config/collaboration-tools/browser")
    timeout: int = Field(default=30000)


class EmailConfig(BaseModel):
    """Email notification configuration."""
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    smtp_use_tls: bool = Field(default=True)
    sendgrid_api_key: Optional[str] = None


class IMConfig(BaseModel):
    """Instant messaging configuration."""
    telegram_bot_token: Optional[str] = None
    telegram_default_chat_id: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None


class HITLConfig(BaseModel):
    """Human-in-the-loop configuration."""
    admin_email: Optional[str] = None
    webhook_url: Optional[str] = None
    timeout_seconds: int = Field(default=3600)


class TimerConfig(BaseModel):
    """Timer management configuration."""
    storage_path: str = Field(default="~/.config/collaboration-tools/timers.json")


class Config(BaseModel):
    """Main configuration object."""
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    im: IMConfig = Field(default_factory=IMConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    timer: TimerConfig = Field(default_factory=TimerConfig)
    log_level: str = Field(default="INFO")


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        browser=BrowserConfig(
            headless=os.getenv("BROWSER_HEADLESS", "false").lower() == "true",
            user_data_dir=os.getenv("BROWSER_USER_DATA_DIR", "~/.config/collaboration-tools/browser"),
            timeout=_env_int("BROWSER_TIMEOUT", 30000)
        ),
        email=EmailConfig(
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=_env_int("SMTP_PORT", 587),
            smtp_username=_env_secret("SMTP_USERNAME"),
            smtp_password=_env_secret("SMTP_PASSWORD"),
            smtp_from_email=_env_secret("SMTP_FROM_EMAIL"),
            smtp_use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
            sendgrid_api_key=_env_secret("SENDGRID_API_KEY")
        ),
        im=IMConfig(
            telegram_bot_token=_env_secret("TELEGRAM_BOT_TOKEN"),
            telegram_default_chat_id=_env_secret("TELEGRAM_DEFAULT_CHAT_ID"),
            slack_webhook_url=_env_secret("SLACK_WEBHOOK_URL"),
            discord_webhook_url=_env_secret("DISCORD_WEBHOOK_URL")
        ),
        hitl=HITLConfig(
            admin_email=_env_secret("HITL_ADMIN_EMAIL"),
            webhook_url=_env_secret("HITL_WEBHOOK_URL"),
            timeout_seconds=_env_int("HITL_TIMEOUT_SECONDS", 3600)
        ),
        timer=TimerConfig(
            storage_path=os.getenv("TIMER_STORAGE_PATH", "~/.config/collaboration-tools/timers.json")
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )


# Global config instance
config = load_config()
