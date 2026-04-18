# config.py
"""
Configuration management for Human-in-the-Loop Agent System.
Centralized settings for development, testing, and production environments.
"""
from __future__ import annotations

import os
from typing import Any


class Config(dict):
    """Dictionary config with attribute access for callers that prefer dot syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def env_value(name: str, default: str, legacy_name: str | None = None) -> str:
    """Read HITL-prefixed env vars while accepting older names for notebooks/tests."""
    if name in os.environ:
        return os.environ[name]
    if legacy_name and legacy_name in os.environ:
        return os.environ[legacy_name]
    return default


def env_bool(name: str, default: bool, legacy_name: str | None = None) -> bool:
    """Read a boolean environment variable."""
    value = env_value(name, str(default), legacy_name)
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int, legacy_name: str | None = None) -> int:
    """Read an integer environment variable."""
    raw = env_value(name, str(default), legacy_name)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}")


def env_float(name: str, default: float, legacy_name: str | None = None) -> float:
    """Read a float environment variable."""
    raw = env_value(name, str(default), legacy_name)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw!r}")


# Environment settings
ENVIRONMENT = os.getenv("HITL_ENV", "development")
DEBUG = ENVIRONMENT == "development"

# LangGraph settings
DEFAULT_THREAD_ID = "hitl_demo_thread"
CHECKPOINT_TYPE = os.getenv("HITL_CHECKPOINT_TYPE", "memory")  # memory, postgres, redis

# LLM settings
LLM_MODEL = env_value("HITL_LLM_MODEL", "gpt-4o-mini", "LLM_MODEL")
LLM_TEMPERATURE = env_float("HITL_LLM_TEMPERATURE", 0.2, "LLM_TEMPERATURE")
LLM_MAX_TOKENS = env_int("HITL_LLM_MAX_TOKENS", 1000, "LLM_MAX_TOKENS")

# Tool settings
PURCHASE_CURRENCY = os.getenv("HITL_CURRENCY", "CNY")
PURCHASE_TIMEOUT_SECONDS = env_int("HITL_PURCHASE_TIMEOUT", 300)  # 5 minutes

# Product database (in production, this would be a real database)
PRODUCT_DATABASE = {
    "MacBook Pro": {
        "name": "MacBook Pro M3",
        "price": 15999.0,
        "vendor": "Apple Official Store",
        "price_range": "15,999–25,999",
        "vendors": ["Apple Official Store", "JD.com", "Tmall"],
        "currency": "CNY",
    },
    "MacBook Air": {
        "name": "MacBook Air M3",
        "price": 9999.0,
        "vendor": "Apple Official Store",
        "price_range": "9,999–13,999",
        "vendors": ["Apple Official Store", "JD.com", "Amazon"],
        "currency": "CNY",
    },
    "iPhone": {
        "name": "iPhone 15 Pro",
        "price": 7999.0,
        "vendor": "Apple Official Store",
        "price_range": "7,999–9,999",
        "vendors": ["Apple Official Store", "JD.com", "Tmall"],
        "currency": "CNY",
    },
    "default": {
        "name": "Generic Product",
        "price": 1000.0,
        "vendor": "Multiple vendors",
        "price_range": "1,000–5,000",
        "vendors": ["Multiple vendors"],
        "currency": "CNY",
    },
}

# Approval settings
APPROVAL_TIMEOUT_HOURS = env_int("HITL_APPROVAL_TIMEOUT_HOURS", 24, "APPROVAL_TIMEOUT_HOURS")
AUTO_REJECT_AFTER_TIMEOUT = env_bool("HITL_AUTO_REJECT_TIMEOUT", True)

# Logging settings
LOG_LEVEL = os.getenv("HITL_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("HITL_LOG_FILE", "hitl_agent.log")

# Web interface settings (optional)
ENABLE_WEB_INTERFACE = env_bool("HITL_WEB_INTERFACE", False)
WEB_HOST = os.getenv("HITL_WEB_HOST", "127.0.0.1")
WEB_PORT = env_int("HITL_WEB_PORT", 8000)

# Database settings (for production checkpointing)
DATABASE_URL = os.getenv("HITL_DATABASE_URL", "")
REDIS_URL = os.getenv("HITL_REDIS_URL", "")

# Security settings
ENABLE_AUDIT_LOG = env_bool("HITL_AUDIT_LOG", True)
AUDIT_LOG_FILE = os.getenv("HITL_AUDIT_LOG_FILE", "hitl_audit.log")

# User interface settings
CLI_PROMPT_TIMEOUT = env_int("HITL_CLI_TIMEOUT", 60)  # seconds
SHOW_DETAILED_LOGS = env_bool("HITL_VERBOSE", False)

# Test settings
TEST_MODE = env_bool("HITL_TEST_MODE", False)
MOCK_APPROVAL_RESPONSE = os.getenv("HITL_MOCK_APPROVAL", "")  # for testing


def get_product_info(query: str) -> dict[str, Any]:
    """Get product information from database."""
    query_lower = query.lower().strip()

    # Try exact matches first
    for key, info in PRODUCT_DATABASE.items():
        if key == "default":
            continue
        if key.lower() in query_lower or info["name"].lower() in query_lower:
            return info

    # Return default if no match
    return PRODUCT_DATABASE["default"]


def is_production() -> bool:
    """Check if running in production environment."""
    return ENVIRONMENT == "production"


def get_config() -> Config:
    """Get all configuration as a dictionary."""
    return Config({
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "approval_timeout_hours": APPROVAL_TIMEOUT_HOURS,
        "auto_reject_after_timeout": AUTO_REJECT_AFTER_TIMEOUT,
        "purchase_currency": PURCHASE_CURRENCY,
        "purchase_timeout_seconds": PURCHASE_TIMEOUT_SECONDS,
        "enable_web_interface": ENABLE_WEB_INTERFACE,
        "web_host": WEB_HOST,
        "web_port": WEB_PORT,
        "log_level": LOG_LEVEL,
        "log_file": LOG_FILE,
        "enable_audit_log": ENABLE_AUDIT_LOG,
        "audit_log_file": AUDIT_LOG_FILE,
        "test_mode": TEST_MODE,
        "mock_approval_response": MOCK_APPROVAL_RESPONSE,
        "checkpoint_type": CHECKPOINT_TYPE,
        "database_url": DATABASE_URL,
        "redis_url": REDIS_URL,
    })


def is_development() -> bool:
    """Check if running in development environment."""
    return ENVIRONMENT == "development"


def get_log_level() -> str:
    """Get appropriate log level based on environment."""
    if DEBUG:
        return "DEBUG"
    return LOG_LEVEL
