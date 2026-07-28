"""Configuration management for the execution tools MCP server."""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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


def _env_float(name: str, default: float) -> float:
    """Read a float env var; fall back to default (with a warning) if malformed."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Warning: invalid {name}={raw!r} (must be a number); using default {default}",
              file=sys.stderr)
        return default


class Config:
    """Configuration for the MCP server."""
    
    # LLM Configuration (direct OpenAI)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Model name; override with MODEL.
    MODEL: str = os.getenv("MODEL", "gpt-5.6")

    # Model parameters
    TEMPERATURE: float = _env_float("TEMPERATURE", 0.7)
    MAX_TOKENS: int = _env_int("MAX_TOKENS", 4096)
    
    # External Services
    GOOGLE_CALENDAR_CREDENTIALS_FILE: str = os.getenv(
        "GOOGLE_CALENDAR_CREDENTIALS_FILE", 
        "credentials.json"
    )
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
    
    # Safety Settings
    REQUIRE_APPROVAL_FOR_DANGEROUS_OPS: bool = (
        os.getenv("REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", "true").lower() == "true"
    )
    AUTO_SUMMARIZE_COMPLEX_OUTPUT: bool = (
        os.getenv("AUTO_SUMMARIZE_COMPLEX_OUTPUT", "true").lower() == "true"
    )
    AUTO_VERIFY_CODE: bool = (
        os.getenv("AUTO_VERIFY_CODE", "true").lower() == "true"
    )
    AUTO_ANALYZE_ERRORS: bool = (
        os.getenv("AUTO_ANALYZE_ERRORS", "true").lower() == "true"
    )
    MAX_OUTPUT_LENGTH: int = _env_int("MAX_OUTPUT_LENGTH", 1000)
    
    # Workspace Configuration
    WORKSPACE_DIR: Path = Path(os.getenv("WORKSPACE_DIR", os.getcwd()))
    
    @classmethod
    def validate(cls) -> None:
        """Validate the configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for LLM operations.")

    @classmethod
    def get_llm_config(cls) -> dict:
        """Return the direct OpenAI LLM configuration."""
        cls.validate()
        return {
            "provider": "openai",
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.MODEL,
        }


# Note: configuration is validated lazily when the LLM is actually used
# (see LLMHelper), so that execution tools which do not require an LLM
# (file write, code run, terminal) can be used offline without an API key.
