"""
Configuration and constants for the MCP examples.
"""
from __future__ import annotations

import os


def _get_int(name: str, default: int) -> int:
    """Read an integer env var while preserving a safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _get_float(name: str, default: float) -> float:
    """Read a float env var while preserving a safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


# MCP Server settings
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = _get_int("MCP_SERVER_PORT", 8000)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}")

# API settings
REQUEST_TIMEOUT = _get_int("REQUEST_TIMEOUT", 10)
MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "1.0")

# Tool calling settings
TOOL_CALL_MAX_RETRIES = _get_int("TOOL_CALL_MAX_RETRIES", 3)
TOOL_CALL_TIMEOUT = _get_int("TOOL_CALL_TIMEOUT", 30)

# Search settings
SEARCH_DOCS_DATABASE = [
    "MCP is a protocol for tool discovery and interaction.",
    "Tool calling usually relies on pre-defined function schemas.",
    "Plugins are often static integrations bundled into an app platform.",
    "Model Context Protocol enables standardized tool communication.",
    "MCP servers expose tools through REST or other transport protocols.",
]

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.0)
