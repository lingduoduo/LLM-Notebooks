# agent_client.py
"""
MCP Agent Client - Simple routing-based agent for tool discovery and execution.
This module provides a basic agent that can discover tools from an MCP server
and route queries to appropriate tools.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException, Timeout

from config import MCP_SERVER_URL, REQUEST_TIMEOUT
from config import MCP_API_KEY, MCP_AUTH_HEADER

logger = logging.getLogger(__name__)

MCP_MANIFEST_PATH = "/mcp/manifest"
MCP_TOOLS_PATH = "/mcp/tools"
MCP_CALL_PATH = "/mcp/call"
MCP_EVENTS_PATH = "/mcp/events"
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
# Minimum overlap score required to route a query to a tool.
# A score of 1 (single token overlap) is too permissive and leads to wrong
# tool selection; 2 requires at least two intent signals to align.
_MIN_TOOL_MATCH_SCORE = 2


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    """Reuse a single HTTP session for lower connection overhead."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    if MCP_API_KEY:
        session.headers.update({MCP_AUTH_HEADER: MCP_API_KEY})
    return session


# -------------------------
# Tool Discovery & Execution
# -------------------------
def get_server_manifest() -> Dict[str, Any]:
    """Fetch the server manifest describing capabilities, transports, and tools."""
    try:
        logger.info("Fetching MCP server manifest")
        resp = get_http_session().get(
            f"{MCP_SERVER_URL}{MCP_MANIFEST_PATH}",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        manifest = resp.json()
        logger.info("Fetched MCP manifest for server %s", manifest.get("server_name", "unknown"))
        return manifest
    except Timeout:
        logger.error("Manifest fetch timeout")
        raise
    except RequestException as e:
        logger.error("Manifest fetch failed: %s", e)
        raise


def _parse_sse_tools(response: requests.Response) -> List[Dict[str, Any]]:
    """Extract the tools list from an MCP SSE event stream.

    Reads lines until the 'tools' event data arrives, then closes the connection.
    This implements the SSE-based plug-and-play discovery path described in the MCP spec.
    """
    pending_event = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            pending_event = ""
            continue
        if line.startswith("event:"):
            pending_event = line[6:].strip()
        elif line.startswith("data:") and pending_event == "tools":
            payload = json.loads(line[5:].strip())
            return payload.get("tools", [])
    return []


def _discover_tools_via_sse() -> List[Dict[str, Any]]:
    """Discover tools by subscribing to the MCP SSE event stream.

    Any MCP server that supports /mcp/events can be discovered this way —
    no endpoint-specific wiring required (the USB-C approach).
    """
    logger.info("Attempting SSE-based tool discovery from %s%s", MCP_SERVER_URL, MCP_EVENTS_PATH)
    with get_http_session().get(
        f"{MCP_SERVER_URL}{MCP_EVENTS_PATH}",
        stream=True,
        timeout=REQUEST_TIMEOUT,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        tools = _parse_sse_tools(resp)
    logger.info("Discovered %s tools via SSE stream", len(tools))
    return tools


def discover_tools() -> List[Dict[str, Any]]:
    """Discover available tools using a three-tier fallback chain.

    1. GET /mcp/manifest   — preferred; returns capabilities + tools in one call
    2. GET /mcp/tools      — REST fallback for servers without the manifest endpoint
    3. GET /mcp/events     — SSE fallback; works with any MCP-compliant streaming server

    Raises:
        RequestException: If all three discovery paths fail
    """
    # Tier 1: manifest
    try:
        manifest = get_server_manifest()
        tools = manifest.get("tools", [])
        logger.info("Discovered %s tools via manifest", len(tools))
        return tools
    except Timeout:
        logger.error("Tool discovery timeout")
        raise
    except RequestException:
        pass

    # Tier 2: REST tools endpoint
    logger.warning("Manifest unavailable, trying /mcp/tools")
    try:
        resp = get_http_session().get(
            f"{MCP_SERVER_URL}{MCP_TOOLS_PATH}",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        tools = resp.json().get("tools", [])
        logger.info("Discovered %s tools via /mcp/tools", len(tools))
        return tools
    except RequestException:
        pass

    # Tier 3: SSE event stream
    logger.warning("/mcp/tools unavailable, falling back to SSE stream")
    return _discover_tools_via_sse()


def get_server_manifest_cached(force_refresh: bool = False) -> Dict[str, Any]:
    """Cache manifest discovery for local agent runs."""
    if force_refresh:
        _get_server_manifest_cached.cache_clear()
    return _get_server_manifest_cached()


@lru_cache(maxsize=1)
def _get_server_manifest_cached() -> Dict[str, Any]:
    return get_server_manifest()


def discover_tools_cached(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Cache tool discovery for local agent runs."""
    if force_refresh:
        _discover_tools_cached.cache_clear()
    return _discover_tools_cached()


@lru_cache(maxsize=1)
def _discover_tools_cached() -> List[Dict[str, Any]]:
    return discover_tools()


def call_tool(
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a tool on the MCP server.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        
    Returns:
        Tool execution result
        
    Raises:
        RequestException: If server communication fails
        ValueError: If tool execution fails
    """
    logger.info("Calling tool: %s with args: %s", name, arguments)
    try:
        resp = get_http_session().post(
            f"{MCP_SERVER_URL}{MCP_CALL_PATH}",
            json={"name": name, "arguments": arguments},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok", True):
            error_msg = data.get("error") or "unknown error"
            logger.error("Tool '%s' returned ok=False: %s", name, error_msg)
            raise ValueError(f"Tool '{name}' failed: {error_msg}")
        return data
    except Timeout:
        logger.error("Tool call timeout: %s", name)
        raise
    except RequestException as e:
        logger.error("Tool call failed: %s - %s", name, e)
        raise
    except ValueError as e:
        logger.error("Invalid tool response: %s", e)
        raise


# -------------------------
# Simple Router
# -------------------------
def choose_tool(
    user_query: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Simple rule-based tool selection based on query content.
    In production, this would use an LLM to select tools.
    
    Args:
        user_query: User's input query
        tools: Available tools (optional, for validation)
        
    Returns:
        Tool request dict with name and arguments, or None if no match
    """
    q = user_query.strip()
    query_tokens = _tokenize_text(q)
    logger.debug("Routing query dynamically: %s", q.lower())

    if not tools:
        logger.warning("No tools available for dynamic routing")
        return None

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0

    for tool in tools:
        score = _score_tool_match(query_tokens, tool)
        if score < _MIN_TOOL_MATCH_SCORE:
            continue

        arguments = _build_tool_arguments(tool, user_query)
        if arguments is None:
            continue

        if score > best_score:
            best_score = score
            best_match = {"name": tool.get("name", ""), "arguments": arguments}

    if best_match:
        logger.info(
            "Dynamically routed query to %s with score %s",
            best_match["name"],
            best_score,
        )
        return best_match

    logger.warning("No suitable tool found for query: %s", q.lower())
    return None


def _tokenize_text(text: str) -> set[str]:
    """Normalize free text into comparable lowercase tokens."""
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _tool_properties(tool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return schema properties for a discovered tool."""
    schema = tool.get("input_schema", {})
    properties = schema.get("properties", {})
    return properties if isinstance(properties, dict) else {}


def _tool_required_properties(tool: Dict[str, Any]) -> set[str]:
    """Return required schema properties for a discovered tool."""
    schema = tool.get("input_schema", {})
    required = schema.get("required", [])
    if not isinstance(required, list):
        return set()
    return {str(item) for item in required}


def _tool_intent_hints(tool: Dict[str, Any]) -> set[str]:
    """Infer likely query intents from tool metadata and schema."""
    hints = _tokenize_text(
        f"{tool.get('name', '')} {tool.get('description', '')}"
    )
    properties = _tool_properties(tool)
    property_names = {name.lower() for name in properties}
    hints |= property_names

    if {"city", "unit"} & property_names:
        hints |= {"weather", "temperature", "climate", "forecast", "hot", "cold"}
    if "query" in property_names:
        hints |= {"search", "find", "lookup", "look", "docs", "document", "information", "info"}
    if {"a", "b"} <= property_names:
        hints |= {"add", "sum", "total", "math", "calculate", "plus"}

    for prop in properties.values():
        hints |= _tokenize_text(str(prop.get("description", "")))

    return hints


def _score_tool_match(query_tokens: set[str], tool: Dict[str, Any]) -> int:
    """Score how well a discovered tool matches the current query."""
    hints = _tool_intent_hints(tool)
    if not hints:
        return 0

    overlap = query_tokens & hints
    score = len(overlap)

    property_names = set(_tool_properties(tool))
    if "query" in property_names and {"search", "find", "lookup", "look"} & query_tokens:
        score += 2
    if {"city", "unit"} & property_names and {"weather", "temperature", "climate"} & query_tokens:
        score += 2
    if {"a", "b"} <= property_names and {"add", "sum", "math", "calculate", "plus"} & query_tokens:
        score += 2

    return score


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric literals from a user query."""
    numbers = []
    for match in _NUMBER_PATTERN.findall(text):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def _extract_city(text: str) -> str | None:
    """Best-effort city extraction from phrases like 'weather in Boston'."""
    match = re.search(r"\bin\s+([A-Za-z][A-Za-z\s-]{1,60})", text)
    if not match:
        return None
    city = match.group(1).strip(" ?!.,")
    return city if city else None


def _coerce_default_value(schema: Dict[str, Any]) -> Any:
    """Return a usable default for a schema property when available."""
    if "default" in schema:
        return schema["default"]

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return None


def _build_tool_arguments(tool: Dict[str, Any], user_query: str) -> Optional[Dict[str, Any]]:
    """Build arguments dynamically from the discovered schema."""
    properties = _tool_properties(tool)
    required = _tool_required_properties(tool)
    numbers = _extract_numbers(user_query)
    city = _extract_city(user_query)
    arguments: Dict[str, Any] = {}
    number_index = 0

    for name, schema in properties.items():
        key = name.lower()
        value = _coerce_default_value(schema)

        if key == "query":
            value = user_query.strip()
        elif key in {"city", "location"}:
            value = city or value or "Boston"
        elif key in {"a", "b"}:
            if number_index < len(numbers):
                value = numbers[number_index]
                number_index += 1
            elif value is None and key == "a":
                value = 10
            elif value is None and key == "b":
                value = 20

        if value is not None:
            arguments[name] = value

    if any(name not in arguments for name in required):
        return None

    return arguments


# -------------------------
# Main Agent
# -------------------------
def run_agent(user_query: str) -> None:
    """
    Run the simple routing-based agent.
    
    Args:
        user_query: User's input query
    """
    print("\n" + "=" * 60)
    print("SIMPLE ROUTING AGENT")
    print("=" * 60)
    
    # Step 1: Discover tools
    print("\n[Step 1] Discovering tools from MCP server...")
    try:
        tools = discover_tools_cached()
        print(json.dumps(tools, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: Failed to discover tools: {e}")
        return
    
    # Step 2: Route query to tool
    print("\n[Step 2] Routing query to appropriate tool...")
    try:
        tool_request = choose_tool(user_query, tools)
        if not tool_request:
            print("ERROR: No suitable tool found for the query")
            return
        print(json.dumps(tool_request, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: Failed to choose tool: {e}")
        return
    
    # Step 3: Execute tool
    print("\n[Step 3] Executing tool...")
    try:
        result = call_tool(tool_request["name"], tool_request["arguments"])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: Tool execution failed: {e}")
        return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_agent("Can you search documents about MCP?")
