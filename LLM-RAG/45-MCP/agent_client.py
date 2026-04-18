# agent_client.py
"""
MCP Agent Client - Simple routing-based agent for tool discovery and execution.
This module provides a basic agent that can discover tools from an MCP server
and route queries to appropriate tools.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException, Timeout

from config import MCP_SERVER_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_http_session() -> requests.Session:
    """Reuse a single HTTP session for lower connection overhead."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


# -------------------------
# Tool Discovery & Execution
# -------------------------
def discover_tools() -> List[Dict[str, Any]]:
    """
    Discover available tools from the MCP server.
    
    Returns:
        List of tool definitions with name, description, and schema
        
    Raises:
        RequestException: If server communication fails
    """
    try:
        logger.info("Discovering tools from MCP server")
        resp = get_http_session().get(
            f"{MCP_SERVER_URL}/mcp/tools",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tools = data.get("tools", [])
        logger.info(f"Discovered {len(tools)} tools")
        return tools
    except Timeout:
        logger.error("Tool discovery timeout")
        raise
    except RequestException as e:
        logger.error(f"Tool discovery failed: {e}")
        raise


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
    logger.info(f"Calling tool: {name} with args: {arguments}")
    try:
        resp = get_http_session().post(
            f"{MCP_SERVER_URL}/mcp/call",
            json={"name": name, "arguments": arguments},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Timeout:
        logger.error(f"Tool call timeout: {name}")
        raise
    except RequestException as e:
        logger.error(f"Tool call failed: {name} - {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid tool response: {e}")
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
    q = user_query.lower().strip()
    logger.debug(f"Routing query: {q}")
    
    # Rule-based routing
    if any(word in q for word in ["weather", "temperature", "climate", "hot", "cold"]):
        logger.info("Routing to get_weather tool")
        return {
            "name": "get_weather",
            "arguments": {
                "city": "New York",
                "unit": "celsius",
            },
        }
    
    if any(word in q for word in ["search", "mcp", "document", "find", "look"]):
        logger.info("Routing to search_docs tool")
        return {
            "name": "search_docs",
            "arguments": {"query": "MCP"},
        }
    
    if any(word in q for word in ["add", "sum", "total", "calculate", "math"]):
        logger.info("Routing to add_numbers tool")
        return {
            "name": "add_numbers",
            "arguments": {"a": 3, "b": 5},
        }
    
    logger.warning(f"No suitable tool found for query: {q}")
    return None


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
