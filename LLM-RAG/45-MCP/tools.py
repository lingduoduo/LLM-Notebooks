# tools.py
"""
Tool registry and management for MCP server.
Handles tool registration, discovery, and invocation with validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from config import SEARCH_DOCS_DATABASE

_MAX_CITY_LENGTH = 100
_MAX_QUERY_LENGTH = 500

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a callable tool with metadata and invocation handler."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]
    required_params: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate tool configuration after initialization."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tool name must be a non-empty string")
        if not self.handler or not callable(self.handler):
            raise ValueError(f"Tool {self.name} handler must be callable")


class ToolRegistry:
    """Manages tool registration, discovery, and invocation."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initialized")

    def register(self, tool: Tool) -> None:
        """Register a tool with validation."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._list_tools_cache.cache_clear()
        logger.info(f"Tool registered: {tool.name}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools with metadata."""
        result = self._list_tools_cache()
        logger.debug(f"Listing {len(result)} tools")
        return result

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with validation."""
        if name not in self._tools:
            logger.error(f"Tool not found: {name}")
            raise ValueError(f"Unknown tool: {name}")
        
        tool = self._tools[name]
        
        # Validate required parameters if specified
        if tool.required_params:
            missing = [p for p in tool.required_params if p not in arguments]
            if missing:
                logger.error(f"Missing parameters for tool {name}: {missing}")
                raise ValueError(f"Missing required parameters: {missing}")
        
        logger.info(f"Calling tool: {name} with args: {arguments}")
        try:
            result = tool.handler(arguments)
            logger.debug(f"Tool {name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            raise

    @lru_cache(maxsize=1)
    def _list_tools_cache(self) -> List[Dict[str, Any]]:
        """Cache the serialized tool catalog until the registry changes."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]


registry = ToolRegistry()
LOWERCASE_DOCS_DATABASE = [(doc, doc.lower()) for doc in SEARCH_DOCS_DATABASE]


# -------------------------
# Tool 1: Weather
# -------------------------
def weather_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get current weather for a city."""
    city = args.get("city")
    if not city:
        raise ValueError("city parameter is required")
    city = str(city).strip()
    if len(city) > _MAX_CITY_LENGTH:
        raise ValueError(f"city must be {_MAX_CITY_LENGTH} characters or fewer")

    unit = args.get("unit", "celsius")
    if unit not in ("celsius", "fahrenheit"):
        raise ValueError(f"Invalid unit: {unit}. Must be 'celsius' or 'fahrenheit'")
    
    logger.info(f"Weather query: {city} ({unit})")
    
    # Mock weather data
    temp = 22 if unit == "celsius" else 72
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny",
    }


registry.register(
    Tool(
        name="get_weather",
        description="Get current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
        handler=weather_handler,
        required_params=["city"],
    )
)


# -------------------------
# Tool 2: Search Docs
# -------------------------
def search_docs_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search documentation for matching content."""
    query = args.get("query")
    if not query:
        raise ValueError("query parameter is required")
    
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(f"query must be {_MAX_QUERY_LENGTH} characters or fewer")
    
    logger.info(f"Search query: {query}")

    query_lower = query.lower()
    matched = [doc for doc, normalized in LOWERCASE_DOCS_DATABASE if query_lower in normalized]
    
    return {
        "query": query,
        "results": matched,
        "count": len(matched),
    }


registry.register(
    Tool(
        name="search_docs",
        description="Search documentation for relevant information.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        handler=search_docs_handler,
        required_params=["query"],
    )
)


# -------------------------
# Tool 3: Add Numbers
# -------------------------
def add_numbers_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add two numbers together."""
    if "a" not in args or "b" not in args:
        missing = [p for p in ("a", "b") if p not in args]
        raise ValueError(f"Missing required parameters: {missing}")
    try:
        a = float(args["a"])
        b = float(args["b"])
    except (TypeError, ValueError):
        raise ValueError("Parameters 'a' and 'b' must be numeric")
    
    logger.info(f"Add numbers: {a} + {b}")
    
    return {
        "a": a,
        "b": b,
        "result": a + b,
    }


registry.register(
    Tool(
        name="add_numbers",
        description="Add two numbers together.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        handler=add_numbers_handler,
        required_params=["a", "b"],
    )
)
