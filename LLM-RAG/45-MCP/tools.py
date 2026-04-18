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
        _list_tools_cached.cache_clear()
        logger.info("Tool registered: %s", tool.name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools with metadata."""
        result = _list_tools_cached(tuple(self._tools.keys()), self)
        logger.debug("Listing %s tools", len(result))
        return result

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with validation."""
        if name not in self._tools:
            logger.error("Tool not found: %s", name)
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]

        if tool.required_params:
            missing = [p for p in tool.required_params if p not in arguments]
            if missing:
                logger.error("Missing parameters for tool %s: %s", name, missing)
                raise ValueError(f"Missing required parameters: {missing}")
        self._validate_arguments(tool, arguments)

        logger.info("Calling tool: %s with args: %s", name, arguments)
        try:
            result = tool.handler(arguments)
            logger.debug("Tool %s executed successfully", name)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            raise

    def _validate_arguments(self, tool: Tool, arguments: Dict[str, Any]) -> None:
        """Validate arguments against the tool's JSON-schema-like metadata."""
        schema = tool.input_schema
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return

        allowed = set(properties)
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unknown parameters for {tool.name}: {unknown}")

        for name, value in arguments.items():
            prop_schema = properties.get(name, {})
            if not isinstance(prop_schema, dict):
                continue

            expected_type = prop_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Parameter '{name}' must be a string")
            if expected_type == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{name}' must be numeric")

            enum_values = prop_schema.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                raise ValueError(f"Parameter '{name}' must be one of {enum_values}")


@lru_cache(maxsize=8)
def _list_tools_cached(
    _tool_names: tuple[str, ...], reg: "ToolRegistry"
) -> List[Dict[str, Any]]:
    """Module-level cache for the serialized tool catalog.

    Keyed on the ordered tuple of registered tool names so it auto-invalidates
    whenever a tool is added (register() calls cache_clear()).
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in reg._tools.values()
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
    
    logger.info("Weather query: %s (%s)", city, unit)
    
    # Mock weather data
    temp = 22 if unit == "celsius" else 72
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny",
        "display_hint": "weather",
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
    
    logger.info("Search query: %s", query)

    query_lower = query.lower()
    matched = [doc for doc, normalized in LOWERCASE_DOCS_DATABASE if query_lower in normalized]
    
    return {
        "query": query,
        "results": matched,
        "count": len(matched),
        "display_hint": "search_results",
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
    
    logger.info("Add numbers: %s + %s", a, b)
    
    return {
        "a": a,
        "b": b,
        "result": a + b,
        "display_hint": "arithmetic",
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
