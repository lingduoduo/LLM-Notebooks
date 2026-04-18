"""
Generic MCP-style tool registry for local RPA examples.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)
_UPSERTED_RECORDS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and handler for a discoverable RPA tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """Small in-process MCP-style tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self.list_tools.cache_clear()

    @lru_cache(maxsize=1)
    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        _validate_arguments(tool, arguments)
        logger.info("Calling RPA tool: %s", name)
        return tool.handler(arguments)


def _validate_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> None:
    schema = tool.input_schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"Missing required parameters for {tool.name}: {missing}")

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ValueError(f"Unknown parameters for {tool.name}: {unknown}")

    for name, value in arguments.items():
        expected_type = properties.get(name, {}).get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"Parameter '{name}' must be a string")
        if expected_type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{name}' must be numeric")
        if expected_type == "array" and not isinstance(value, list):
            raise ValueError(f"Parameter '{name}' must be an array")
        if expected_type == "object" and not isinstance(value, dict):
            raise ValueError(f"Parameter '{name}' must be an object")


def extract_record_fields(args: dict[str, Any]) -> dict[str, Any]:
    """Extract generic record fields from semi-structured text."""
    raw_text = args["raw_text"]
    record_id = _extract_pattern(raw_text, r"\b(?:ID|RECORD)\s*[:=]?\s*([A-Za-z]\d+)\b", "R123")
    customer = _extract_pattern(
        raw_text,
        r"\bcustomer\s+([A-Za-z][A-Za-z\s-]*?)(?=\s+(?:amount|status|id|record)\b|$)",
        "Alice",
    )
    amount = _extract_number(raw_text, r"\bamount\s*[:=]?\s*(\d+(?:\.\d+)?)", 250)
    status = _extract_pattern(raw_text, r"\bstatus\s+([A-Za-z_]+)", "pending").lower()

    return {
        "record_id": record_id,
        "customer": customer.strip(),
        "amount": amount,
        "status": status,
        "confidence": 0.95,
    }


def validate_record(args: dict[str, Any]) -> dict[str, Any]:
    """Validate a generic business record before write operations."""
    record = args["record"]
    issues = []

    if not record.get("record_id"):
        issues.append("record_id is required")
    if float(record.get("amount", 0)) <= 0:
        issues.append("amount must be greater than zero")
    if record.get("status") not in {"pending", "approved", "open", "new"}:
        issues.append(f"unsupported status: {record.get('status')}")

    return {
        "valid": not issues,
        "record_id": record.get("record_id"),
        "issues": issues,
    }


def upsert_record(args: dict[str, Any]) -> dict[str, Any]:
    """Create or update a record idempotently in a simulated target system."""
    record = dict(args["record"])
    record_id = record["record_id"]

    if record_id in _UPSERTED_RECORDS and _UPSERTED_RECORDS[record_id] == record:
        return {
            "status": "unchanged",
            "record_id": record_id,
            "idempotent": True,
            "record": _UPSERTED_RECORDS[record_id],
        }

    operation = "updated" if record_id in _UPSERTED_RECORDS else "created"
    _UPSERTED_RECORDS[record_id] = record
    return {
        "status": operation,
        "record_id": record_id,
        "idempotent": False,
        "record": record,
    }


def generate_report(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a generic RPA run report."""
    records = args["records"]
    failures = [record for record in records if record.get("status") == "error"]
    invalid = [record for record in records if record.get("valid") is False]
    return {
        "total": len(records),
        "failures": len(failures),
        "invalid": len(invalid),
        "requires_review": bool(failures or invalid),
        "records": records,
    }


def _extract_pattern(text: str, pattern: str, default: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip(" .,!?:;") if match else default


def _extract_number(text: str, pattern: str, default: float) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def build_registry() -> ToolRegistry:
    """Create a registry with generic RPA tools registered."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="extract_record_fields",
            description="Extract generic record fields from raw text.",
            input_schema={
                "type": "object",
                "properties": {"raw_text": {"type": "string"}},
                "required": ["raw_text"],
            },
            handler=extract_record_fields,
        )
    )
    registry.register(
        ToolSpec(
            name="validate_record",
            description="Validate a generic business record.",
            input_schema={
                "type": "object",
                "properties": {"record": {"type": "object"}},
                "required": ["record"],
            },
            handler=validate_record,
        )
    )
    registry.register(
        ToolSpec(
            name="upsert_record",
            description="Create or update a record idempotently.",
            input_schema={
                "type": "object",
                "properties": {"record": {"type": "object"}},
                "required": ["record"],
            },
            handler=upsert_record,
        )
    )
    registry.register(
        ToolSpec(
            name="generate_report",
            description="Generate a generic RPA execution report.",
            input_schema={
                "type": "object",
                "properties": {"records": {"type": "array"}},
                "required": ["records"],
            },
            handler=generate_report,
        )
    )
    return registry


registry = build_registry()


async def discover_mcp_tools() -> list[dict[str, Any]]:
    """Async wrapper matching an MCP discovery call."""
    return registry.list_tools()


async def call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Async wrapper matching an MCP tool invocation."""
    return registry.call(name, arguments)
