"""
Finance-focused MCP-style tool registry for local RPA examples.
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)
_UPSERTED_INVOICES: dict[str, dict[str, Any]] = {}
VALID_INVOICE_STATUSES = {"pending", "approved", "paid", "open"}
EXTRACT_INVOICE_FIELDS = "extract_invoice_fields"
VALIDATE_INVOICE = "validate_invoice"
UPSERT_INVOICE = "upsert_invoice"
GENERATE_FINANCE_REPORT = "generate_finance_report"
FINANCE_TOOL_SEQUENCE = (
    EXTRACT_INVOICE_FIELDS,
    VALIDATE_INVOICE,
    UPSERT_INVOICE,
    GENERATE_FINANCE_REPORT,
)
FINANCE_REPORTABLE_TOOLS = frozenset({VALIDATE_INVOICE, UPSERT_INVOICE})


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

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(tool.input_schema),
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


def object_schema(properties: dict[str, dict[str, str]], required: list[str]) -> dict[str, Any]:
    """Build the small JSON-schema shape used by local tool specs."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


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
        if expected_type == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise ValueError(f"Parameter '{name}' must be numeric")
        if expected_type == "array" and not isinstance(value, list):
            raise ValueError(f"Parameter '{name}' must be an array")
        if expected_type == "object" and not isinstance(value, dict):
            raise ValueError(f"Parameter '{name}' must be an object")


def extract_invoice_fields(args: dict[str, Any]) -> dict[str, Any]:
    """Extract finance invoice fields from semi-structured text."""
    raw_text = args["raw_text"]
    invoice_id = _extract_pattern(
        raw_text,
        r"\b(?:invoice\s+)?ID\s*[:=]?\s*([A-Za-z]+[-_]?\d+)\b",
    )
    vendor = _extract_pattern(
        raw_text,
        r"\bvendor\s+([A-Za-z][A-Za-z\s&.-]*?)(?=\s+(?:amount|currency|status|id|invoice)\b|$)",
    )
    amount = _extract_number(raw_text, r"\bamount\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)")
    currency = _extract_pattern(raw_text, r"\bcurrency\s+([A-Z]{3})\b")
    status = _extract_pattern(raw_text, r"\bstatus\s+([A-Za-z_]+)")
    normalized_status = status.lower() if status else None
    extracted_count = sum(
        value is not None
        for value in (invoice_id, vendor, amount, currency, normalized_status)
    )

    return {
        "invoice_id": invoice_id,
        "vendor": vendor.strip() if vendor else None,
        "amount": amount,
        "currency": currency,
        "status": normalized_status,
        "confidence": round(extracted_count / 5, 2),
    }


def validate_invoice(args: dict[str, Any]) -> dict[str, Any]:
    """Validate a finance invoice before write operations."""
    invoice = args["invoice"]
    issues = []
    amount = invoice.get("amount")

    if not invoice.get("invoice_id"):
        issues.append("invoice_id is required")
    if not invoice.get("vendor"):
        issues.append("vendor is required")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        issues.append("amount must be numeric")
    elif amount <= 0:
        issues.append("amount must be greater than zero")
    if invoice.get("currency") != "USD":
        issues.append(f"unsupported currency: {invoice.get('currency')}")
    if invoice.get("status") not in VALID_INVOICE_STATUSES:
        issues.append(f"unsupported status: {invoice.get('status')}")

    return {
        "valid": not issues,
        "invoice_id": invoice.get("invoice_id"),
        "issues": issues,
    }


def upsert_invoice(args: dict[str, Any]) -> dict[str, Any]:
    """Create or update an invoice idempotently in a simulated finance system."""
    invoice = deepcopy(args["invoice"])
    invoice_id = invoice["invoice_id"]

    if invoice_id in _UPSERTED_INVOICES and _UPSERTED_INVOICES[invoice_id] == invoice:
        return {
            "status": "unchanged",
            "invoice_id": invoice_id,
            "idempotent": True,
            "invoice": deepcopy(_UPSERTED_INVOICES[invoice_id]),
        }

    operation = "updated" if invoice_id in _UPSERTED_INVOICES else "created"
    _UPSERTED_INVOICES[invoice_id] = deepcopy(invoice)
    return {
        "status": operation,
        "invoice_id": invoice_id,
        "idempotent": False,
        "invoice": invoice,
    }


def generate_finance_report(args: dict[str, Any]) -> dict[str, Any]:
    """Generate a finance RPA run report."""
    invoices = args["invoices"]
    failures = [invoice for invoice in invoices if invoice.get("status") == "error"]
    invalid = [invoice for invoice in invoices if invoice.get("valid") is False]
    return {
        "total": len(invoices),
        "failures": len(failures),
        "invalid": len(invalid),
        "requires_review": bool(failures or invalid),
        "invoices": invoices,
    }


def _extract_pattern(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip(" .,!?:;") if match else None


def _extract_number(text: str, pattern: str) -> int | float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def build_registry() -> ToolRegistry:
    """Create a registry with finance RPA tools registered."""
    registry = ToolRegistry()
    tools = (
        ToolSpec(
            EXTRACT_INVOICE_FIELDS,
            "Extract finance invoice fields from raw text.",
            object_schema({"raw_text": {"type": "string"}}, ["raw_text"]),
            extract_invoice_fields,
        ),
        ToolSpec(
            VALIDATE_INVOICE,
            "Validate a finance invoice.",
            object_schema({"invoice": {"type": "object"}}, ["invoice"]),
            validate_invoice,
        ),
        ToolSpec(
            UPSERT_INVOICE,
            "Create or update an invoice idempotently.",
            object_schema({"invoice": {"type": "object"}}, ["invoice"]),
            upsert_invoice,
        ),
        ToolSpec(
            GENERATE_FINANCE_REPORT,
            "Generate a finance RPA execution report.",
            object_schema({"invoices": {"type": "array"}}, ["invoices"]),
            generate_finance_report,
        ),
    )
    for tool in tools:
        registry.register(tool)
    return registry


registry = build_registry()


async def discover_mcp_tools() -> list[dict[str, Any]]:
    """Async wrapper matching an MCP discovery call."""
    return registry.list_tools()


async def call_mcp_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Async wrapper matching an MCP tool invocation."""
    return registry.call(name, arguments)
