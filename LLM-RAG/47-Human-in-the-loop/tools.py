# tools.py
"""
Tool definitions for Human-in-the-Loop Agent System.
Provides purchase and search functionality with HITL integration.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.tools import tool
from langgraph.types import interrupt

from config import PURCHASE_CURRENCY, get_product_info
from logger import get_logger, audit_logger

logger = get_logger(__name__)


@tool
def purchase_item(
    item: str,
    price: float,
    vendor: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Purchase item tool with Human-in-the-Loop approval.

    This tool triggers a human approval process before execution:
    1. Pause execution using interrupt() and send approval details
    2. Wait for a human to resume execution via Command(resume=...)
    3. Execute, modify parameters, or reject based on the approval result

    Args:
        item: Name of the item to purchase
        price: Price of the item
        vendor: Vendor/supplier name
        thread_id: Conversation thread ID for audit logging
        user_id: User ID for audit logging

    Returns:
        Purchase result message
    """
    # Input validation
    if not item or not isinstance(item, str):
        raise ValueError("item must be a non-empty string")
    if not isinstance(price, (int, float)) or price <= 0:
        raise ValueError("price must be a positive number")
    if not vendor or not isinstance(vendor, str):
        raise ValueError("vendor must be a non-empty string")

    # Sanitize inputs
    item = item.strip()
    vendor = vendor.strip()

    logger.info(f"Purchase request: {item} for {price} {PURCHASE_CURRENCY} from {vendor}")

    # Prepare approval request
    approval_data = {
        "action": "purchase",
        "item": item,
        "price": price,
        "vendor": vendor,
        "currency": PURCHASE_CURRENCY,
        "message": f"About to purchase {item} for {price} {PURCHASE_CURRENCY} from {vendor}. Please approve or suggest changes.",
        "thread_id": thread_id,
        "user_id": user_id
    }

    try:
        # Trigger human approval interrupt
        response = interrupt(approval_data)
        if isinstance(response, str):
            response = {"type": response}
        logger.info(f"Received approval response: {response.get('type', 'unknown')}")

        # Process approval response
        response_type = response.get("type", "").lower()

        if response_type in {"accept", "approved", "approve"}:
            # Approval granted - proceed with purchase
            result = f"Successfully purchased {item} for {price} {PURCHASE_CURRENCY} from {vendor}."
            audit_logger.log_purchase_approval(
                thread_id or "unknown",
                user_id,
                item,
                price,
                vendor,
                "approved"
            )

        elif response_type == "edit":
            # Parameters modified - update and proceed
            args = response.get("args", {})

            # Apply modifications
            new_item = args.get("item") or args.get("item_name", item)
            new_price = args.get("price", price)
            new_vendor = args.get("vendor", vendor)

            # Re-validate modified parameters
            if not isinstance(new_price, (int, float)) or new_price <= 0:
                raise ValueError("Modified price must be a positive number")

            result = f"Successfully purchased {new_item} for {new_price} {PURCHASE_CURRENCY} from {new_vendor} (parameters modified)."
            audit_logger.log_purchase_approval(
                thread_id or "unknown",
                user_id,
                new_item,
                new_price,
                new_vendor,
                "approved_with_changes",
                {"original_item": item, "original_price": price, "original_vendor": vendor}
            )

        elif response_type == "reject":
            # Purchase rejected
            result = "Purchase request has been rejected by human approval."
            audit_logger.log_purchase_approval(
                thread_id or "unknown",
                user_id,
                item,
                price,
                vendor,
                "rejected"
            )

        else:
            error_msg = f"Unknown approval response type: {response_type}"
            logger.error(error_msg)
            audit_logger.log_purchase_approval(
                thread_id or "unknown",
                user_id,
                item,
                price,
                vendor,
                "error",
                {"error": error_msg}
            )
            raise ValueError(error_msg)

        logger.info(f"Purchase completed: {result}")
        return result

    except Exception as e:
        error_msg = f"Purchase failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        audit_logger.log_purchase_approval(
            thread_id or "unknown",
            user_id,
            item,
            price,
            vendor,
            "error",
            {"error": str(e)}
        )
        raise


@tool
def search_product(query: str) -> list[dict[str, Any]]:
    """
    Product search tool with enhanced matching.

    Searches the product database for matching items and returns
    product information and pricing details.

    Args:
        query: Search query string

    Returns:
        Formatted search results
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    logger.info(f"Product search: {query}")

    try:
        product_info = get_product_info(query)
        result = [{
            "name": product_info["name"],
            "price": product_info["price"],
            "vendor": product_info["vendor"],
            "currency": product_info["currency"],
            "price_range": product_info["price_range"],
            "vendors": product_info["vendors"],
        }]

        logger.debug(f"Search result: {result}")
        return result

    except Exception as e:
        error_msg = f"Product search failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


# Tool registry for easy access
TOOLS = [purchase_item, search_product]
TOOL_NAMES = [tool.name for tool in TOOLS]
REQUIRED_TOOL_ARGS = {
    "purchase_item": ("item", "price", "vendor"),
    "search_product": ("query",),
}

def get_tool_by_name(name: str) -> Optional[object]:
    """Get a tool by name."""
    for registered_tool in TOOLS:
        if registered_tool.name == name:
            return registered_tool
    return None

def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> bool:
    """Validate tool arguments."""
    try:
        tool = get_tool_by_name(tool_name)
        if not tool:
            return False

        # Basic validation - could be enhanced with more sophisticated schema validation
        if tool_name == "purchase_item":
            required = REQUIRED_TOOL_ARGS[tool_name]
            for req in required:
                if req not in args:
                    logger.warning(f"Missing required argument: {req}")
                    return False
            if not isinstance(args.get("price"), (int, float)) or args["price"] <= 0:
                logger.warning("Invalid price argument")
                return False
            if not isinstance(args.get("item"), str) or not args["item"].strip():
                logger.warning("Invalid item argument")
                return False
            if not isinstance(args.get("vendor"), str) or not args["vendor"].strip():
                logger.warning("Invalid vendor argument")
                return False

        elif tool_name == "search_product":
            if "query" not in args or not isinstance(args["query"], str) or not args["query"].strip():
                logger.warning("Invalid query argument")
                return False

        return True

    except Exception as e:
        logger.error(f"Tool validation error: {e}")
        return False
