"""
State and helpers for a generic RPA workflow.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict


class AuditEvent(TypedDict):
    timestamp: float
    task_id: str
    event: str
    status: str
    detail: dict[str, Any]


class RPAState(TypedDict, total=False):
    task_id: str
    user_request: str
    messages: list[Any]
    discovered_tools: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    plan: dict[str, Any]
    extracted_data: dict[str, Any]
    execution_log: list[dict[str, Any]]
    verification: dict[str, Any]
    approval_required: bool
    approval_status: str | None
    final_result: dict[str, Any]
    audit_log: list[AuditEvent]
    error: str | None


def default_rpa_request() -> str:
    """Return a deterministic sample request for local demos."""
    return "Process record ID R123 customer Alice amount 250 status pending"


def default_trade_request() -> str:
    """Backward-compatible alias for older notebooks."""
    return default_rpa_request()


def new_task_id() -> str:
    """Create a trace id for one RPA run."""
    return str(uuid.uuid4())


def audit_event(
    task_id: str,
    event: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> AuditEvent:
    """Build a structured audit event for RPA observability."""
    return {
        "timestamp": time.time(),
        "task_id": task_id,
        "event": event,
        "status": status,
        "detail": detail or {},
    }
