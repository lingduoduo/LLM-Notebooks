"""
State and helpers for a finance RPA workflow.
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


class PlanStep(TypedDict, total=False):
    tool_name: str
    arguments: dict[str, Any]
    arguments_from: str
    save_as: str
    when: dict[str, Any]


class WorkflowPlan(TypedDict):
    requires_human_approval: bool
    steps: list[PlanStep]


class RPAState(TypedDict, total=False):
    task_id: str
    user_request: str
    discovered_tools: list[dict[str, Any]]
    plan: WorkflowPlan
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
    return "Process invoice ID INV-1001 vendor Acme amount 2500 currency USD status pending"


def new_task_id() -> str:
    """Create a trace id for one RPA run."""
    return str(uuid.uuid4())


def create_initial_state(user_request: str | None = None) -> RPAState:
    """Create the minimal state required to start one workflow run."""
    return {
        "task_id": new_task_id(),
        "user_request": user_request or default_rpa_request(),
        "audit_log": [],
    }


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
