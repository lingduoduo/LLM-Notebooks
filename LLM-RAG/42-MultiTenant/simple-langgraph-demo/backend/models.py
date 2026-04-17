#!/usr/bin/env python3

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, TypedDict

from typing_extensions import Annotated

from .compat import BaseMessage, add_messages


@dataclass
class TenantContext:
    tenant_id: str
    user_id: str
    session_id: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = field(default_factory=lambda: str(threading.current_thread().ident or "0"))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    scopes: List[str] = field(default_factory=list)

    def display_info(self) -> str:
        return (
            f"[tenant:{self.tenant_id}|user:{self.user_id}|"
            f"session:{self.session_id}|request:{self.request_id[:8]}]"
        )


class ChatState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[TenantContext]
    metadata: Dict[str, Any]
    user_memory: Dict[str, Any]
    conversation_summary: str
    last_topics: List[str]


@dataclass
class TenantConfig:
    tenant_id: str
    display_name: str
    allowed_users: List[str]
    token: str
    model: str = "fallback"
    rate_limit_per_minute: int = 30
    workflow_name: str = "support"
    dify_app_id: Optional[str] = None
    issuer: str = "multitenant-demo"
    audience: str = "multitenant-api"
    retention_days: int = 30
    compliance_tags: List[str] = field(default_factory=list)
    user_roles: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChatResult:
    tenant_id: str
    user_id: str
    session_id: str
    answer: str
    model: str
    workflow: str
    metadata: Dict[str, Any]


@dataclass
class ComplianceDeletionRecord:
    tenant_id: str
    user_id: str
    session_id: Optional[str]
    requested_at: str
    reason: str
    status: str
    deleted_at: Optional[str] = None
