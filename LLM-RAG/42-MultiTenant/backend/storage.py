#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit import AuditLogger
from .compliance import ComplianceManager
from .config import (
    AUDIT_LOG_FILE,
    COMPLIANCE_FILE,
    ENCRYPTION_KEY_FILE,
    SESSION_FILE,
    SESSION_MAP_FILE,
    load_tenant_configs,
)
from .compat import AIMessage, BaseMessage, HumanMessage
from .models import ChatState, TenantConfig, TenantContext
from .security import AESGCMCipher


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    return {"type": "ai", "content": message.content}


def _deserialize_message(payload: Dict[str, Any]) -> BaseMessage:
    if payload.get("type") == "human":
        return HumanMessage(content=payload.get("content", ""))
    return AIMessage(content=payload.get("content", ""))


class TenantConfigRepository:
    def __init__(
        self,
        config_file: Optional[Path] = None,
        initial_configs: Optional[Dict[str, TenantConfig]] = None,
    ):
        if initial_configs is not None:
            self._configs = initial_configs
        else:
            self._configs = load_tenant_configs(config_file=config_file)

    def all(self) -> Dict[str, TenantConfig]:
        return dict(self._configs)

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        return self._configs.get(tenant_id)


class SessionManager:
    def __init__(self, file_path: Path = SESSION_MAP_FILE):
        self._file_path = file_path
        self._lock = threading.RLock()
        self._sessions: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return
        try:
            payload = json.loads(self._file_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._sessions = payload
        except Exception:
            self._sessions = {}

    def _save(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(self._sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}_{user_id}"

    def get_session_id(self, tenant_id: str, user_id: str) -> str:
        with self._lock:
            key = self._key(tenant_id, user_id)
            if key not in self._sessions:
                self._sessions[key] = str(uuid.uuid4())
                self._save()
            return self._sessions[key]

    def clear_session(self, tenant_id: str, user_id: str) -> str:
        with self._lock:
            key = self._key(tenant_id, user_id)
            self._sessions[key] = str(uuid.uuid4())
            self._save()
            return self._sessions[key]

    def get_all_sessions(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._sessions)


class MultiTenantStorage:
    def __init__(
        self,
        storage_file: Path = SESSION_FILE,
        encryption_cipher: Optional[AESGCMCipher] = None,
        audit_logger: Optional[AuditLogger] = None,
        compliance_manager: Optional[ComplianceManager] = None,
    ):
        self.storage_file = storage_file
        self._storage: Dict[str, Dict[str, Dict[str, ChatState]]] = {}
        self._cache: Dict[str, ChatState] = {}
        self._lock = threading.RLock()
        self.encryption_cipher = encryption_cipher or AESGCMCipher(ENCRYPTION_KEY_FILE)
        self.audit_logger = audit_logger or AuditLogger(AUDIT_LOG_FILE)
        self.compliance_manager = compliance_manager or ComplianceManager(COMPLIANCE_FILE, self.audit_logger)
        self._load_from_file()

    def _cache_key(self, context: TenantContext) -> str:
        return f"{context.tenant_id}:{context.user_id}:{context.session_id}"

    def _load_from_file(self) -> None:
        if not self.storage_file.exists():
            return
        try:
            payload = json.loads(self.storage_file.read_text(encoding="utf-8"))
        except Exception:
            return

        for tenant_id, tenant_data in payload.items():
            self._storage.setdefault(tenant_id, {})
            for user_id, user_sessions in tenant_data.items():
                self._storage[tenant_id].setdefault(user_id, {})
                for session_id, session_data in user_sessions.items():
                    self._storage[tenant_id][user_id][session_id] = {
                        "messages": [_deserialize_message(m) for m in session_data.get("messages", [])],
                        "context": None,
                        "metadata": session_data.get("metadata", {}),
                        "user_memory": self._restore_user_memory(session_data),
                        "conversation_summary": session_data.get("conversation_summary", ""),
                        "last_topics": session_data.get("last_topics", []),
                    }

    def _restore_user_memory(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        if "encrypted_user_memory" in session_data:
            return self.encryption_cipher.decrypt_json(session_data["encrypted_user_memory"])
        return session_data.get("user_memory", {})

    def _save_to_file(self) -> None:
        serializable: Dict[str, Any] = {}
        for tenant_id, tenant_data in self._storage.items():
            serializable[tenant_id] = {}
            for user_id, user_sessions in tenant_data.items():
                serializable[tenant_id][user_id] = {}
                for session_id, state in user_sessions.items():
                    metadata = dict(state.get("metadata", {}))
                    for key, value in list(metadata.items()):
                        if isinstance(value, datetime):
                            metadata[key] = value.isoformat()
                    serializable[tenant_id][user_id][session_id] = {
                        "messages": [_serialize_message(m) for m in state.get("messages", [])],
                        "metadata": metadata,
                        "encrypted_user_memory": self.encryption_cipher.encrypt_json(
                            state.get("user_memory", {})
                        ),
                        "conversation_summary": state.get("conversation_summary", ""),
                        "last_topics": state.get("last_topics", []),
                    }
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.storage_file.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_or_create_state(self, context: TenantContext) -> ChatState:
        with self._lock:
            cache_key = self._cache_key(context)
            if cache_key in self._cache:
                self._cache[cache_key]["context"] = context
                return self._cache[cache_key]

            self._storage.setdefault(context.tenant_id, {})
            self._storage[context.tenant_id].setdefault(context.user_id, {})
            user_sessions = self._storage[context.tenant_id][context.user_id]
            if context.session_id not in user_sessions:
                user_sessions[context.session_id] = {
                    "messages": [],
                    "context": context,
                    "metadata": {},
                    "user_memory": {},
                    "conversation_summary": "",
                    "last_topics": [],
                }
                self._save_to_file()
            else:
                user_sessions[context.session_id]["context"] = context

            self._cache[cache_key] = user_sessions[context.session_id]
            return user_sessions[context.session_id]

    def update_session(self, context: TenantContext, state: ChatState) -> None:
        with self._lock:
            self._storage.setdefault(context.tenant_id, {})
            self._storage[context.tenant_id].setdefault(context.user_id, {})
            self._storage[context.tenant_id][context.user_id][context.session_id] = state
            self._cache[self._cache_key(context)] = state
            self._save_to_file()
            self.audit_logger.log(
                "storage.session_updated",
                {
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "message_count": len(state.get("messages", [])),
                },
            )

    def list_sessions(self, tenant_id: str, user_id: str) -> List[str]:
        return list(self._storage.get(tenant_id, {}).get(user_id, {}).keys())

    def export_user_data(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        data = self._storage.get(tenant_id, {}).get(user_id, {})
        self.audit_logger.log(
            "compliance.export_user_data",
            {"tenant_id": tenant_id, "user_id": user_id, "session_count": len(data)},
        )
        return data

    def delete_user_data(self, tenant_id: str, user_id: str, reason: str = "user_request") -> None:
        user_sessions = self._storage.get(tenant_id, {}).get(user_id, {})
        for session_id in list(user_sessions.keys()):
            self.compliance_manager.request_deletion(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                reason=reason,
            )
            user_sessions.pop(session_id, None)
            self._cache.pop(f"{tenant_id}:{user_id}:{session_id}", None)
            self.compliance_manager.mark_deleted(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )
        self._save_to_file()

    def show_isolation_status(self) -> str:
        lines = ["Multi-tenant data overview:"]
        for tenant_id, tenant_data in self._storage.items():
            lines.append(f"  Tenant {tenant_id}:")
            for user_id, user_sessions in tenant_data.items():
                lines.append(f"    User {user_id}: {len(user_sessions)} session(s)")
        return "\n".join(lines)
