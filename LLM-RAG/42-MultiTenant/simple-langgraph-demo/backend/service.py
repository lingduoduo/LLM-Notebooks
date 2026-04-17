#!/usr/bin/env python3

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .audit import AuditLogger
from .auth import ApiGatewayAuth, InMemoryRateLimiter, ROLE_SCOPES
from .compliance import ComplianceManager
from .compat import AIMessage
from .config import (
    AUDIT_LOG_FILE,
    COMPLIANCE_FILE,
    ENCRYPTION_KEY_FILE,
    SECURITY_DIR,
    SESSION_FILE,
    SESSION_MAP_FILE,
)
from .graph_engine import LangGraphExecutor
from .model_provider import ModelRouter
from .models import ChatResult, TenantConfig, TenantContext
from .security import AESGCMCipher, JWTService, RSAKeyManager
from .storage import MultiTenantStorage, SessionManager, TenantConfigRepository


_context_local = threading.local()


def get_current_context() -> Optional[TenantContext]:
    return getattr(_context_local, "current_context", None)


@contextmanager
def tenant_context(tenant_id: str, user_id: str, session_id: Optional[str] = None):
    previous = get_current_context()
    sid = session_id or global_session_manager.get_session_id(tenant_id, user_id)
    ctx = TenantContext(tenant_id=tenant_id, user_id=user_id, session_id=sid)
    _context_local.current_context = ctx
    try:
        yield ctx
    finally:
        _context_local.current_context = previous


class DifyWorkflowService:
    def __init__(self, executor: LangGraphExecutor):
        self.executor = executor

    def run(self, context: TenantContext, tenant_config: TenantConfig, message: str) -> ChatResult:
        state = self.executor.invoke(context, tenant_config, message)
        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break
        return ChatResult(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            session_id=context.session_id,
            answer=answer or "(No AI reply generated)",
            model=tenant_config.model,
            workflow=tenant_config.workflow_name,
            metadata=state.get("metadata", {}),
        )


class MultiTenantPlatformService:
    def __init__(
        self,
        storage_file: Optional[Path] = None,
        session_map_file: Optional[Path] = None,
        tenant_config_file: Optional[Path] = None,
        audit_log_file: Optional[Path] = None,
        compliance_file: Optional[Path] = None,
        encryption_key_file: Optional[Path] = None,
        security_dir: Optional[Path] = None,
        tenant_repo: Optional[TenantConfigRepository] = None,
    ):
        self.tenant_repo = tenant_repo or TenantConfigRepository(config_file=tenant_config_file)
        self.session_manager = SessionManager(file_path=session_map_file or SESSION_MAP_FILE)
        self.audit_logger = AuditLogger(audit_log_file or AUDIT_LOG_FILE)
        self.compliance_manager = ComplianceManager(
            compliance_file or COMPLIANCE_FILE,
            self.audit_logger,
        )
        self.key_manager = RSAKeyManager(security_dir or SECURITY_DIR)
        self.jwt_service = JWTService(self.key_manager)
        self.encryption_cipher = AESGCMCipher(encryption_key_file or ENCRYPTION_KEY_FILE)
        self.storage = MultiTenantStorage(
            storage_file=storage_file or SESSION_FILE,
            encryption_cipher=self.encryption_cipher,
            audit_logger=self.audit_logger,
            compliance_manager=self.compliance_manager,
        )
        self.auth = ApiGatewayAuth(self.tenant_repo, self.jwt_service)
        self.rate_limiter = InMemoryRateLimiter()
        self.executor = LangGraphExecutor(self.storage, ModelRouter())
        self.dify = DifyWorkflowService(self.executor)

    def handle_message(
        self,
        tenant_id: str,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
    ) -> ChatResult:
        tenant_config = self.tenant_repo.get(tenant_id)
        if not tenant_config:
            raise ValueError(f"Unknown tenant '{tenant_id}'")
        resolved_session_id = session_id or self.session_manager.get_session_id(tenant_id, user_id)
        context = TenantContext(tenant_id=tenant_id, user_id=user_id, session_id=resolved_session_id)
        return self.dify.run(context, tenant_config, message)

    def handle_authenticated_message(
        self,
        token: str,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
    ) -> ChatResult:
        principal = self.auth.authenticate(token, user_id, required_scopes=["chat:write"])
        tenant_config = self.tenant_repo.get(principal.tenant_id)
        if not tenant_config:
            raise ValueError(f"Unknown tenant '{principal.tenant_id}'")
        self.rate_limiter.check(tenant_config, user_id)
        resolved_session_id = session_id or self.session_manager.get_session_id(principal.tenant_id, principal.user_id)
        context = TenantContext(tenant_id=principal.tenant_id, user_id=principal.user_id, session_id=resolved_session_id)
        result = self.dify.run(context, tenant_config, message)
        self.audit_logger.log(
            "api.chat.success",
            {
                "tenant_id": principal.tenant_id,
                "user_id": principal.user_id,
                "role": principal.role,
                "scopes": principal.scopes,
                "session_id": result.session_id,
            },
        )
        return result

    def issue_demo_token(self, tenant_id: str, user_id: str, expires_in_seconds: int = 3600) -> str:
        config = self.tenant_repo.get(tenant_id)
        if not config:
            raise ValueError(f"Unknown tenant '{tenant_id}'")
        role = config.user_roles.get(user_id, "viewer")
        scopes = ROLE_SCOPES.get(role, ["chat:read"])
        return self.jwt_service.issue_token(
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            scopes=scopes,
            issuer=config.issuer,
            audience=config.audience,
            expires_in_seconds=expires_in_seconds,
        )


class MultiTenantCustomerService:
    def __init__(self, platform_service: Optional[MultiTenantPlatformService] = None):
        self.platform_service = platform_service or global_platform_service

    def process_message(self, message: str) -> str:
        context = get_current_context()
        if not context:
            raise RuntimeError("No tenant context detected; call inside tenant_context")
        result = self.platform_service.handle_message(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            message=message,
            session_id=context.session_id,
        )
        return result.answer


global_platform_service = MultiTenantPlatformService()
global_session_manager = global_platform_service.session_manager
global_storage = global_platform_service.storage
