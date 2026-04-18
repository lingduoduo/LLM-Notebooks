#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .models import TenantConfig


BASE_DIR = Path(__file__).resolve().parent.parent
TENANT_CONFIG_FILE = BASE_DIR / "tenant_configs.json"
RUNTIME_DIR = Path(os.getenv("MULTITENANT_RUNTIME_DIR", str(BASE_DIR / ".runtime")))
SESSION_FILE = RUNTIME_DIR / "sessions.json"
SESSION_MAP_FILE = RUNTIME_DIR / "session_map.json"
SECURITY_DIR = RUNTIME_DIR / "security"
AUDIT_LOG_FILE = RUNTIME_DIR / "audit" / "api_audit.jsonl"
COMPLIANCE_FILE = RUNTIME_DIR / "audit" / "compliance_lifecycle.json"
ENCRYPTION_KEY_FILE = SECURITY_DIR / "aes256.key"
ALLOW_DEMO_TOKEN_ISSUANCE = os.getenv("MULTITENANT_ENABLE_DEMO_TOKEN_ISSUANCE", "false").lower() == "true"

RATE_LIMIT_WINDOW_SECONDS: int = 60
RATE_LIMIT_CLEANUP_INTERVAL: int = 3600
JWT_DEFAULT_EXPIRY_SECONDS: int = 3600
CONVERSATION_HISTORY_WINDOW: int = 8


DEFAULT_TENANTS = {
    "company-a": {
        "display_name": "Company A",
        "allowed_users": ["alice", "bob"],
        "model": "fallback",
        "rate_limit_per_minute": 30,
        "workflow_name": "customer-support",
        "dify_app_id": "dify-company-a",
        "issuer": "multitenant-demo",
        "audience": "multitenant-api",
        "retention_days": 30,
        "compliance_tags": ["GDPR", "SOX"],
        "user_roles": {"alice": "admin", "bob": "viewer"},
    },
    "company-b": {
        "display_name": "Company B",
        "allowed_users": ["charlie", "diana"],
        "model": "fallback",
        "rate_limit_per_minute": 20,
        "workflow_name": "customer-support",
        "dify_app_id": "dify-company-b",
        "issuer": "multitenant-demo",
        "audience": "multitenant-api",
        "retention_days": 60,
        "compliance_tags": ["GDPR", "HIPAA"],
        "user_roles": {"charlie": "editor", "diana": "viewer"},
    },
    "enterprise-x": {
        "display_name": "Enterprise X",
        "allowed_users": ["manager1", "manager2"],
        "model": "fallback",
        "rate_limit_per_minute": 60,
        "workflow_name": "premium-support",
        "dify_app_id": "dify-enterprise-x",
        "issuer": "multitenant-demo",
        "audience": "multitenant-api",
        "retention_days": 90,
        "compliance_tags": ["GDPR", "SOX", "HIPAA"],
        "user_roles": {"manager1": "admin", "manager2": "editor"},
    },
}


def ensure_default_tenant_config() -> None:
    if TENANT_CONFIG_FILE.exists():
        return
    TENANT_CONFIG_FILE.write_text(
        json.dumps(DEFAULT_TENANTS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tenant_configs(config_file: Optional[Path] = None) -> Dict[str, TenantConfig]:
    config_path = config_file or TENANT_CONFIG_FILE
    if config_path == TENANT_CONFIG_FILE:
        ensure_default_tenant_config()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        tenant_id: TenantConfig(tenant_id=tenant_id, **payload)
        for tenant_id, payload in raw.items()
    }
