#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api import create_app
from backend.service import MultiTenantPlatformService


TEST_TENANTS = {
    "company-a": {
        "display_name": "Company A",
        "allowed_users": ["alice", "bob"],
        "token": "token-company-a",
        "model": "fallback",
        "rate_limit_per_minute": 2,
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
        "allowed_users": ["charlie"],
        "token": "token-company-b",
        "model": "fallback",
        "rate_limit_per_minute": 5,
        "workflow_name": "customer-support",
        "dify_app_id": "dify-company-b",
        "issuer": "multitenant-demo",
        "audience": "multitenant-api",
        "retention_days": 60,
        "compliance_tags": ["GDPR", "HIPAA"],
        "user_roles": {"charlie": "editor"},
    },
}


class MultiTenantApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.storage_file = base / "sessions.json"
        self.session_map_file = base / "session_map.json"
        self.tenant_config_file = base / "tenant_configs.json"
        self.tenant_config_file.write_text(
            json.dumps(TEST_TENANTS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.platform_service = MultiTenantPlatformService(
            storage_file=self.storage_file,
            session_map_file=self.session_map_file,
            tenant_config_file=self.tenant_config_file,
            audit_log_file=base / "audit.jsonl",
            compliance_file=base / "compliance.json",
            encryption_key_file=base / "aes.key",
            security_dir=base / "security",
        )
        self.client = TestClient(create_app(self.platform_service))

    def issue_token(self, tenant_id: str, user_id: str, expires: int = 3600) -> str:
        return self.platform_service.issue_demo_token(tenant_id, user_id, expires_in_seconds=expires)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_healthz(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_authentication_failure(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": "Bearer bad-token"},
            json={"user_id": "alice", "message": "hello"},
        )
        self.assertEqual(response.status_code, 401)

    def test_authorization_failure(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {self.issue_token('company-a', 'alice')}"},
            json={"user_id": "charlie", "message": "hello"},
        )
        self.assertEqual(response.status_code, 401)

    def test_tenant_session_isolation(self) -> None:
        token = self.issue_token("company-a", "alice")
        first = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"user_id": "alice", "message": "My hometown is in Beijing."},
        )
        self.assertEqual(first.status_code, 200)
        session_id = first.json()["session_id"]

        follow_up = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "user_id": "alice",
                "session_id": session_id,
                "message": "Where is my hometown?",
            },
        )
        self.assertEqual(follow_up.status_code, 200)
        self.assertIn("Beijing", follow_up.json()["answer"])

        other_tenant = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {self.issue_token('company-b', 'charlie')}"},
            json={"user_id": "charlie", "message": "Where is my hometown?"},
        )
        self.assertEqual(other_tenant.status_code, 200)
        self.assertNotIn("Beijing", other_tenant.json()["answer"])

    def test_rate_limit(self) -> None:
        headers = {"Authorization": f"Bearer {self.issue_token('company-a', 'alice')}"}
        payload = {"user_id": "alice", "message": "hello"}
        first = self.client.post("/api/v1/chat", headers=headers, json=payload)
        second = self.client.post("/api/v1/chat", headers=headers, json=payload)
        third = self.client.post("/api/v1/chat", headers=headers, json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)

    def test_rbac_and_compliance_delete(self) -> None:
        admin_token = self.issue_token("company-a", "alice")
        create = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"user_id": "alice", "message": "My hometown is in Beijing."},
        )
        self.assertEqual(create.status_code, 200)

        export_resp = self.client.get(
            "/api/v1/compliance/export/company-a/alice",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(export_resp.status_code, 200)
        self.assertTrue(export_resp.json()["data"])

        delete_resp = self.client.request(
            "DELETE",
            "/api/v1/compliance/delete/company-a",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"user_id": "alice", "reason": "gdpr_erasure"},
        )
        self.assertEqual(delete_resp.status_code, 200)

    def test_viewer_cannot_write(self) -> None:
        viewer_token = self.issue_token("company-a", "bob")
        response = self.client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {viewer_token}"},
            json={"user_id": "bob", "message": "hello"},
        )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
