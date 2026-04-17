#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .audit import AuditLogger
from .models import ComplianceDeletionRecord


class ComplianceManager:
    def __init__(self, lifecycle_file: Path, audit_logger: AuditLogger):
        self.lifecycle_file = lifecycle_file
        self.audit_logger = audit_logger
        self.lifecycle_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records = self._load()

    def _load(self) -> list[Dict[str, Any]]:
        if not self.lifecycle_file.exists():
            return []
        try:
            return json.loads(self.lifecycle_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self) -> None:
        self.lifecycle_file.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def request_deletion(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str],
        reason: str,
    ) -> ComplianceDeletionRecord:
        record = ComplianceDeletionRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            requested_at=datetime.now(UTC).isoformat(),
            reason=reason,
            status="requested",
        )
        with self._lock:
            self._records.append(record.__dict__)
            self._save()
        self.audit_logger.log(
            "compliance.deletion_requested",
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "reason": reason,
            },
        )
        return record

    def mark_deleted(self, *, tenant_id: str, user_id: str, session_id: Optional[str]) -> None:
        with self._lock:
            for record in reversed(self._records):
                if (
                    record["tenant_id"] == tenant_id
                    and record["user_id"] == user_id
                    and record["session_id"] == session_id
                    and record["status"] == "requested"
                ):
                    record["status"] = "deleted"
                    record["deleted_at"] = datetime.now(UTC).isoformat()
                    break
            self._save()
        self.audit_logger.log(
            "compliance.deletion_completed",
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
            },
        )
