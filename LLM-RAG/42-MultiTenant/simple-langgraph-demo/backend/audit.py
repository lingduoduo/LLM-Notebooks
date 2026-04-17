#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    def __init__(self, audit_file: Path):
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
        with self._lock:
            with self.audit_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
