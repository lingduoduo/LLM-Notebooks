# logger.py
"""
Logging and audit system for Human-in-the-Loop Agent.
Provides structured logging and audit trail capabilities.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, ENABLE_AUDIT_LOG, AUDIT_LOG_FILE, PURCHASE_CURRENCY


@dataclass
class AuditEvent:
    """Audit event data structure."""
    thread_id: str
    user_id: Optional[str]
    action: str
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_type: str = "agent_action"
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AuditLogger:
    """Handles audit logging for compliance and tracking."""

    def __init__(self, log_file: str | Path = AUDIT_LOG_FILE):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event to file."""
        if not ENABLE_AUDIT_LOG:
            return

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            # Fallback to basic logging if audit logging fails
            logging.error(f"Failed to write audit log: {e}")

    def log_purchase_approval(
        self,
        thread_id: str,
        user_id: Optional[str],
        item: str,
        price: float,
        vendor: str,
        decision: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log purchase approval decision."""
        event = AuditEvent(
            event_type="purchase_approval",
            thread_id=thread_id,
            user_id=user_id,
            action=f"purchase_{decision}",
            details={
                "item": item,
                "price": price,
                "vendor": vendor,
                "currency": PURCHASE_CURRENCY,
                **(details or {})
            },
            result=decision
        )
        self.log_event(event)

    def log_agent_action(
        self,
        thread_id: str,
        user_id: Optional[str],
        action: str,
        details: Dict[str, Any]
    ) -> None:
        """Log general agent actions."""
        event = AuditEvent(
            event_type="agent_action",
            thread_id=thread_id,
            user_id=user_id,
            action=action,
            details=details
        )
        self.log_event(event)


# Global audit logger instance
audit_logger = AuditLogger()


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = LOG_FILE,
    format_string: str = LOG_FORMAT
) -> None:
    """Setup comprehensive logging configuration."""
    if level is None:
        level = LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    root_logger.addHandler(console_handler)

    # Set specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP noise
    logging.getLogger("openai").setLevel(logging.WARNING)  # Reduce OpenAI noise

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


