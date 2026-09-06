"""OpenAI-compatible API client and evidence receipts for real LLM calls.

Follow the conventions in chapter8/self-modifying-agent/llm_generator.py:
record raw requests/responses, token usage, latency, and request/response hashes.
Store full evidence in validation/<run>/evidence.json and point
validation/latest.json to the latest run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path(__file__).resolve().parent

PROVIDER_DEFAULTS = {
    "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini"),
    "ark": ("ARK_API_KEY", "https://ark.cn-beijing.volces.com/api/v3", None),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
}


def make_client(provider: str) -> tuple[OpenAI, dict[str, Any]]:
    """Return an OpenAI-compatible client and backend metadata without credentials."""
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider: {provider} (choose openai/ark/openrouter)")
    env_name, base_url, _ = PROVIDER_DEFAULTS[provider]
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(f"Set the {env_name} environment variable")
    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    backend = {
        "provider": provider,
        "endpoint": (base_url or "https://api.openai.com/v1") + "/chat/completions",
        "credential_env": env_name,
        "credential_value_recorded": False,
    }
    return client, backend


def default_model(provider: str) -> str:
    if provider == "ark":
        return os.getenv("ARK_MODEL", "doubao-seed-1-6-250615")
    return PROVIDER_DEFAULTS[provider][2] or "gpt-4o-mini"


def chat_with_receipt(
    client: OpenAI,
    backend: dict[str, Any],
    request: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Make one Chat Completions call and return (text, evidence receipt)."""
    started = time.perf_counter()
    response = client.chat.completions.create(**request)
    elapsed = time.perf_counter() - started
    raw = response.model_dump(mode="json", exclude_none=True)
    usage = raw.get("usage") or {}
    receipt = {
        "backend": {**backend, "model": request.get("model")},
        "request": request,
        "response": raw,
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "response_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "elapsed_seconds": round(elapsed, 6),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }
    return (response.choices[0].message.content or ""), receipt


def save_evidence(run: str, receipts: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> Path:
    """Write run receipts to validation/<run>/evidence.json and update latest.json."""
    run_dir = ROOT / "validation" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "experiment": "8-17 premature-completion-dpo",
        "run": run,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "extra": extra or {},
    }
    path = run_dir / "evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = ROOT / "validation" / "latest.json"
    latest.write_text(json.dumps({"run": run, "evidence": str(path.relative_to(ROOT))},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    return path
