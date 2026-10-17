"""Offline tests: hosted Chinese-model API teachers are rejected; supported teachers still pass."""

import asyncio
import re
import sys
from pathlib import Path

import pytest

import generate_data as gd

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize("base_url", [
    "https://api.moonshot.cn/v1",
    "https://api.deepseek.com/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://ark.cn-beijing.volces.com/api/v3",
    "https://api.siliconflow.cn/v1",
    "https://open.bigmodel.cn/api/paas/v4",
])
def test_removed_endpoints_rejected(base_url):
    with pytest.raises(SystemExit):
        gd.check_teacher_backend("anthropic/claude-opus-4.8", base_url)


@pytest.mark.parametrize("model", [
    "moonshotai/kimi-k2", "kimi-k3", "deepseek/deepseek-r1", "deepseek-reasoner",
    "qwen/qwen3-235b-a22b", "z-ai/glm-4.6", "minimax/minimax-m1", "doubao-seed-1-6-250615",
])
def test_removed_models_rejected_via_openrouter(model):
    with pytest.raises(SystemExit):
        gd.check_teacher_backend(model, "https://openrouter.ai/api/v1")


@pytest.mark.parametrize("model,base_url", [
    ("anthropic/claude-opus-4.8", "https://openrouter.ai/api/v1"),
    ("openai/gpt-4o-mini", "https://openrouter.ai/api/v1"),
    ("gpt-4o-mini", "https://api.openai.com/v1"),
])
def test_supported_teachers_allowed(model, base_url):
    gd.check_teacher_backend(model, base_url)


def test_main_rejects_removed_backend_before_any_call(tmp_path, monkeypatch):
    (tmp_path / "p.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("MOONSHOT_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", [
        "generate_data.py", "--input", str(tmp_path / "p.jsonl"),
        "--model", "kimi-k3", "--base_url", "https://api.moonshot.cn/v1",
        "--api_key_env", "MOONSHOT_API_KEY",
        "--raw_output", str(tmp_path / "raw.jsonl"), "--sft_output", str(tmp_path / "sft.jsonl"),
    ])
    with pytest.raises(SystemExit, match="removed"):
        asyncio.run(gd.main())
    assert not (tmp_path / "raw.jsonl").exists()


def test_code_and_docs_do_not_recommend_removed_backends():
    src = (ROOT / "generate_data.py").read_text(encoding="utf-8")
    # Only the blocklist itself may name removed providers.
    body = src.split("REMOVED_MODEL_PREFIXES", 1)[0].split("REMOVED_API_HOSTS", 1)[0]
    body += src.split("def check_teacher_backend", 1)[1]
    assert not re.search(r"moonshot|kimi|deepseek|dashscope|qwen", body, re.I)
    assert "anthropic/claude-opus-4.8" in src
