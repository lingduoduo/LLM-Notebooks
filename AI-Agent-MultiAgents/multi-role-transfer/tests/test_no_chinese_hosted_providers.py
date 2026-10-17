"""Offline checks that hosted Chinese-model backends are gone and OpenAI is the default."""

import re
from pathlib import Path

import package_comparison
import run_official_experiment


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = sorted(p.name for p in ROOT.glob("*.py") if not p.name.startswith("test_")) + ["env.example"]
BANNED = re.compile(
    r"doubao|volces|volcengine|ark_api|moonshot|kimi|deepseek|dashscope|qwen|siliconflow|"
    r"stepfun|minimax|bigmodel|zhipu",
    re.IGNORECASE,
)
# Paths into retained historical bundles are recorded-data locators, not backends.
RECORDED_RUN_PATH = re.compile(r"exp10-1-(kimi-k2\.5-tavily[\w.-]*|qwen35flash[\w.-]*)")


def test_runtime_sources_have_no_chinese_hosted_provider_references():
    for name in RUNTIME_FILES:
        text = RECORDED_RUN_PATH.sub("", (ROOT / name).read_text(encoding="utf-8"))
        hits = sorted(set(m.group(0).lower() for m in BANNED.finditer(text)))
        assert not hits, f"{name} still references {hits}"


def test_moonshot_provider_label_removed_and_openai_is_default():
    assert "api.moonshot.cn" not in run_official_experiment.PROVIDER_LABELS
    assert run_official_experiment.DEFAULT_BASE_URL == "https://api.openai.com/v1"
    assert run_official_experiment.provider_label(run_official_experiment.DEFAULT_BASE_URL) == "openai"


def test_moonshot_key_not_in_credential_scans():
    assert "MOONSHOT_API_KEY" not in run_official_experiment.SECRET_ENV_NAMES
    assert "MOONSHOT_API_KEY" not in package_comparison.SECRET_ENV_NAMES
    assert "OPENAI_API_KEY" in run_official_experiment.SECRET_ENV_NAMES


def test_live_entry_points_default_to_openai():
    for name in ("demo.py", "run_comparison.py", "judge_comparison.py", "run_official_experiment.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "https://api.openai.com/v1" in text, name
    env = (ROOT / "env.example").read_text(encoding="utf-8")
    assert "OPENAI_BASE_URL=https://api.openai.com/v1" in env
    assert "OPENAI_API_KEY=" in env
