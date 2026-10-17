"""Offline guard: Qwen teacher/student are local open-weight models (vLLM/transformers),
never a hosted Chinese-model API backend."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
HOSTED = re.compile(
    r"moonshot|dashscope|volces|volcengine|ark_api|siliconflow|stepfun|bigmodel|zhipu|"
    r"api\.deepseek\.com|openrouter|base_url|DEEPSEEK_API_KEY|DASHSCOPE_API_KEY",
    re.I,
)
SOURCES = ["create_data.py", "evaluate.py", "compare.py", "train_sft_trl.py",
           "create_data_h100x8.sh", "train_trl.sh", "README.md", "requirements.txt"]


@pytest.mark.parametrize("name", SOURCES)
def test_no_hosted_chinese_api_backend(name):
    assert not HOSTED.search((ROOT / name).read_text(encoding="utf-8"))


def test_teacher_generation_runs_locally_with_vllm():
    src = (ROOT / "create_data.py").read_text(encoding="utf-8")
    assert "from vllm import LLM, SamplingParams" in src
    assert "import openai" not in src and "from openai" not in src
