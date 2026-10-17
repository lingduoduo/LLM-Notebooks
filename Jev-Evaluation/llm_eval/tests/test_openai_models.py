"""OpenAI provider presets, per-model pricing, and reasoning-model requests.

No network: http_post_json is replaced with a stub that records the body.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_eval.adapters import (  # noqa: E402
    OPENAI_MODELS,
    PROVIDERS,
    REASONING_MIN_OUTPUT_TOKENS,
    effective_max_output_tokens,
    get_adapter,
    resolve_prices,
)
from llm_eval.adapters import openai_compat  # noqa: E402
from llm_eval.adapters.openai_compat import is_reasoning_model  # noqa: E402
from llm_eval.task import load_tasks  # noqa: E402


def test_only_jev_and_openai_presets_remain():
    assert set(PROVIDERS) == {"jev", "typesafe", "openai"}
    assert PROVIDERS["openai"]["base_url"] == "https://api.openai.com/v1"
    assert PROVIDERS["openai"]["default_model"] in OPENAI_MODELS


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-6.1-sol", True),
        ("gpt-6-luna", True),
        ("gpt-5.6-terra", True),
        ("o4-mini", True),
        ("openai/gpt-5.6-luna", True),
        ("gpt-4.1-mini", False),
        ("gpt-4o", False),
        ("", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


def test_effective_max_output_tokens_floors_reasoning_models_only():
    assert effective_max_output_tokens("gpt-6-luna", 200) == REASONING_MIN_OUTPUT_TOKENS
    assert effective_max_output_tokens("gpt-6-luna", 4096) == 4096
    assert effective_max_output_tokens("gpt-4.1-mini", 200) == 200


@pytest.mark.parametrize("model", sorted(OPENAI_MODELS))
def test_resolve_prices_uses_the_models_own_price(model):
    expected = OPENAI_MODELS[model]
    assert resolve_prices("openai", None, None, model=model) == (
        expected["price_in_per_m"],
        expected["price_out_per_m"],
    )


def test_resolve_prices_defaults_and_overrides():
    preset = PROVIDERS["openai"]
    # No model -> the preset's default model price.
    assert resolve_prices("openai", None, None) == (
        preset["price_in_per_m"],
        preset["price_out_per_m"],
    )
    # Unknown model -> preset price; explicit prices always win.
    assert resolve_prices("openai", None, None, model="gpt-9") == (
        preset["price_in_per_m"],
        preset["price_out_per_m"],
    )
    assert resolve_prices("openai", 1.0, 2.0, model="gpt-6.1-sol") == (1.0, 2.0)


@pytest.fixture
def captured(monkeypatch):
    bodies = []

    def fake_post(url, body, headers, timeout_s=120.0):
        bodies.append(body)
        labels = json.loads(body["messages"][-1]["content"].split("\n", 1)[1])["labels"]
        content = json.dumps({"probs": {l: 1 / len(labels) for l in labels}})
        return 200, {"choices": [{"message": {"content": content}}], "usage": {}}, 0.01

    monkeypatch.setattr(openai_compat, "http_post_json", fake_post)
    return bodies


@pytest.fixture
def task():
    return load_tasks(str(ROOT / "tasks" / "smoke.jsonl"))[0]


def test_reasoning_model_request_shape(captured, task):
    get_adapter(name="openai", model="gpt-6.1-sol", key="k").call(task)
    body = captured[0]
    assert body["max_completion_tokens"] == REASONING_MIN_OUTPUT_TOKENS
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body
    assert "max_tokens" not in body


def test_non_reasoning_model_request_shape(captured, task):
    get_adapter(name="openai", model="gpt-4.1-mini", key="k").call(task)
    body = captured[0]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 200
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


def test_request_options_none_removes_reasoning_effort(captured, task):
    get_adapter(
        name="openai",
        model="gpt-6-luna",
        key="k",
        request_options={"reasoning_effort": None},
    ).call(task)
    assert "reasoning_effort" not in captured[0]


def test_default_openai_adapter_uses_preset_model(captured, task):
    result = get_adapter(name="openai", key="k").call(task)
    assert captured[0]["model"] == PROVIDERS["openai"]["default_model"]
    assert set(result["probs"]) == set(task.labels)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
