from pathlib import Path
from unittest.mock import patch

import pytest

from perception_tools.media_processing_tools import _make_vision_client


ROOT = Path(__file__).parents[1]


def test_vision_client_uses_only_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PERCEPTION_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ignored")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ignored.example/v1")

    with patch("openai.OpenAI") as client:
        _, model = _make_vision_client()

    client.assert_called_once_with(api_key="test-key")
    assert model == "gpt-5.6-luna"


def test_missing_openai_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _make_vision_client()


def test_active_surfaces_have_no_alternate_llm_routing():
    paths = [
        *ROOT.glob("perception_tools/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("*.md"),
        ROOT / "env.example",
    ]
    removed_terms = ("openrouter", "openai_base_url")
    for path in paths:
        if path.is_file() and path.name != Path(__file__).name:
            content = path.read_text(encoding="utf-8").lower()
            for term in removed_terms:
                assert term not in content, f"{term!r} remains in {path}"
