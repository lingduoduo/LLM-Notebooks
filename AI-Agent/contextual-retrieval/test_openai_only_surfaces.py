from pathlib import Path


ROOT = Path(__file__).parent
ACTIVE_SURFACES = [
    ROOT / "config.py",
    ROOT / "agent.py",
    ROOT / "contextual_chunking.py",
    ROOT / "main.py",
    ROOT / "index_local_laws_contextual.py",
    ROOT / "compare_retrieval.py",
    ROOT / "test_simple.py",
    ROOT / "env.example",
    ROOT / "README.md",
    ROOT / "README_LEGAL_INDEXING.md",
    ROOT / "evaluation" / "evaluate.py",
]
REMOVED_TERMS = (
    "kimi",
    "moonshot",
    "doubao",
    "siliconflow",
    "openrouter",
    "groq",
    "together ai",
    "deepseek",
    "llm_provider",
)


def test_active_surfaces_are_openai_only():
    for path in ACTIVE_SURFACES:
        content = path.read_text(encoding="utf-8").lower()
        for term in REMOVED_TERMS:
            assert term not in content, f"{term!r} remains in {path.relative_to(ROOT)}"


def test_environment_template_uses_openai_defaults():
    content = (ROOT / "env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=your_openai_api_key_here" in content
    assert "LLM_MODEL=gpt-5.6-terra" in content
    assert "LLM_PROVIDER=" not in content
