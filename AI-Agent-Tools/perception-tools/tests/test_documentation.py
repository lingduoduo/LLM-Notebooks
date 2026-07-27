"""Regression checks for package-facing documentation and launchers."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "SETUP.md",
    "QUICK_START.md",
    "ARCHITECTURE.md",
    "TOOL_REFERENCE.md",
    "INDEX.md",
    "PROJECT_SUMMARY.md",
    "CHANGES.md",
]


def test_docs_use_current_package_layout_and_tool_count():
    content = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in DOCS)

    assert "chapter4/perception-tools" not in content
    assert "src/main.py" not in content
    assert "18 tools" not in content
    assert "OpenWeather" not in content
    assert ("Open" + "Router") not in content
    assert "53" in content


def test_dockerfile_runs_installed_mcp_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["perception-tools-mcp"]' in dockerfile
    assert "python:3.12-slim" in dockerfile
    assert "src/main.py" not in dockerfile
