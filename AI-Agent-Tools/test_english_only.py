"""Regression test ensuring maintained project text remains English-only.

Scope: every maintained package under AI-Agent-Tools. `perception-tools` also
ships its own narrower check (tests/test_english_only.py) covering just that
package; this one is the repository-wide net and does not replace it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    "Dockerfile",
    "env.example",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".venv",
}

# Written with escape sequences so this file is itself pure ASCII.
NON_ENGLISH_SCRIPT = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    "\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\uf900-\ufaff]"
)


def _maintained_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if EXCLUDED_PARTS & set(path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
            yield path


def _findings():
    findings = []
    for path in _maintained_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if NON_ENGLISH_SCRIPT.search(line):
                relative = path.relative_to(ROOT)
                findings.append(f"{relative}:{number}:{line.strip()}")
    return sorted(findings)


def test_maintained_text_is_english_only():
    findings = _findings()
    assert not findings, "Non-English text found:\n" + "\n".join(findings)
