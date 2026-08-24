"""Regression checks for English-only maintained text files."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".example"}
HAN_CHARACTER = re.compile(r"[\u4e00-\u9fff]")


def test_maintained_text_is_english_only() -> None:
    occurrences: list[str] = []

    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if HAN_CHARACTER.search(line):
                relative_path = path.relative_to(PROJECT_ROOT)
                occurrences.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert not occurrences, "Chinese text remains:\n" + "\n".join(occurrences)
