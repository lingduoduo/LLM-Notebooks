from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def test_active_surfaces_are_english_only():
    paths = [
        *ROOT.glob("perception_tools/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("*.md"),
        ROOT / "env.example",
        ROOT / "Dockerfile",
        ROOT / "pyproject.toml",
    ]
    for path in paths:
        if path.is_file():
            assert not HAN.search(path.read_text(encoding="utf-8")), path
