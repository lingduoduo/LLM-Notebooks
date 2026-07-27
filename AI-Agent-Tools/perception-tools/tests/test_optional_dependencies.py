import subprocess
import sys


OPTIONAL_MODULES = {
    "arxiv",
    "cv2",
    "docx",
    "pandas",
    "PIL",
    "pptx",
    "PyPDF2",
    "waybackpy",
    "wikipedia",
    "yfinance",
}


def test_server_import_survives_missing_optional_dependencies():
    blocked = repr(sorted(OPTIONAL_MODULES))
    code = f"""
import importlib.abc
import sys

blocked = set({blocked})

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError(f"blocked optional dependency: {{fullname}}")
        return None

sys.meta_path.insert(0, BlockOptional())
from perception_tools.server import mcp
assert mcp is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
