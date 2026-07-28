"""Shared pytest configuration.

The tool modules under src/ import each other with bare names (`from config
import config`), so src/ must be on sys.path for any of them to import. Most
test files do their own `sys.path.insert(...)`, but that only works for tests
that live in the repo root -- `src/test_excel_max_rows_all_sheets.py` sits
inside src/ and imported `excel_tools` bare, which failed collection whenever
pytest was invoked from the project root.

Putting src/ on the path here makes every test module importable regardless of
where it lives or where pytest is started from.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
