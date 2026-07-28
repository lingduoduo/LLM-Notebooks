"""Shared pytest fixtures for the execution tools test suite.

Two problems this file exists to solve:

1. ``Config.WORKSPACE_DIR`` defaults to ``os.getcwd()``. Without isolation the
   file and terminal tools write into whatever directory pytest was started
   from -- i.e. the repository itself. The autouse fixture below repoints the
   workspace at a per-test temporary directory.
2. Several test modules used to install a stub ``sys.modules['config']`` at
   import time. Because module objects are cached, whichever test module was
   imported first won the race and the others silently operated against the
   wrong workspace, so results depended on collection order. Patching the real
   ``Config`` here keeps every test order-independent.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def no_live_api_calls(monkeypatch):
    """Keep the suite hermetic.

    config.py calls load_dotenv() at import, so a populated .env reaches the
    tests. Without this, OPENAI_API_KEY makes them issue real billable
    requests, and GITHUB_TOKEN turns the external-tool tests into live
    authenticated calls that try to open a pull request on github.com. Tests
    that need a client install a fake one directly.
    """
    import config

    for variable in ("OPENAI_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(config.Config, variable, None)

    # Point calendar credentials at a path that cannot exist, so no OAuth
    # browser flow can be triggered from a test run.
    monkeypatch.setattr(
        config.Config, "GOOGLE_CALENDAR_CREDENTIALS_FILE", "/nonexistent/credentials.json"
    )


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    """Point every tool's workspace at a throwaway directory."""
    import config

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(config.Config, "WORKSPACE_DIR", workspace)
    monkeypatch.chdir(workspace)
    return workspace


@pytest.fixture
def workspace(isolated_workspace) -> Path:
    """Explicit alias for tests that need to reference the workspace path."""
    return isolated_workspace


@pytest.fixture
def offline_safety(monkeypatch):
    """Disable the LLM-backed safety layers for tests that exercise execution.

    Approval and non-Python syntax checks both require an API key; tests that
    are not about those layers turn them off so they run offline.
    """
    import config

    monkeypatch.setattr(config.Config, "REQUIRE_APPROVAL_FOR_DANGEROUS_OPS", False)
    monkeypatch.setattr(config.Config, "AUTO_SUMMARIZE_COMPLEX_OUTPUT", False)
    monkeypatch.setattr(config.Config, "AUTO_VERIFY_CODE", True)
