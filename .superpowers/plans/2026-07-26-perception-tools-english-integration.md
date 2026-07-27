# Perception Tools English Repository Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `AI-Agent-Tools/perception-tools` into an installable, English-only, OpenAI-only Python project with stable CLI and MCP entry points and a deterministic test suite.

**Architecture:** Move the path-dependent `src` modules into the `perception_tools` package and expose console scripts through `pyproject.toml`. Preserve MCP tool names and `ActionResponse`, make specialized integrations lazy, remove alternate LLM routing, and validate active surfaces with static regression tests.

**Tech Stack:** Python 3.11+, FastMCP, Pydantic 2, OpenAI Python SDK, pytest, Ruff, Docker

## Global Constraints

- Canonical path: `AI-Agent-Tools/perception-tools`.
- All active source, tests, configuration, CLI output, and documentation are English.
- Hosted AI-model calls use only `OPENAI_API_KEY` through the official OpenAI client.
- Existing MCP tool names and `ActionResponse` JSON shape remain compatible.
- Default tests make no network calls.
- Missing optional integrations do not prevent package import, CLI discovery, offline demo, or MCP server startup.

---

### Task 1: Establish package metadata and importable structure

**Files:**
- Create: `AI-Agent-Tools/perception-tools/pyproject.toml`
- Move: `AI-Agent-Tools/perception-tools/src/*.py` to `AI-Agent-Tools/perception-tools/perception_tools/*.py`
- Move: `AI-Agent-Tools/perception-tools/cli.py` to `AI-Agent-Tools/perception-tools/perception_tools/cli.py`
- Create: `AI-Agent-Tools/perception-tools/tests/test_package_entrypoints.py`
- Modify: all moved production modules

**Interfaces:**
- Produces: import package `perception_tools`
- Produces: `perception_tools.cli:main`
- Produces: `perception_tools.server:main`
- Preserves: all existing tool function names and `ActionResponse`

- [x] **Step 1: Write failing package and metadata tests**

```python
from importlib import import_module
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_package_imports_without_sys_path_mutation():
    package = import_module("perception_tools")
    assert package.__name__ == "perception_tools"
    import_module("perception_tools.cli")
    import_module("perception_tools.server")


def test_console_entry_points_are_declared():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["perception-tools"] == "perception_tools.cli:main"
    assert scripts["perception-tools-mcp"] == "perception_tools.server:main"
```

- [x] **Step 2: Run the focused tests and verify expected failures**

Run:

```bash
cd AI-Agent-Tools/perception-tools
python -m pytest -q tests/test_package_entrypoints.py
```

Expected: import and metadata failures because the package and `pyproject.toml` do not exist.

- [x] **Step 3: Move production files into the package**

Use `git mv` for tracked files after the project is staged. Rename `src/main.py`
to `perception_tools/server.py` and root `cli.py` to
`perception_tools/cli.py`. Move the three `src/test_*.py` modules to `tests/`
instead of the package.

- [x] **Step 4: Convert internal imports to package-relative imports**

Use exact forms such as:

```python
from .base import ActionResponse, validate_file_path
from .search_tools import search_web, download_file, search_knowledge_base
```

Delete all production `sys.path` mutations.

- [x] **Step 5: Add package metadata**

`pyproject.toml` must include:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "perception-tools"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "mcp>=0.9.0",
  "pydantic>=2.0.0",
  "python-dotenv>=1.0.0",
  "requests>=2.31.0",
  "beautifulsoup4>=4.12.0",
]

[project.scripts]
perception-tools = "perception_tools.cli:main"
perception-tools-mcp = "perception_tools.server:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
target-version = "py311"
line-length = 100
```

- [x] **Step 6: Add callable module entry points**

`perception_tools/server.py` must expose:

```python
def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

`perception_tools/cli.py` must retain a callable `main(argv: list[str] | None = None) -> int`
and use `raise SystemExit(main())` only inside its `__main__` guard.

- [x] **Step 7: Run package tests**

Run:

```bash
python -m pytest -q tests/test_package_entrypoints.py
python -m build
```

Expected: tests pass and both wheel and source distribution build.

- [x] **Step 8: Commit the package boundary**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "refactor: package perception tools"
```

---

### Task 2: Convert the CLI and active surfaces to English

**Files:**
- Modify: `AI-Agent-Tools/perception-tools/perception_tools/cli.py`
- Modify: `AI-Agent-Tools/perception-tools/README.md`
- Modify: supporting Markdown and configuration files
- Create: `AI-Agent-Tools/perception-tools/tests/test_english_only.py`
- Create: `AI-Agent-Tools/perception-tools/tests/test_cli.py`

**Interfaces:**
- Consumes: packaged tool registry
- Produces: English `list`, `info`, `run`, and `demo` commands
- Produces: static English-only audit

- [x] **Step 1: Write failing English-only and CLI tests**

```python
from pathlib import Path
import re

from perception_tools.cli import main


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


def test_cli_help_is_english(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "Perception Tools MCP Server" in output
    assert "Search" in output
```

- [x] **Step 2: Run focused tests and verify failures**

Run:

```bash
python -m pytest -q tests/test_english_only.py tests/test_cli.py
```

Expected: Han-character and English-help assertions fail.

- [x] **Step 3: Translate the registry and CLI**

Translate category labels, all 53 tool descriptions, dependency notes, help
text, errors, demo fixtures, comments, docstrings, and console output. Preserve
command names, tool names, parameter names, and JSON field names.

- [x] **Step 4: Make CLI exit behavior testable**

Parse passed arguments:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)
```

Do not call `sys.exit` outside the `__main__` guard.

- [x] **Step 5: Run CLI smoke checks**

Run:

```bash
python -m perception_tools.cli --help
python -m perception_tools.cli list
python -m perception_tools.cli info weather
python -m perception_tools.cli demo --offline
```

Expected: every command exits 0 and prints English output.

- [x] **Step 6: Run English-only tests**

Run:

```bash
python -m pytest -q tests/test_english_only.py tests/test_cli.py
```

Expected: all tests pass.

- [x] **Step 7: Commit English conversion**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "refactor: make perception tools English-only"
```

---

### Task 3: Enforce official OpenAI-only model clients

**Files:**
- Modify: `AI-Agent-Tools/perception-tools/perception_tools/media_processing_tools.py`
- Modify: `AI-Agent-Tools/perception-tools/env.example`
- Modify: `AI-Agent-Tools/perception-tools/README.md`
- Create: `AI-Agent-Tools/perception-tools/tests/test_openai_only.py`

**Interfaces:**
- Consumes: `OPENAI_API_KEY`, optional `PERCEPTION_VISION_MODEL`
- Produces: `_make_vision_client(default_model: str) -> tuple[OpenAI, str]`
- Removes: OpenRouter and custom LLM base URL behavior

- [x] **Step 1: Write failing OpenAI-only tests**

```python
from unittest.mock import patch
import pytest

from perception_tools.media_processing_tools import _make_vision_client


def test_vision_client_uses_only_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PERCEPTION_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ignored")
    with patch("openai.OpenAI") as client:
        _, model = _make_vision_client()
    client.assert_called_once_with(api_key="test-key")
    assert model == "gpt-5.6-luna"


def test_missing_openai_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _make_vision_client()
```

Add a static audit asserting `OPENROUTER`, `openrouter`, and
`OPENAI_BASE_URL` are absent from active source, docs, and `env.example`.

- [x] **Step 2: Run tests and verify expected failures**

Run:

```bash
python -m pytest -q tests/test_openai_only.py
```

Expected: current code routes GPT-5 through OpenRouter and accepts alternate keys/base URLs.

- [x] **Step 3: Implement the minimal official-client boundary**

```python
def _make_vision_client(default_model: str = "gpt-5.6-luna"):
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set the OPENAI_API_KEY environment variable."
        )
    model = os.getenv("PERCEPTION_VISION_MODEL", default_model)
    return OpenAI(api_key=api_key), model
```

Delete `_map_model_for_openrouter` and all alternate routing.

- [x] **Step 4: Update configuration surfaces**

`env.example` documents only:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
PERCEPTION_VISION_MODEL=gpt-5.6-luna
```

Retain unrelated Google Calendar and Notion configuration.

- [x] **Step 5: Run OpenAI-only tests**

Run:

```bash
python -m pytest -q tests/test_openai_only.py
```

Expected: all tests pass without network calls.

- [x] **Step 6: Commit provider cleanup**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "refactor: use OpenAI-only perception models"
```

---

### Task 4: Make optional integrations lazy and server startup resilient

**Files:**
- Modify: package tool modules with eager optional imports
- Modify: `AI-Agent-Tools/perception-tools/perception_tools/server.py`
- Modify: `AI-Agent-Tools/perception-tools/pyproject.toml`
- Modify: `AI-Agent-Tools/perception-tools/requirements.txt`
- Create: `AI-Agent-Tools/perception-tools/tests/test_optional_dependencies.py`
- Create: `AI-Agent-Tools/perception-tools/tests/test_server.py`

**Interfaces:**
- Produces: package import and server construction without private credentials
- Produces: structured missing-dependency failures at invocation time
- Preserves: registered MCP tool names

- [x] **Step 1: Write failing optional-dependency tests**

Use `monkeypatch` on `builtins.__import__` or `sys.modules` to simulate absent
`wikipedia`, `arxiv`, `yfinance`, `cv2`, Google, and Notion packages. Assert:

```python
def test_package_import_survives_missing_wikipedia(block_import):
    block_import("wikipedia")
    import perception_tools
    from perception_tools.server import build_server
    assert build_server() is not None
```

For an invoked unavailable tool, parse the returned `TextContent.text` and
assert `success` is false and the message names the missing package.

- [x] **Step 2: Run focused tests and verify failures**

Run:

```bash
python -m pytest -q tests/test_optional_dependencies.py tests/test_server.py
```

Expected: eager imports fail.

- [x] **Step 3: Move specialized imports into tool functions or guarded helpers**

Use:

```python
def _require_dependency(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required. Install {install_hint}."
        ) from exc
```

Convert expected missing-dependency errors into the existing structured
`ActionResponse`.

- [x] **Step 4: Separate core and optional dependency declarations**

Keep server/CLI core dependencies in `[project.dependencies]`; define extras
such as:

```toml
[project.optional-dependencies]
documents = ["PyPDF2>=3", "python-docx>=1.1", "python-pptx>=0.6.23"]
media = ["opencv-python-headless>=4.8", "Pillow>=10", "yt-dlp>=2023.0.0"]
data = ["wikipedia>=1.4", "arxiv>=2", "yfinance>=0.2", "pandas>=2"]
private = ["notion-client>=2", "google-api-python-client>=2"]
all = ["perception-tools[documents,media,data,private]"]
```

Keep `requirements.txt` as the full installation list used by Docker.

- [x] **Step 5: Verify server construction and tool registry**

Tests must assert the expected count derived from the registry rather than a
duplicated magic number, and verify representative tool names from each
category.

- [x] **Step 6: Run focused tests**

Run:

```bash
python -m pytest -q tests/test_optional_dependencies.py tests/test_server.py
```

Expected: all tests pass.

- [x] **Step 7: Commit lazy integrations**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "fix: isolate optional perception integrations"
```

---

### Task 5: Rebuild the test suite as deterministic pytest tests

**Files:**
- Move/modify: all root and former `src/test_*.py` modules into `tests/`
- Create: `AI-Agent-Tools/perception-tools/tests/conftest.py`
- Modify: production code only for confirmed regression failures

**Interfaces:**
- Produces: default offline pytest suite
- Produces: explicit `live` marker gated by `RUN_LIVE_PERCEPTION_TESTS=1`

- [x] **Step 1: Classify every existing test module**

For each file, record whether it is deterministic, requires an optional local
binary, or requires network/API access. Preserve regression assertions for CSV,
grep, negative lengths, page ranges, media cleanup, keyframes, wiki dates,
YouTube, PubChem, and Yahoo Finance.

- [x] **Step 2: Replace executable smoke scripts with pytest functions**

`test_imports.py` must become assertions and must not print or call `sys.exit`.
Network demos such as `test_new_tools.py` become either mocked unit tests or
explicitly marked live tests.

- [x] **Step 3: Add live-test gating**

In `tests/conftest.py`:

```python
def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_LIVE_PERCEPTION_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_LIVE_PERCEPTION_TESTS=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
```

- [x] **Step 4: Run tests and diagnose failures one at a time**

Run:

```bash
python -m pytest -q
```

For each product defect, add or retain a failing regression test before editing
production code. Do not weaken assertions or silently catch exceptions.

- [x] **Step 5: Verify no test mutates import paths or exits during collection**

Run:

```bash
rg -n "sys\\.path|sys\\.exit" tests perception_tools
python -m pytest --collect-only -q
```

Expected: only the CLI `__main__` guard may raise `SystemExit`; collection exits 0.

- [x] **Step 6: Commit deterministic tests and repairs**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "test: make perception suite deterministic"
```

---

### Task 6: Align Docker, documentation, and repository usage

**Files:**
- Modify: `AI-Agent-Tools/perception-tools/Dockerfile`
- Modify: all project Markdown documents
- Modify: `AI-Agent-Tools/perception-tools/env.example`
- Remove or consolidate: obsolete summary/change/index documents where duplicated
- Create/modify: documentation static tests

**Interfaces:**
- Produces: accurate fresh-checkout install, CLI, MCP, Docker, and test instructions
- Produces: container command `perception-tools-mcp`

- [x] **Step 1: Add failing documentation consistency tests**

Assert active docs contain the canonical path and console commands, and exclude:

```python
RETIRED_TEXT = (
    "chapter4/perception-tools",
    "18 tools",
    "OpenWeather API",
    "OPENROUTER_API_KEY",
    "Chinese help",
)
```

- [x] **Step 2: Replace Dockerfile with an official Python base**

Use:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 1000 mcpuser
USER mcpuser
ENTRYPOINT ["perception-tools-mcp"]
```

Add only required OS libraries for enabled full-install features and use
English OCR language data only.

- [x] **Step 3: Consolidate documentation around implemented behavior**

README must cover:

- canonical repository path;
- core and full install;
- English CLI examples;
- MCP client JSON using `perception-tools-mcp`;
- `OPENAI_API_KEY` and `PERCEPTION_VISION_MODEL`;
- optional dependency groups and private credentials;
- Docker commands;
- offline tests and gated live tests;
- actual registry-derived tool count.

Update or remove stale supporting documents so they do not contradict README.

- [x] **Step 4: Run documentation and English audits**

Run:

```bash
python -m pytest -q tests/test_english_only.py tests/test_documentation.py
rg -n --pcre2 '[\\p{Han}]' .
```

Expected: tests pass and `rg` returns no matches in active project files.

- [x] **Step 5: Build and inspect Docker image if Docker is available**

Run:

```bash
docker build -t perception-tools:test .
docker run --rm perception-tools:test
```

For stdio-server smoke validation, use a bounded process or MCP client probe;
do not leave a blocking container running.

- [x] **Step 6: Commit operational integration**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "docs: integrate perception tools with repository"
```

---

### Task 7: Perform fresh-install and final verification

**Files:**
- Modify only files implicated by verification failures

**Interfaces:**
- Produces: release-ready validation evidence

- [x] **Step 1: Create a clean virtual environment outside the project**

Run:

```bash
python -m venv /tmp/perception-tools-venv
/tmp/perception-tools-venv/bin/python -m pip install --upgrade pip
/tmp/perception-tools-venv/bin/python -m pip install -e \
  AI-Agent-Tools/perception-tools
```

- [x] **Step 2: Run installed entry-point smoke checks**

Run:

```bash
/tmp/perception-tools-venv/bin/perception-tools --help
/tmp/perception-tools-venv/bin/perception-tools list
/tmp/perception-tools-venv/bin/perception-tools info weather
/tmp/perception-tools-venv/bin/perception-tools demo --offline
```

Expected: all commands exit 0 with English output.

- [x] **Step 3: Run the complete project verification**

Run:

```bash
cd AI-Agent-Tools/perception-tools
python -m pytest -q
python -m build
ruff check .
python -m compileall -q perception_tools tests
python -m perception_tools.cli demo --offline
git diff --check
```

Expected: all commands exit 0; live tests are skipped unless explicitly enabled.

- [x] **Step 4: Run final static audits**

Run:

```bash
rg -n --pcre2 '[\\p{Han}]' .
rg -ni "openrouter|OPENROUTER|OPENAI_BASE_URL|chapter4/perception-tools|18 tools" .
rg -n "sys\\.path|sys\\.exit" perception_tools tests
```

Expected: no prohibited language, provider routing, stale paths, counts, or
import-path mutations. `SystemExit` is allowed only in CLI/module guards.

- [x] **Step 5: Review the full diff**

Run:

```bash
git status --short
git diff --stat main...HEAD
git diff --check
```

Confirm no unrelated repository files are included.

- [x] **Step 6: Commit final fixes if needed**

```bash
git add AI-Agent-Tools/perception-tools
git commit -m "fix: finish perception tools integration"
```

Do not create an empty commit.
