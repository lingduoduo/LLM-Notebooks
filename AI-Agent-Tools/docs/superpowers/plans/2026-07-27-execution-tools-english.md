# AI Agent Tools English and OpenAI Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all maintained `AI-Agent-Tools` content English and migrate `execution-tools` to a direct OpenAI-only LLM configuration.

**Architecture:** Add a repository-level regression test that rejects non-English scripts in maintained text, translate every finding, and replace the provider router with one direct OpenAI configuration. Keep existing module boundaries and lazy client creation, then validate both packages, CLI behavior, and configuration behavior.

**Tech Stack:** Python 3, pytest, argparse, pathlib, regular expressions

## Global Constraints

- Translate all maintained human-language content in `AI-Agent-Tools` to English.
- Preserve Python identifiers, imports, CLI command and option names, environment variables, JSON keys, paths, external APIs, configuration semantics, safety checks, and execution behavior.
- Replace Chinese runtime strings with concise technical English.
- Use "execution tools", "workspace", "validation", "approval", "truncation and persistence", and "offline demo" consistently.
- Audit `perception-tools`; modify it only if the repository-wide scan finds non-English maintained content.
- Do not perform unrelated refactoring or packaging changes.
- Remove SiliconFlow, Doubao, Kimi/Moonshot, OpenRouter, Qwen, and Gemini provider/model configuration and references.
- Use direct OpenAI with `OPENAI_API_KEY`, default `MODEL` to `gpt-5.6`, and preserve `MODEL` as an override.

## File Structure

- Create `test_english_only.py`: regression test that scans maintained repository text for non-English scripts.
- Modify `execution-tools/execution_tools.py`: English comments and long-output runtime messages.
- Modify `execution-tools/cli.py`: English module documentation, comments, metadata, help, errors, and offline-demo output.
- Modify `execution-tools/config.py`: direct OpenAI-only configuration.
- Modify `execution-tools/llm_helper.py`: direct OpenAI client setup and OpenAI-only documentation.
- Modify `execution-tools/test_config_env.py`: OpenAI configuration regression coverage.
- Modify `execution-tools/env.example`, `execution-tools/README.md`, and `execution-tools/EXPERIMENT.md`: OpenAI-only setup instructions.

---

### Task 1: Add the Repository-Wide English-Only Regression Test

**Files:**
- Create: `test_english_only.py`

**Interfaces:**
- Consumes: the `AI-Agent-Tools` directory containing maintained source, tests, examples, documentation, and configuration.
- Produces: `test_maintained_text_is_english_only()`, which fails with a sorted list of `path:line:text` findings when Han characters are present.

- [ ] **Step 1: Write the failing test**

```python
"""Regression test ensuring maintained project text remains English-only."""

from pathlib import Path
import re


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
}
NON_ENGLISH_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\uf900-\ufaff]"
)


def _maintained_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
            yield path


def test_maintained_text_is_english_only():
    findings = []
    for path in _maintained_text_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if NON_ENGLISH_SCRIPT.search(line):
                relative = path.relative_to(ROOT)
                findings.append(f"{relative}:{line_number}: {line.strip()}")

    assert not findings, "Non-English script found:\n" + "\n".join(sorted(findings))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest test_english_only.py -v
```

Expected: FAIL listing Chinese text in `execution-tools/cli.py` and
`execution-tools/execution_tools.py`.

- [ ] **Step 3: Confirm scan scope**

Run:

```bash
python -m pytest test_english_only.py -v 2>&1 | rg "cli.py|execution_tools.py"
```

Expected: both filenames appear in the failure output; no generated cache or distribution path appears.

- [ ] **Step 4: Commit the regression test**

```bash
git add test_english_only.py
git commit -m "test: enforce English-only AI agent tools text"
```

---

### Task 2: Translate Core Runtime Text

**Files:**
- Modify: `execution-tools/execution_tools.py`
- Test: `test_english_only.py`
- Test: `execution-tools/test_execution_tools.py`

**Interfaces:**
- Consumes: existing output-truncation functions and their unchanged arguments and result dictionaries.
- Produces: the same structured results, with English guidance and omission messages in long-output text.

- [ ] **Step 1: Capture the current runtime-test baseline**

Run:

```bash
python -m pytest execution-tools/test_execution_tools.py -v
```

Expected: existing runtime tests pass before translation.

- [ ] **Step 2: Translate the long-output text**

In `execution_tools.py`, make these exact semantic replacements while preserving interpolation variables and control flow:

```python
# Long-output handling thresholds (see "Long-output truncation and persistence" in chapter 4).
```

```python
guide = f"[To read the complete output, use the read_file tool on {path}]"
```

```python
middle = f"... [{omitted} lines omitted; complete output saved to {path}] ..."
```

- [ ] **Step 3: Run focused runtime and language tests**

Run:

```bash
python -m pytest execution-tools/test_execution_tools.py test_english_only.py -v
```

Expected: runtime tests PASS; English-only test still FAILS only because
`execution-tools/cli.py` contains non-English text.

- [ ] **Step 4: Verify the change is text-only**

Run:

```bash
git diff --word-diff -- execution-tools/execution_tools.py
```

Expected: only comments and string literals changed; function signatures, branches, result keys, and calls are unchanged.

- [ ] **Step 5: Commit the runtime translation**

```bash
git add execution-tools/execution_tools.py
git commit -m "feat: translate execution runtime messages"
```

---

### Task 3: Translate CLI Documentation and Visible Output

**Files:**
- Modify: `execution-tools/cli.py`
- Test: `test_english_only.py`
- Test: `execution-tools/test_cli.py`

**Interfaces:**
- Consumes: existing CLI parser, subcommand handlers, tool constructors, and result dictionaries.
- Produces: the same `list`, `demo`, `code`, `shell`, `write`, `edit`, `calendar`, and `pr` commands with English help, errors, headings, and demonstration output.

- [ ] **Step 1: Capture the current CLI-test baseline**

Run:

```bash
python -m pytest execution-tools/test_cli.py -v
```

Expected: existing CLI tests pass before translation.

- [ ] **Step 2: Translate module documentation and developer comments**

Translate the module docstring, function docstrings, section comments, inline comments, and references to book sections. Preserve code examples, command names, option names, filenames, environment variables, provider names, and function identifiers verbatim.

Use these English terms consistently:

```text
execution tools
file system
general execution
external system
workspace
validation
approval
long-output truncation and persistence
offline demo
```

- [ ] **Step 3: Translate metadata, errors, and list output**

Translate every human-readable value in `TOOLS`, `_print_json` documentation, list headings, guidance lines, and missing-input errors. Preserve tuple ordering, tool names, JSON rendering, exit codes, and `stderr` routing.

The list heading and columns should read naturally as:

```python
print("Available execution tools:\n")
print(f"  {'Tool':<20} {'Category':<16} Description")
```

The guidance should retain the existing commands:

```python
print("\nUse `python cli.py <subcommand> --help` to view each tool's arguments.")
print("Use `python cli.py demo` to run the end-to-end offline demo.")
```

- [ ] **Step 4: Translate the offline demo**

Translate all demo section titles, generated sample prose, status labels, explanatory messages, and comments. Keep the seven existing stages, temporary-workspace isolation, tool inputs, safety settings, sample counts, structured result access, and cleanup behavior unchanged.

Retain the semantic sequence:

```text
1. Write the word-frequency script and validate it
2. Reject invalid Python syntax
3. Generate sample data
4. Execute the statistics code
5. Verify the data with the shell
6. Truncate and persist long output
7. Require approval for a dangerous command
```

- [ ] **Step 5: Translate parser help and interruption output**

Translate `ArgumentParser` description and epilog, global-option help, subparser metavar, every subcommand help string, every argument help string, and the `KeyboardInterrupt` message. Preserve all parser destinations, defaults, required flags, choices, and dispatch behavior.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m pytest execution-tools/test_cli.py test_english_only.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Inspect all CLI surfaces**

Run:

```bash
python execution-tools/cli.py --help
python execution-tools/cli.py list
python execution-tools/cli.py code --help
python execution-tools/cli.py shell --help
python execution-tools/cli.py write --help
python execution-tools/cli.py edit --help
python execution-tools/cli.py calendar --help
python execution-tools/cli.py pr --help
```

Expected: all descriptions, labels, errors, and help are English; command and option names are unchanged.

- [ ] **Step 8: Run the offline demo**

Run:

```bash
python execution-tools/cli.py --no-approval --no-summarize demo
```

Expected: exit code 0, all visible prose is English, the demo reaches "Demo complete", and it reports the temporary artifact location.

- [ ] **Step 9: Commit the CLI translation**

```bash
git add execution-tools/cli.py
git commit -m "feat: translate execution tools CLI"
```

---

### Task 4: Migrate the LLM Integration to Direct OpenAI

**Files:**
- Modify: `execution-tools/config.py`
- Modify: `execution-tools/llm_helper.py`
- Modify: `execution-tools/cli.py`
- Modify: `execution-tools/test_config_env.py`
- Modify: `execution-tools/env.example`
- Modify: `execution-tools/README.md`
- Modify: `execution-tools/EXPERIMENT.md`

**Interfaces:**
- Consumes: `OPENAI_API_KEY`, optional `MODEL`, existing numeric model parameters, and lazy `LLMHelper` initialization.
- Produces: `Config.get_llm_config() -> dict` containing exactly `provider`, `api_key`, and `model`, with `provider == "openai"` and default model `gpt-5.6`.

- [ ] **Step 1: Add failing OpenAI configuration tests**

Append these tests to `execution-tools/test_config_env.py`:

```python
import pytest


def test_openai_config_uses_direct_api_and_default_model(monkeypatch):
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(cfg.Config, "MODEL", "gpt-5.6")

    assert cfg.Config.get_llm_config() == {
        "provider": "openai",
        "api_key": "test-openai-key",
        "model": "gpt-5.6",
    }


def test_openai_model_can_be_overridden(monkeypatch):
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(cfg.Config, "MODEL", "gpt-5.6-terra")

    assert cfg.Config.get_llm_config()["model"] == "gpt-5.6-terra"


def test_openai_key_is_required(monkeypatch):
    monkeypatch.setattr(cfg.Config, "OPENAI_API_KEY", None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        cfg.Config.get_llm_config()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest execution-tools/test_config_env.py -v
```

Expected: FAIL because `Config.OPENAI_API_KEY` and the direct OpenAI result do not exist.

- [ ] **Step 3: Replace provider routing with direct OpenAI configuration**

In `execution-tools/config.py`:

1. Delete `PROVIDER`, all provider-specific key attributes, `get_api_key()`,
   and `effective_provider()`.
2. Define:

```python
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
MODEL: str = os.getenv("MODEL", "gpt-5.6")
```

3. Make `validate()` raise:

```python
raise ValueError("OPENAI_API_KEY is required for LLM operations.")
```

when the key is absent.
4. Implement:

```python
@classmethod
def get_llm_config(cls) -> dict:
    """Return the direct OpenAI LLM configuration."""
    cls.validate()
    return {
        "provider": "openai",
        "api_key": cls.OPENAI_API_KEY,
        "model": cls.MODEL,
    }
```

- [ ] **Step 4: Simplify direct OpenAI client creation**

In `execution-tools/llm_helper.py`, keep lazy initialization but construct the client without a third-party base URL:

```python
self.client = OpenAI(api_key=llm_config["api_key"])
self.model = llm_config["model"]
self.provider = llm_config["provider"]
```

Rewrite `_reasoning_safe_temperature()` documentation to discuss GPT-5 only,
and reduce its condition to:

```python
return 1 if "gpt-5" in m else requested
```

Rewrite `_parse_json_response()` documentation generically: fenced JSON is
accepted because model responses may include Markdown fences. Do not change
its parsing behavior.

- [ ] **Step 5: Remove the obsolete CLI provider option**

In `execution-tools/cli.py`, delete the `args.provider` environment override
and the `--provider` argument. Keep `--workspace`, `--no-approval`,
`--no-verify`, and `--no-summarize` unchanged.

- [ ] **Step 6: Update example configuration**

Replace the LLM section of `execution-tools/env.example` with:

```dotenv
# Direct OpenAI configuration for safety checks and summarization
OPENAI_API_KEY=your_openai_api_key

# Optional model override; defaults to gpt-5.6
# MODEL=gpt-5.6

# Model parameters
TEMPERATURE=0.7
MAX_TOKENS=4096
```

Keep unrelated Google Calendar, GitHub, safety, and workspace settings.

- [ ] **Step 7: Update user and experiment documentation**

In `execution-tools/README.md` and `execution-tools/EXPERIMENT.md`:

- Replace provider-selection instructions with `OPENAI_API_KEY`.
- Document `gpt-5.6` as the default and `MODEL` as the optional override.
- Remove provider lists, third-party endpoints, fallback behavior, and
  provider-specific examples.
- Preserve offline-operation documentation and unrelated integrations.

- [ ] **Step 8: Run focused configuration and helper tests**

Run:

```bash
python -m pytest \
  execution-tools/test_config_env.py \
  execution-tools/test_execution_tools.py \
  execution-tools/test_cli.py -v
```

Expected: all tests PASS.

- [ ] **Step 9: Verify obsolete provider configuration is absent**

Run the following from `AI-Agent-Tools`:

```bash
rg -n -i \
  'siliconflow|doubao|kimi|moonshot|openrouter|qwen|gemini|PROVIDER=|--provider' \
  execution-tools \
  --glob '!test_config_env.py' \
  --glob '!**/__pycache__/**' \
  --glob '!**/.pytest_cache/**'
```

Expected: exit code 1 with no matches.

- [ ] **Step 10: Commit the OpenAI migration**

```bash
git add \
  execution-tools/config.py \
  execution-tools/llm_helper.py \
  execution-tools/cli.py \
  execution-tools/test_config_env.py \
  execution-tools/env.example \
  execution-tools/README.md \
  execution-tools/EXPERIMENT.md
git commit -m "feat: use direct OpenAI models"
```

---

### Task 5: Complete Repository Validation

**Files:**
- Verify: `execution-tools/`
- Verify: `perception-tools/`
- Test: `test_english_only.py`

**Interfaces:**
- Consumes: the translated package and English-only regression test.
- Produces: evidence that the conversion is complete and behavior remains compatible.

- [ ] **Step 1: Run a direct Unicode scan**

Run:

```bash
rg -n '[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}\p{Cyrillic}\p{Arabic}]' . \
  --glob '!**/.git/**' \
  --glob '!**/.pytest_cache/**' \
  --glob '!**/.ruff_cache/**' \
  --glob '!**/__pycache__/**' \
  --glob '!**/build/**' \
  --glob '!**/dist/**'
```

Expected: exit code 1 with no matches.

- [ ] **Step 2: Run the full execution-tools suite**

Run:

```bash
python -m pytest test_english_only.py -v
python -m pytest execution-tools -v
python -m pytest perception-tools/tests -v
```

Expected: both packages' non-live tests PASS; only explicitly configured external/live tests may be skipped.

- [ ] **Step 3: Run syntax compilation**

Run:

```bash
python -m compileall -q execution-tools perception-tools
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 4: Check patch quality and scope**

Run:

```bash
cd ..
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; no `perception-tools` modifications; changes are limited to the specification, plan, `execution-tools` translation, and English-only test.

- [ ] **Step 5: Commit any final test-only adjustment**

If validation required a correction to `test_english_only.py`, commit only that correction:

```bash
git add test_english_only.py
git commit -m "test: finalize English-only coverage"
```

If no correction was required, do not create an empty commit.
