# User Memory: Remove Doubao Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `AI-Agent-KnowledgeBase/user-memory` has no Doubao provider; OpenAI is the way to run what Doubao ran.

**Spec:** `.superpowers/specs/2026-10-17-user-memory-remove-doubao-design.md`

**Tech Stack:** Python 3, `openai` SDK, argparse, pytest

## Global Constraints

- Leave SiliconFlow, OpenRouter and the OpenRouter fallback untouched.
- No network in tests. Provider validation happens before any client call.

---

### Task 1: Failing tests

**Files:** Create `AI-Agent-KnowledgeBase/user-memory/test_no_doubao_provider.py`

- [x] Config has no Doubao attributes or default model; `get_api_key("doubao")` is `None`; `validate("doubao")` is `False` and prints no `DOUBAO` hint.
- [x] `UserMemoryAgent` and `ConversationalAgent` raise `ValueError("Unsupported provider: doubao…")`.
- [x] `memory_cli.build_parser()` exits on `extract --provider doubao`.
- [x] OpenAI key resolution still works.
- [x] Run against the previous code: 5 failed, 1 passed (the OpenAI check).

### Task 2: Remove the provider

**Files:** Modify `config.py`, `agent.py`, `conversational_agent.py`, `main.py`, `memory_cli.py`, `background_memory_processor.py` and `env.example`

- [x] Delete each Doubao branch, attribute and choice listed in the spec, using exact-match edits that assert a single occurrence.
- [x] Update the unsupported-provider messages and docstrings to `'openai', 'siliconflow', 'openrouter'`.

### Task 3: Verify

- [x] `python -m pytest -q` in `user-memory/`: 13 passed (7 existing + 6 new).
- [x] `grep -rniI "doubao|volces"` finds nothing outside the new test.
- [x] `python -m py_compile` on the six edited modules.
