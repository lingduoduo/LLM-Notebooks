# TTS Quality Eval: Remove Doubao Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tts-quality-eval` offers OpenAI, ElevenLabs, Fish Audio and Minimax, with consistent provider tables and no Doubao.

**Spec:** `.superpowers/specs/2026-10-17-tts-remove-doubao-design.md`

**Tech Stack:** Python 3, pytest

## Global Constraints

- No network in tests; `synthesize()` must reject an unknown provider before calling out.
- Keep the user's original `config.py` edit (PROVIDERS entry removal) as the starting point.

---

### Task 1: Tests first

**Files:** Create `AI-Agent-Evaluation/tts-quality-eval/test_provider_registry.py`

- [x] Consistency: each `PROVIDER_CONFIGS[name].provider == name` and is in both `PROVIDERS` and `_SYNTH_DISPATCH`; the three key sets are equal.
- [x] Doubao absent from all tables and `_synth_doubao` gone; `select_configs("doubao")` raises; `synthesize()` raises `Unknown provider: 'doubao'`.
- [x] `supports_speed` for openai tts-1 / gpt-4o-mini-tts, minimax, elevenlabs and doubao.
- [x] Confirm failures: 6 on the half-removed local state, 4 on `origin/main`.

### Task 2: Remove Doubao

- [x] `config.py`: drop the `PROVIDER_CONFIGS["doubao"]` preset; `supports_speed` → `self.provider == "minimax"` for non-OpenAI; update the docstring and comment.
- [x] `pipeline.py`: delete `_synth_doubao` and the dispatch entry (keep `import base64`).
- [x] `demo.py`, `README.md`, `env.example`: remove Doubao mentions and variables.

### Task 3: Verify

- [x] `python -m pytest -q` in `tts-quality-eval/`: 37 passed.
- [x] `grep -rniI "doubao|volc|openspeech|bytedance"` finds nothing outside the new test.
