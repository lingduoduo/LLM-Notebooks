# JevBench Snapshot OpenAI Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The snapshot scripts target Jev plus the OpenAI lineup, and the old-model figures are removed.

**Spec:** `.superpowers/specs/2026-10-17-jev-snapshot-openai-template-design.md`

**Tech Stack:** Python 3 standard library (zipfile, argparse, SVG strings), pytest

## Global Constraints

- Never write invented scores; tests use synthetic archives under `tmp_path` only.
- Five-model output must stay geometrically identical to the old renderer.

---

### Task 1: Tests first

**Files:** Create `Jev-Evaluation/benchmark/tests/test_snapshot_scripts.py`

- [x] Lineup is `["jev", *OPENAI_MODELS]` with no old provider ids.
- [x] `prepare` writes `figures/jevbench-<date>/snapshot.json` with matching `snapshot_date` and `models_order`, and skips models missing from the archive with a `skip <id>` line.
- [x] `render` with six models keeps every `<rect>` and `<text>` inside the canvas; the dashboard shows `6-provider snapshot` and the formatted date.
- [x] `row_positions` and `column_width` reproduce the five-model geometry.
- [x] Confirm all fail against the previous scripts: 5 failed.

### Task 2: `prepare_jevbench_snapshot.py`

- [x] Import `OPENAI_MODELS` from `benchmark/llm_eval/adapters` and build `MODEL_IDS` / `DISPLAY_NAMES` from it.
- [x] Add `--date` (default today) and derive the output path from it.
- [x] Skip models missing from the archive; exit if none are present; `models_order` lists only the present ones.

### Task 3: `render_jevbench_snapshot.py`

- [x] Add `--snapshot` (default: newest `figures/jevbench-*/snapshot.json`) and write the SVGs next to it.
- [x] Take the dashboard title count and date from the snapshot.
- [x] Add `row_positions` / `column_width`; replace `range(5)`, `top + i * gap` and `gap * 5` geometry in the dashboard, composite, split-coverage, family-yield and failure charts.
- [x] Re-key `MODEL_COLORS` for the new lineup.

### Task 4: Remove the old figures

- [x] `git rm -r Jev-Evaluation/figures/jevbench-2026-09-26`.

### Task 5: Verify

- [x] `pytest Jev-Evaluation/benchmark/tests`: 5 passed.
- [x] Re-render the archived five-model snapshot into a scratch dir: four SVGs byte-identical, the dashboard differs only in the title wording.
- [x] Render a six-model scratch snapshot: no element outside its canvas.
- [x] No old provider names remain in `Jev-Evaluation` outside recorded `runs/` data.
