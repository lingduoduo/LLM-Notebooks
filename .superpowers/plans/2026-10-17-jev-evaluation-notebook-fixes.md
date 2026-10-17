# Jev-Evaluation Notebook Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All three Jev-Evaluation notebooks run end to end, apart from their paid cells.

**Spec:** `.superpowers/specs/2026-10-17-jev-evaluation-notebook-fixes-design.md`

**Architecture:** Notebook-cell edits plus committed task fixtures. No library code changes.

**Tech Stack:** Jupyter (`llm` kernel), nbclient, pytest

## Global Constraints

- Do not execute cells that call paid APIs (`run_multi` with real adapters) during verification.
- Edit notebooks as JSON with exact-match replacements, keeping `indent=1` and the trailing newline so diffs stay minimal.
- Task fixtures must be byte-identical to what `scripts/convert_jevbench.py` produces.

---

### Task 1: Project discovery in `01_模型评测.ipynb`

**Files:** Modify `Jev-Evaluation/01_模型评测.ipynb` (cell 1)

- [x] Replace the `laya/benchmark`-only search with the multi-layout search used by `laya_vs_jev.ipynb` cell 1 (`.`, `benchmark`, `Jev-Evaluation/benchmark`, `laya/benchmark`), raising `FileNotFoundError` when nothing matches.
- [x] Verify: executing cell 1 from `Jev-Evaluation/` prints `PROJ: …/Jev-Evaluation/benchmark`.

### Task 2: None-safe verdict formatting

**Files:** Modify `Jev-Evaluation/01_模型评测.ipynb` (cell 17) and `Jev-Evaluation/benchmark/notebooks/laya_vs_jev.ipynb` (cell 17)

- [x] Add `def num(x, nd=3): return f'{x:.{nd}f}' if x is not None else 'n/a'` after `ms`.
- [x] Use `num()` for brier, ECE and the JevBench score.
- [x] Verify: with no laya-local results, the cell prints `brier = n/a, ECE = n/a` and `JevBench Score = n/a` instead of raising `TypeError`.

### Task 3: Task fixtures for `jevbench_intro.ipynb`

**Files:**
- Create: `Jev-Evaluation/llm_eval/tasks/public/{easy,original,hard}.jsonl` (copied from `benchmark/tasks/`)
- Create: `Jev-Evaluation/llm_eval/tasks/public_all.jsonl`
- Create: `Jev-Evaluation/llm_eval/tasks/smoke.jsonl`
- Create: `Jev-Evaluation/llm_eval/tests/test_task_files.py`

- [x] **Step 1:** Write `test_task_files.py` (231 unique ids; matches raw tier files; smoke is a verbatim subset; 6 tasks, 3 question types, one paraphrase pair).
- [x] **Step 2:** Run it and confirm it fails while the fixtures are missing.
- [x] **Step 3:** Generate `public_all.jsonl` with `convert_jevbench.py` and confirm it is byte-identical to `benchmark/tasks/public_all.jsonl`.
- [x] **Step 4:** Build `smoke.jsonl` from these ids, in order: `easy-intent-00`, `easy-fact-00`, `original-policy-01-0`, `original-ordinal-01-0`, `original-ordinal-01-1`, `original-routing-01-0`.
- [x] **Step 5:** `python -m pytest -q tests` reports 4 passed.

### Task 4: Cell 12 and cell 14 of `jevbench_intro.ipynb`

**Files:** Modify `Jev-Evaluation/llm_eval/notebooks/jevbench_intro.ipynb`

- [x] Cell 12: `print("\nbatch summary: …")` becomes `print(f"\nbatch summary: …")`.
- [x] Cell 14: insert `TASKS_PATH = globals().get("TASKS_PATH", "tasks/public_all.jsonl")` before `load_tasks(TASKS_PATH)`.

### Task 5: Laya notebooks in a fresh clone

**Files:** Modify `Jev-Evaluation/01_模型评测.ipynb` and `Jev-Evaluation/benchmark/notebooks/laya_vs_jev.ipynb` (cells 1 and 15)

- [x] Cell 1: after `os.chdir(PROJ)`, if `tasks/public_all.jsonl` is missing, run `scripts/convert_jevbench.py` on `easy`, `original`, `hard` and concatenate the `.llmeval.jsonl` outputs in that order.
- [x] Cell 15: guard the `compare.md` read with `exists()` and print a hint otherwise.
- [x] Keep each cell's original `source` JSON form (string or list) so the diff stays minimal.

### Task 6: End-to-end verification

- [x] In a clean worktree of the branch (no `.env`, `runs/` or generated task files), execute each notebook with nbclient (`kernel_name='llm'`, `resources={'metadata': {'path': <notebook dir>}}`), replacing only the paid cell with `print("SKIPPED paid cell")`. This is `01` cell 11, `laya_vs_jev` cell 11 and `jevbench_intro` cell 12. Expect zero error outputs.
- [x] `python scripts/lint_notebook.py` reports `notebook OK`.
