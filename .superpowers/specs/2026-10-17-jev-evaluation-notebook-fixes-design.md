# Jev-Evaluation Notebook Fixes

## Goal

Make the three Jev-Evaluation notebooks run end to end from a fresh checkout of this repository, without changing what they measure.

- `Jev-Evaluation/01_模型评测.ipynb`
- `Jev-Evaluation/benchmark/notebooks/laya_vs_jev.ipynb`
- `Jev-Evaluation/llm_eval/notebooks/jevbench_intro.ipynb`

## Problems

1. **`01_模型评测.ipynb` cannot find its project.** Cell 1 only searches for `laya/benchmark`, the layout of the upstream jev-cookbook repository. In this repository the project is `Jev-Evaluation/benchmark`, so cell 1 raises `AssertionError` and every later cell fails with `NameError`.
2. **Conclusion cells crash when a runner has no results.** The text-verdict loop in `01_模型评测.ipynb` and `laya_vs_jev.ipynb` formats `brier`, `ece` and the JevBench score with `:.3f` / `:.2f`. When Laya serve is not running, laya-local has no scorable results, these values are `None`, and the cell raises `TypeError`.
3. **`jevbench_intro.ipynb` has no task files.** Cells 6, 8, 9 and 13 read `tasks/public_all.jsonl` and `tasks/smoke.jsonl` relative to `Jev-Evaluation/llm_eval/`, but that directory has no `tasks/`.
4. **`jevbench_intro.ipynb` cell 12 prints a literal placeholder.** The batch summary line is missing its `f` prefix and prints `{len(all_runs)}`.
5. **`jevbench_intro.ipynb` cell 14 depends on cell 12's session state.** It reads `TASKS_PATH`, which only cell 12 defines. After a kernel restart, re-running the chart cell without re-running the paid cell 12 raises `NameError`.
6. **The Laya notebooks need a generated, gitignored task file.** `benchmark/.gitignore` excludes `tasks/public_all.jsonl`, so in a fresh clone cell 5 of `01_模型评测.ipynb` and `laya_vs_jev.ipynb` raises `FileNotFoundError` unless the user first runs the conversion steps by hand.
7. **The Laya compare cell crashes when there are no runner results.** `write_markdown_table` writes nothing when there are no rows, but cell 15 reads `compare.md` unconditionally. In a fresh clone it raises `FileNotFoundError`, and that cascades into `NameError` in cells 17 and 19.

## Design

### Project discovery (problem 1)

Reuse the search that `laya_vs_jev.ipynb` already uses: walk up from the working directory and try `.`, `benchmark`, `Jev-Evaluation/benchmark` and `laya/benchmark`. Accept a candidate only if it is named `benchmark` and contains an `llm_eval/` package. Raise `FileNotFoundError` naming the starting directory if nothing matches. The notebook keeps working in the upstream layout.

### None-safe verdict formatting (problem 2)

Add a `num(x, nd=3)` helper next to the existing `pct` and `ms` helpers. It returns `'n/a'` for `None` and the formatted number otherwise. Use it for brier, ECE and the JevBench score. The numeric summary tables above the verdict are unchanged.

### Task files (problem 3)

Create `Jev-Evaluation/llm_eval/tasks/`:

- `public/{easy,original,hard}.jsonl`: the raw JevBench public files, copied from `benchmark/tasks/`. This matches the layout the notebook's own structure cell documents.
- `public_all.jsonl`: the 231 converted tasks. Regenerating it with `scripts/convert_jevbench.py` from the raw files produces a byte-identical file.
- `smoke.jsonl`: six tasks taken verbatim from `public_all.jsonl`. The notebook documents it as "6 tasks, one paraphrase pair"; the original file was never committed. The selection covers all three question types and uses `original-ordinal-01-0/1` as the paraphrase pair: `easy-intent-00`, `easy-fact-00`, `original-policy-01-0`, `original-ordinal-01-0`, `original-ordinal-01-1`, `original-routing-01-0`.

`llm_eval/.gitignore` does not ignore `tasks/`, so these files are committed. In contrast, `benchmark/.gitignore` ignores its generated `public_all.jsonl`.

### Cell 12 and cell 14 (problems 4 and 5)

- Add the missing `f` prefix in cell 12.
- In cell 14, fall back to `tasks/public_all.jsonl` when `TASKS_PATH` is not defined: `TASKS_PATH = globals().get("TASKS_PATH", "tasks/public_all.jsonl")`. This matches cell 12's own default.

### Laya notebooks in a fresh clone (problems 6 and 7)

- In cell 1 of both Laya notebooks, after `os.chdir(PROJ)`, build `tasks/public_all.jsonl` when it is missing. Run `scripts/convert_jevbench.py` on each raw tier file, then concatenate `easy`, `original`, `hard`. These are the same steps `scripts/fetch_public_tasks.sh` prints, so the result is identical to a hand-built file. An existing file is left alone.
- In cell 15, print `compare.md` only if it exists; otherwise print a hint to run section 5 first.

## Non-goals

- Running the paid multi-run cells (Jev API, real LLM providers) or starting Laya serve.
- Changing metrics, scoring, or the providers under evaluation.
- Syncing `scripts/build_notebook.py` with the notebook. The generator has already drifted from the committed notebook. That is handled by the OpenAI-models change, which must touch the generator anyway.

## Verification

- Execute each notebook with nbclient on the `llm` kernel from its own directory, in a clean worktree of the branch (no `.env`, no `runs/`, no generated task files), replacing only the paid cells with a no-op. Every remaining cell must finish without an error output.
- `pytest Jev-Evaluation/llm_eval/tests/test_task_files.py`: task files load, `public_all` has 231 unique ids that match the raw tier files, and `smoke` is a verbatim subset covering all question types with exactly one paraphrase pair.
- `scripts/lint_notebook.py` passes.
