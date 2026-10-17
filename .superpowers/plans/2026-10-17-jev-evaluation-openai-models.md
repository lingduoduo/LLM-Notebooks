# Jev-Evaluation OpenAI Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make five OpenAI models the verbalized baselines in Jev-Evaluation, replacing the seven Chinese-LLM providers.

**Spec:** `.superpowers/specs/2026-10-17-jev-evaluation-openai-models-design.md`

**Architecture:** One `openai` preset plus an `OPENAI_MODELS` price table. Reasoning-model request rules live in `OpenAICompatAdapter`, and `effective_max_output_tokens` keeps the adapter's limit and the ledger reservation equal.

**Tech Stack:** Python 3, stdlib HTTP (`http_post_json`), pytest + monkeypatch, nbclient

## Global Constraints

- No network in unit tests.
- Do not touch Jev/TypeSafe, Laya-local, mock, `figures/`, the snapshot scripts or recorded `runs/`.
- Apply every library change to both `llm_eval/llm_eval` and `benchmark/llm_eval`. The fork keeps `laya_local`.
- Prices come only from `OPENAI_MODELS`.

---

### Task 1: Tests first

**Files:** Create `Jev-Evaluation/llm_eval/tests/test_openai_models.py`

- [x] Write tests for:
  - preset set `{jev, typesafe, openai}`
  - `is_reasoning_model` (gpt-5/6, o-series, vendor prefix, gpt-4.x false)
  - the `effective_max_output_tokens` floor
  - per-model, default, unknown-model and explicit `resolve_prices`
  - request bodies for `gpt-6.1-sol` and `gpt-4.1-mini` via a monkeypatched `openai_compat.http_post_json`
  - `request_options={"reasoning_effort": None}` removal
  - the default model
- [x] Confirm they fail against the previous code (collection error: `OPENAI_MODELS` missing).

### Task 2: Registry and pricing

**Files:** Modify `llm_eval/adapters/__init__.py` in both packages. Delete `llm_eval/llm_eval/adapters/{deepseek,qwen,glm,moonshot,doubao,stepfun,xiaomi}.py`.

- [x] Replace the seven presets and the `k3` alias with the `openai` preset.
- [x] Add `OPENAI_MODELS`, `REASONING_MIN_OUTPUT_TOKENS = 2048` and `effective_max_output_tokens()`.
- [x] Add a `model=""` parameter to `resolve_prices()` and read `OPENAI_MODELS` for `openai`.
- [x] `get_adapter` passes `effective_max_output_tokens(model, max_output_tokens)` to `OpenAICompatAdapter`.

### Task 3: Adapter request shape

**Files:** Modify `llm_eval/adapters/openai_compat.py` in both packages.

- [x] Add `is_reasoning_model()`.
- [x] Reasoning models: `max_completion_tokens` + `reasoning_effort="low"`, no `temperature`. Others: `temperature=0` + `max_tokens`. Apply `request_options` afterwards (`None` removes a key).

### Task 4: Callers

**Files:** Modify `llm_eval/cli.py` and `llm_eval/multi_runner.py` in both packages.

- [x] Resolve the model (`--model` or the preset default), pass it to `resolve_prices`, and pass `effective_max_output_tokens(...)` to both the adapter and `run()`.
- [x] CLI `--adapter` choices: `mock`, `openai_compat`, `openai`, plus `laya_local` in the fork. Update the usage docstring.

### Task 5: Notebook, generator and docs

**Files:** Modify `notebooks/jevbench_intro.ipynb` (cells 5, 10, 11, 12, 15), `scripts/build_notebook.py`, `notebooks/jevbench_intro.md`, `notebooks/README.md` and `.env.example`.

- [x] Cell 11: two keys; provider table = Jev + `OPENAI_MODELS` rows.
- [x] Cell 12, in this order:
  - fallback table built from `OPENAI_MODELS`
  - OpenAI probe via `GET /models/{model}`
  - delete the ARK discovery code
  - export keys once before the pool, with no per-thread set or pop
  - pass `endpoint=spec["base_url"]` instead of mutating `PROVIDERS`
  - update the troubleshooting notes
- [x] Cells 5, 10, 15: update the file tree, the `export` examples and the empty-response note.
- [x] Re-sync each `CELL_N_SRC` in `build_notebook.py` from the notebook, then exec the module and assert every `CELLS[i]` source equals the notebook cell.
- [x] Regenerate the `.md` with `jupyter nbconvert --to markdown`.
- [x] README: keys, base-URL override, and the price table with the source link and date.

### Task 6: Scripts

**Files:** Modify `scripts/run_real_5.py`, `run_one_real.py`, `patch_summary.py` and `run_notebook.sh`.

- [x] `run_real_5.py` builds its list from `OPENAI_MODELS`, with `OPENAI_API_KEY` and the label set to the model id.
- [x] Remove the kimi `temperature=1` override from `run_one_real.py`.
- [x] `patch_summary.py` prices come from `OPENAI_MODELS`.
- [x] `run_notebook.sh` echoes and unsets only `TYPESAFE_API_KEY` and `OPENAI_API_KEY`.

### Task 7: Verification

- [x] `python -m pytest -q tests`: 24 passed.
- [x] Stub end-to-end, on a scratch copy of the project:
  - run `stub_openai.py`, which returns 400 for `max_tokens` or `temperature != 1` on reasoning models
  - point the cell 11 override at it and set `OPENAI_API_KEY=sk-dummy` and `LBEVAL_TASKS_PATH=tasks/smoke.jsonl`
  - run cells 0–14
  - expect zero errors, every model 6/6 ok, and the request log showing the right parameters for each model kind
- [x] `scripts/lint_notebook.py` passes. `grep` finds no old provider names outside `figures/`, the snapshot scripts and `runs/`.
