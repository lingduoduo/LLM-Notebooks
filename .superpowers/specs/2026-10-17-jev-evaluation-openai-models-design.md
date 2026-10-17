# Jev-Evaluation: OpenAI Models as the Verbalized Baselines

## Goal

Replace every general-purpose LLM in Jev-Evaluation with OpenAI models. The seven providers being replaced are DeepSeek, Qwen, GLM, Kimi/Moonshot, Doubao, StepFun and Xiaomi MiMo. The harness keeps comparing native decision models against LLMs that are asked to write out a probability distribution ("verbalized"). Only the verbalized side changes.

## Scope

In scope:

- `llm_eval` and its `benchmark/` fork: provider presets, the OpenAI-compatible adapter, price resolution, CLI and multi-runner
- The intro notebook (`llm_eval/notebooks/jevbench_intro.ipynb`), its generator `scripts/build_notebook.py`, and its markdown mirror
- The helper scripts `run_real_5.py`, `run_one_real.py`, `patch_summary.py` and `run_notebook.sh`
- `.env.example` and the notebook README

Unchanged:

- **Jev / TypeSafe and Laya-local** are the native systems under evaluation.
- **`mock`** is the zero-cost baseline.
- **`figures/jevbench-2026-09-26/`, the two `*_jevbench_snapshot.py` scripts, and recorded `runs/` data** document a past benchmark of the old models. Renaming them would misstate what was measured.

## Model lineup

Prices are the standard tier, short context, from the official OpenAI pricing page, checked 2026-10-17. They are in USD per 1M tokens.

| Model | In | Out | Kind |
|---|---|---|---|
| `gpt-6.1-sol` | 2.00 | 10.00 | reasoning |
| `gpt-6-luna` | 0.10 | 0.50 | reasoning, preset default |
| `gpt-5.6-terra` | 2.00 | 12.00 | reasoning |
| `gpt-5.6-luna` | 0.20 | 1.20 | reasoning |
| `gpt-4.1-mini` | 0.40 | 1.60 | non-reasoning baseline (`temperature=0`) |

The lineup is five models spanning two generations, three price tiers and one non-reasoning control. `gpt-6-astra` ($10/$50) is excluded because a 231-task run would far exceed the default $1.50-per-provider cap.

## Design

### Registry

- One `openai` preset: `base_url=https://api.openai.com/v1`, `default_model=gpt-6-luna`, `mode=verbalized`.
- `OPENAI_MODELS: dict[model, {price_in_per_m, price_out_per_m}]` is the single price source. The notebook builds its provider table from it, and the scripts and `patch_summary.py` read it too.
- `resolve_prices(name, in, out, model="")` looks for an explicit price first, then the model's entry in `OPENAI_MODELS` (for `openai`), then the preset. Callers in `cli.py` and `multi_runner.py` pass the resolved model, using the preset default when `--model` is empty.
- The seven per-provider shim modules are deleted. Nothing imported them, and the registry routes through `OpenAICompatAdapter` directly.

### Reasoning models

GPT-5+, GPT-6 and o-series models reject `temperature != 1` and `max_tokens`, and they spend output tokens on hidden reasoning, so a 200-token cap returns empty content.

- `is_reasoning_model(model)` lives in `openai_compat.py`, which avoids a circular import, and is re-exported from `adapters`. It matches the `gpt-5`, `gpt-6`, `o1`, `o3` and `o4` prefixes, ignoring any `vendor/` prefix.
- For reasoning models the request sends `max_completion_tokens` and `reasoning_effort="low"`, and drops `temperature`. Other models keep `temperature=0` and `max_tokens`. `request_options` can override or remove any key; a value of `None` removes it.
- `effective_max_output_tokens(model, requested)` raises the limit to 2048 for reasoning models. `get_adapter`, `cli.py` and `multi_runner.py` all use it, so the adapter's request limit and the ledger's per-task reservation stay equal. The runner halts when a settlement exceeds its reservation.

### Notebook

- **Cell 11:** two keys, `TYPESAFE_API_KEY` and `OPENAI_API_KEY`. The provider table is Jev plus one row per `OPENAI_MODELS` entry.
- **Cell 12, key check:** OpenAI models are checked with `GET /models/{model}`. It's free and returns 404 for a model the project can't use. The old one-token `POST /chat/completions` check sent `max_tokens`, which reasoning models reject.
- **Cell 12, shared key:** all OpenAI rows share one key. Keys are written to the environment once, before the thread pool starts, and are never removed. Previously each thread set its key and popped it on exit. A finished thread could then remove the key from a sibling, and the user's own exported key was deleted too.
- **Cell 12, base URL:** a base-URL override is passed per runner as `RunnerConfig.endpoint`. Before, it was patched into the shared `PROVIDERS` dict from several threads.
- **Cell 12, cleanup:** the ARK endpoint discovery code and the Chinese-provider troubleshooting notes are removed.
- **Generator and mirror:** `scripts/build_notebook.py` is re-synced from the notebook so that rebuilding reproduces it exactly; it had already drifted. `jevbench_intro.md` is regenerated with `jupyter nbconvert --to markdown`, which drops a stale export of an old run.

## Non-goals

- Migrating to the Responses API.
- Changing metrics, scoring or the JevBench composite.
- Adding the OpenAI models to the Laya notebooks, which compare only Laya and Jev.

## Risks

- **Not verified against the live OpenAI API**, because no key was available. If a model rejects `reasoning_effort="low"`, the fix is a per-row `request_options={"reasoning_effort": None}`.
- **The $1.50 cap may stop the larger reasoning models** (`gpt-6.1-sol`, `gpt-5.6-terra`) before 231 tasks if they use many reasoning tokens. `LBEVAL_CAP_USD` raises it.

## Verification

- `pytest Jev-Evaluation/llm_eval/tests` (24 tests, no network). They cover the presets, `is_reasoning_model`, `effective_max_output_tokens`, per-model `resolve_prices`, and the request body for reasoning and non-reasoning models, with `request_options` removal. They fail against the previous code.
- Stub end-to-end: a local server mimics `GET /models/{id}` and `POST /chat/completions`, and returns 400 for `max_tokens` or `temperature != 1` on reasoning models. The intro notebook runs cells 0–14 against it on `tasks/smoke.jsonl`: zero error outputs, every model 6/6 ok, and recorded cost equals `tokens × OPENAI_MODELS price`.
- `build_notebook.py` reproduces every notebook cell exactly, and `lint_notebook.py` passes.
