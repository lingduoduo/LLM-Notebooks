# Standalone Jev-Mem pipeline

This directory now runs independently of the upstream `Jev-Mem` repository.
The local graph engine is inspired by the research notebooks; it is **not the
upstream implementation or a reproduction of its published benchmark results**.
The historical notebook commentary is retained and marked as historical. Executable
cells use this pipeline, with old outputs cleared.

Python 3.10 or newer is required. Offline mode uses only the standard library and
never downloads a model or calls an API.

## Run the full pipeline

From the `LLM-Notebooks` repository root:

```bash
python -m Jev_mem run --mode offline --experiment all --scales 48 96 192 384
```

Each dataset runs all four arms using identical answer instructions:

| Arm | Evidence |
| --- | --- |
| A | Last six observations |
| B | Entire conversation |
| C | Top three vectors from the same observation corpus as D |
| D | Vector anchors plus bounded semantic, temporal, causal, and entity traversal |

The runner validates the dataset, builds and saves D's memory, generates every
answer, evaluates the answers in a separate stage, and renders a report. Each
invocation creates fresh run directories under `Jev_mem/artifacts/`. Choose another
parent directory with `--output /path/to/runs`.

Offline mode uses deterministic multilingual lexical hashing, explicit heuristic
probabilities, extractive answers, and **lexical F1**, not an LLM judge. It tests
execution, persistence, evidence handling, and reporting. It does not measure the
quality of real Jev or learned embeddings. API token usage is `null`, never a
fabricated estimate or zero.

## Live providers

Use a dedicated virtual environment to avoid changing another project's packages:

```bash
python -m venv Jev_mem/.venv
source Jev_mem/.venv/bin/activate
python -m pip install -r Jev_mem/requirements-live.txt
```

Real Jev decisions require `typesafe-sdk`. Answer, judge, and embedding calls use
OpenAI-compatible HTTP endpoints through the standard library.

For DeepSeek answers/judging, SiliconFlow embeddings, and real Jev decisions,
export these variables in the shell that starts the command or Jupyter:

```bash
export DEEPSEEK_API_KEY='your-key'
export SILICONFLOW_API_KEY='your-key'
export TYPESAFE_API_KEY='your-key'
python -m Jev_mem run --mode live --experiment walkthrough
```

The runner does not auto-load `.env` or overwrite global `OPENAI_*` settings. Keys
are read from the process environment. Offline mode ignores live keys entirely.
Missing live credentials fail explicitly; provider failures never silently switch
to mock results. Live commands can incur API charges.

Provider configuration:

| Purpose | Key | Optional settings |
| --- | --- | --- |
| Answers | `ANSWER_API_KEY`, otherwise `DEEPSEEK_API_KEY`, otherwise `OPENAI_API_KEY` | `ANSWER_BASE_URL`, `ANSWER_MODEL` |
| Judge | `JUDGE_API_KEY`, otherwise selected answer key | `JUDGE_BASE_URL`, `JUDGE_MODEL` |
| Embeddings | `EMBEDDING_API_KEY`, otherwise `SILICONFLOW_API_KEY`, otherwise `OPENAI_API_KEY` | `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL` |
| Decisions | `TYPESAFE_API_KEY` | `TYPESAFE_BASE_URL`, `JEV_MODEL` |

When selecting an explicit `ANSWER_API_KEY` or `EMBEDDING_API_KEY`, set the matching
base URL and model explicitly for non-OpenAI providers. DeepSeek shorthand defaults
to `https://api.deepseek.com/v1` / `deepseek-chat`; SiliconFlow shorthand defaults to
`https://api.siliconflow.cn/v1` / `Qwen/Qwen3-Embedding-0.6B`. Otherwise the defaults
are the OpenAI endpoint, `gpt-4o-mini`, and `text-embedding-3-small`. Judge defaults
to the selected answer endpoint/model. Explicit CLI options such as
`--answer-base-url`, `--answer-model`, `--judge-model`, `--embedding-model` override
environment defaults.

Live mode normally uses real Jev decisions and remote embeddings. Explicit
`--decisions heuristic` and/or `--embedding lexical` select ablations; manifests
record these choices. They must not be presented as all-Jev/semantic-embedding runs.

## Bring a dataset

```json
{
  "name": "my-dialogue",
  "observations": [
    {
      "content": "Alice is allergic to peanuts.",
      "timestamp": "2026-09-01T09:00:00Z",
      "role": "user",
      "metadata": {"speaker": "Alice", "entities": ["Alice"], "dia_id": "D1:1"}
    }
  ],
  "questions": [
    {"q": "What is Alice allergic to?", "gold": "peanuts", "cat": "single-hop", "evidence": ["D1:1"]}
  ]
}
```

```bash
python -m Jev_mem run --data /path/to/dataset.json --mode offline
python -m Jev_mem run --data /path/to/locomo10.json --sample 0 --max-questions 10 --mode live
```

Native LoCoMo JSON is normalized with original speaker, session, dialogue IDs,
question category, reference answer (including category-5 adversarial references),
image captions, and evidence IDs retained. Offset timestamps
are converted to UTC before ordering. Missing timestamps remain undated. D's
retrieval evidence IDs are used to compute evidence recall **after retrieval**;
references and gold evidence IDs are never supplied to the retriever or answerer.
The supplied LoCoMo file must be downloaded separately. No results for it are bundled.

## Build once, query later

`build` accepts a JSON list of strings/observations, a normalized dataset, or native
LoCoMo data. It refuses to overwrite an existing memory store.

```bash
python -m Jev_mem build --data /path/to/dataset.json --store /tmp/my-memory
python -m Jev_mem query --store /tmp/my-memory 'What is Alice allergic to?'
```

Use `--mode live` for live providers. Build and query must use the same embedding
provider/model. A stored encoder fingerprint and dimension are checked on load;
JSON has a corruption checksum and uses no pickle. Retrieval settings are saved
with the store. Build/run accept `--max-nodes`, `--max-depth`, and `--graph-budget`.

Classification and relations are typed probabilities. Time edges follow timestamp
order deterministically. Live mode delegates classification, relation decisions,
routing, candidate relevance, and stopping to Jev. The graph keeps original text,
role, timestamps and metadata. All observations are retained; admission filtering
and consolidation are not implemented in this research runner.

Budgets are integer allocations across graph views using the largest-remainder
method. Traversal cannot exceed configured edge, depth, and node limits. Local
heuristics and lexical hashing are deliberately modest baselines, not substitutes
for learned decision quality.

## Artifacts and failure handling

| File | Contents |
| --- | --- |
| `manifest.json` | Mode, dataset digest, implementation, models and public provider endpoints |
| `dataset.json` | Validated input used for this run |
| `memory/memory.json` | Nodes, vectors, edges, encoder identity and retrieval configuration |
| `decisions.jsonl` | Build/query decisions, provider source, usage and retrieval diagnostics |
| `answers.jsonl` | Append-only answer-stage records, before evaluation |
| `evaluations.jsonl` | Final answer records with scores or explicit errors |
| `summary.json` | Per-arm and per-category scores, coverage, usage, latency, graph size |
| `provider-usage.json` | Separate decision, embedding, answer and judge usage |
| `report.svg` | Current run comparison, with mode and scoring metric labels |

Answer latency includes each arm's retrieval plus generation. Judge latency is
separate; build time is a separate one-time cost. Token counts come from each
response's usage fields, not another client's accumulated counter. Scores exclude
failed evaluations and always include coverage counts. Missing usage remains null.
Malformed, nonnumeric, nonfinite or out-of-range judge scores become `judge_error`
records with null scores. Successful zero scores remain real zero scores.

Answers are checkpointed individually; an interrupted run retains its log. Automatic
resume is not implemented. Re-running always creates a new run directory. The CLI exits with status 1 if any
answer or evaluation failed, while still writing available reports. Do not
compare runs with different datasets, providers, prompts, coverage or scoring modes
as if they were the same experiment.

```bash
python -m Jev_mem report /path/to/run
python Jev_mem/scripts/render_jev_mem_charts.py --run /path/to/run
```

The renderer reads current `summary.json`; it no longer relies on the missing
historical snapshot or prints hard-coded research numbers.

## Notebooks and tests

Both notebooks run from the repository root or `Jev_mem/`. They default to offline.
Set `JEV_MEM_MODE=live` before launching Jupyter for real APIs, and optionally set
`JEV_MEM_OUTPUT` to a run parent directory. CLI and notebook paths share the same
implementation.

```bash
python -m unittest discover -s Jev_mem/tests -v
```

Tests include both complete notebooks, CLI build/query/report, memory reload and
bounds, synthetic distractors, LoCoMo normalization, provider failure handling,
strict judge parsing, per-call usage attribution, and HTTP adapters against a local
fake server. Live credentials are not needed for tests.
