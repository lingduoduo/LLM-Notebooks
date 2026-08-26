# Active Tool Discovery

## Code map

- **Run first:** `python demo.py --offline` for a complete mechanism smoke test without credentials.
- **Start here:** [`discovery.py`](discovery.py) builds/searches the index; [`agent.py`](agent.py) shows on-demand schema injection.
- **Core behavior:** compare `run_full_injection`, `run_retrieval_prefilter`, and `run_active_discovery`.
- **State / protocol:** available-tool names, appended schemas, and JSON action records.
- **Verifier:** [`test_offline_backend.py`](test_offline_backend.py) plus the exact-match scoring in [`tools_library.py`](tools_library.py).
- **Experiment variable:** catalog size, retrieval `top_k`, strategy and task mix; compare latency, schema footprint and exact completion.
- **Scope:** this directory is a teaching and CI mechanism demo; it does not claim formal real-tool campaign evidence.

---

### Purpose

When an Agent has hundreds of tools, a common approach is to inject every tool JSON schema into the system prompt. That creates two problems:

1. **Token waste**: Full schemas for 126 tools are about **11.6k schema tokens** in the initial prompt and remain part of the context on later reasoning steps.
2. **Instruction-following degradation**: On slightly vague tasks, the model “casts a wide net” and calls generic fallbacks (`web_search` / `google_search` / `universal_search`) together with specialized tools—or even replaces specialized tools with generic search (e.g. looking up a stock price via generic `web_search`).

**Active discovery** keeps only a few base tools plus a `discover_tools(need)` meta-tool in the system prompt. When the model hits a capability gap, it describes the need in natural language; the system retrieves the 3–5 most relevant specialized tools via embedding similarity, appends their schemas as a **user message** (protecting the system-prefix KV cache), and updates the status bar of available tools.

### Mechanisms

```
tools_library.py   126 cross-domain teaching tools (finance/web/arxiv/github/geo/weather/media/...; 17 domains)
                   demo.py uses lightweight mock outputs to isolate tool-selection behavior
                   Intentionally mixes 8 generic/near-synonym tools (web_search, etc.) with inflated descriptions
                   select_tools(size): subset by --tool-set-size to show full injection cost growing with catalog size
discovery.py       Pluggable embedding backend + tool vector index; OpenAIEmbedder uses text-embedding-3-small
                   and caches to .cache/; search(need) = embed need, cosine similarity vs tool vectors, return top-k
agent.py           Three ReAct strategies (text protocol: model outputs one JSON tool call per step)
                   - run_full_injection: all 126 schemas in system prompt
                   - run_retrieval_prefilter: one-shot top-n retrieve by initial query (book’s “retrieval prefilter”)
                   - run_active_discovery: base tools + discover_tools; retrieve-on-demand during execution
offline_backend.py Offline backend: LocalEmbedder (local bag-of-words hash) + MockChatClient (scripted mock)
                   so --offline runs end-to-end without any API key (schema counts/latency measured; accuracy = heuristic routing)
demo.py            Same tasks under selected strategies; prints schema footprint / latency / call traces / exact match; summary table
```

**Why “text inject + text parse” instead of native OpenAI function calling?**
Native function-calling is heavily optimized for tool choice and rarely errs even with hundreds of tools, so it cannot demonstrate long-context instruction-following degradation. Putting schemas in the prompt as plain text and letting the model emit JSON tool calls is the control condition—and matches the book’s “inject schemas into the system prompt (tens of thousands of tokens)” setup.

**Why does embedding retrieval reduce wrong picks?** Generic tools like `web_search` claim to “do everything,” so their semantics are diluted; specialized tools (e.g. `search_news`) have focused descriptions. For a focused `need` (“recent Tesla news”), specialized tools score higher and rank first; generics often never enter top-k and are never loaded—retrieval acts as a precision filter.

**Why isn’t retrieval prefilter enough?** Prefilter (`run_retrieval_prefilter`) matches only the **initial query** once and injects top-n tools. On multi-step cross-domain tasks (e.g. stock price + news), the initial vector often favors the first domain; the second sub-task’s specialized tool may miss top-n. Active discovery defers discovery until each real `need` appears and retrieves separately (offline self-check shows prefilter missing the second tool on half of multi-step tasks—see table below).

### How to run

```bash
# From the repository root:
cd AI-Agent-Tools/active-tool-discovery

# Install this demo's dependencies in your active Python environment:
python -m pip install -r requirements.txt

# Contributors: install test/lint dependencies and run the validation suite:
python -m pip install -r requirements-dev.txt
python -m pytest -q && python -m ruff check .

# Path A: offline mechanism self-check (no keys; schema footprint/latency measured; accuracy = heuristic routing only)
python demo.py --offline

# Path B: real model (needed for small-model instruction-following degradation)
cp env.example .env    # set OPENAI_API_KEY (chat + embeddings both use OpenAI)
# Fallback: if OPENAI_API_KEY is unset but OPENROUTER_API_KEY is set, chat routes via OpenRouter
# (model mapped to openai/gpt-5.6-luna, etc.); tool retrieval falls back to local hash embeddings
# (OpenRouter has no embeddings API).
python demo.py                                   # all 8 tasks × three strategies
python demo.py --strategies full,discovery       # compare only two strategies
python demo.py --tasks finance+news,crypto+news  # selected tasks (comma-separated)
python demo.py --tasks 'opinion(inducement)'      # quote task ids that contain parentheses
python demo.py --tool-set-size 20                # smaller catalog: full-injection disadvantage shrinks
python demo.py --query 'Check NVIDIA stock and find related news' --offline  # one-off task
python demo.py --offline --output results/offline.json         # export structured results
```

Default model `gpt-5.6-luna`; override with `--model` or env: `python demo.py --model gpt-5.6-luna`.
First run builds tool embeddings and caches under `.cache/`. Full flags: `python demo.py --help`
(`--query / --tasks / --strategies / --tool-set-size / --top-k / --prefilter-n / --model / --embed-model / --max-steps / --offline / --output`). Numeric experiment parameters must be positive, and ad-hoc queries that do not map to a known capability are rejected rather than scored vacuously.

### Adaptation / extension

- **Swap chat model**: `MODEL=gpt-4.1-mini python demo.py`; swap embeddings with `EMBED_MODEL=text-embedding-3-large` (cache rebuilds automatically when the embed signature changes).
- **Swap provider / gateway**: chat and embeddings both use the OpenAI SDK; `OpenAI()` reads `OPENAI_BASE_URL`, so any **OpenAI-compatible** gateway works via `OPENAI_BASE_URL=https://your-gateway/v1` (endpoint must offer both chat and embeddings).
- **Swap tasks / inputs**: edit `TASKS` in `tools_library.py` (each has `prompt` and scoring capability slots), or use `--tasks` / `--query`; grow/shrink the catalog in `ALL_TOOLS` in the same file.
- **Offline self-check**: `--offline` uses `offline_backend.py` (local hash embeddings + scripted mock). Good for CI / offline / pipeline smoke tests. It reproduces token/latency structure and “prefilter misses second tool,” not real-model long-context choice behavior (see real gpt-5.6-luna results below).

### Offline mechanism self-check (`python demo.py --offline`)

One real `--offline` run (8 tasks × three strategies). **Schema-token counts and latency are measured with tiktoken/wall-clock**; **accuracy only reflects scripted heuristic routing**, not a real model—the mock is a “strong router” and never degrades, so full injection also scores perfectly. “Complete” requires successful required tool results and a final `finish` action.

| Strategy | Exact match | Task complete | Avg inject tokens | Total inject tokens | Avg latency (s) |
|---|---|---|---|---|---|
| Full injection | 8/8 | 8/8 | 11391 | 91128 | 0.008 |
| Retrieval prefilter | 3/8 | 3/8 | 972 | 7779 | 0.005 |
| Active discovery | 8/8 | 8/8 | 941 | 7529 | 0.009 |

Two **real, reproducible structural** takeaways:

1. **Schema footprint diverges as catalog size grows**: full injection introduces 11,391 schema tokens/task; prefilter and discovery introduce under 1,000 on average (**~12.1×** smaller for active discovery). These are schema-text counts, not cumulative provider billing across every ReAct request. With `--tool-set-size 20` the gap shrinks—confirming “more tools → a larger full-injection context.”
2. **Prefilter structurally misses tools on multi-step cross-domain tasks**: one-shot top-10 completes only 3/8 tasks in the current English catalog; active discovery retrieves per emerging `need` and completes 8/8.

### Conclusions (one real run, gpt-5.6-luna, 2026-07)

> One real LLM run (`python demo.py --model gpt-5.6-luna`, 8 tasks × three strategies, OpenAI chat + `text-embedding-3-small`). gpt-5.6-luna is a reasoning model that only supports default `temperature=1` (no `temperature=0`; code falls back on that error), so this is a **single non-deterministic** run. Scoring: ✅ exact match (all capability slots, no generic fallback misuse); ⚠️ completed but also picked a generic tool; ❌ failed (missed specialized tool, abandoned, or 0 tool calls).

| Task | Full | Prefilter | Discovery | Full tokens | Discovery tokens |
|---|---|---|---|---|---|
| finance+news | ✅ | ❌ | ✅ | 11630 | 883 |
| arxiv+download | ✅ | ❌ | ✅ | 11630 | 927 |
| github+viz | ❌ | ❌ | ❌ | 11630 | 295 |
| weather+calendar | ❌ | ✅ | ✅ | 11630 | 1055 |
| forex+weather | ✅ | ✅ | ❌ | 11630 | 295 |
| crypto+news | ❌ | ⚠️ | ❌ | 11630 | 295 |
| opinion(inducement) | ⚠️ | ❌ | ✅ | 11630 | 688 |
| academic(inducement) | ⚠️ | ⚠️ | ❌ | 11630 | 295 |
| **Exact match** | **3/8** | **2/8** | **4/8** | | |
| **Task complete** | **5/8** | **4/8** | **4/8** | | |
| **Total inject tokens** | | | | **93040** | **4733** |

(Prefilter avg 971 tokens/task, total 7768; mean latency ~11.5 / 9.6 / 10.7 s for the three strategies—measured this run.)

1. **Schema-footprint savings remain robust**: full injection introduced **11,630 schema tokens/task**; discovery **295–1,055**, total 93,040 → 4,733 (**~19.7×**). Part of the larger ratio is gpt-5.6-luna abandoning some tasks without `discover_tools` (only 3 base tools = 295 tokens). This compares schema text introduced into context, not cumulative provider token billing.

2. **Book core phenomenon on two “inducement” tasks**: with vague wording, full injection grabs generic fallbacks—
   - `opinion(inducement)`: full called `search_news, search_news, web_search, search_tweets` (⚠️ included generic **`web_search`**); discovery retrieved `search_news / get_news_by_source / ...` (**no** `web_search`) and only used specialized news tools (✅).
   - `academic(inducement)`: full called 8 tools including **`google_search / universal_search / ask_knowledge_base`** (⚠️); prefilter also misused `google_search / universal_search`.

3. **Another real behavior this run**: conservative reasoning models sometimes **`finish` with 0 tool calls** (“cannot access real-time data”), lowering absolute accuracy for all strategies. Main failure mode was abandon/skip steps—not only wrong tool choice—and it appears under both full injection and discovery.

4. **Boundaries**:
   - Control setup is “schemas as plain text + model emits JSON tool calls”; mock tools return placeholders, so conservative models may refuse—major source of low accuracy here.
   - With `temperature=1` only, per-task outcomes vary across runs; structural conclusions (token savings; full injection misusing generics on inducement tasks) stay directionally stable.
   - For cleaner, reproducible mechanism checks (token/latency + prefilter missing second tool), use the `--offline` table above.

> **One line:** On gpt-5.6-luna, active discovery’s steadiest win is still tokens (~19.7× this run); on vague inducement tasks, embedding retrieval keeps inflated generics (`web_search` / `google_search` / `universal_search`) out of the candidate set. Strong reasoners’ conservative “give up” behavior also lowers absolute accuracy for every strategy—read the table above as-is.

### Model ↔ scaffolding trade-off (weak gpt-4o-mini vs strong gpt-5.6-luna)

> Does stronger models make this scaffolding useless? Comparing the gpt-5.6-luna (strong) run with a gpt-4o-mini (weak) run on the same 8 tasks × three strategies (`python demo.py --model gpt-4o-mini`, 2026-07, same scoring). Scaffolding has two values: one **fades** as models strengthen; one is **model-independent**.

**Weak model gpt-4o-mini real summary:**

| Strategy | Exact match | Task complete | Total inject tokens | Avg latency (s) |
|---|---|---|---|---|
| Full injection | 5/8 | **8/8** | 93040 | 8.38 |
| Retrieval prefilter | 7/8 | 7/8 | 7768 | 4.90 |
| Active discovery | **8/8** | **8/8** | 7266 | 7.65 |

(token 93040 → 7266, **~12.8×**.)

#### Value 1: avoid misusing generic tools — **fades** with stronger models

- **Weak gpt-4o-mini:** under full injection never abandons (8/8 complete) but wide-nets generics on 3 tasks → **5/8 exact**. Discovery blocks inflated generics: **8/8 exact, 0 generic misuse, still 8/8 complete** (+3 exact tasks, zero completion loss).
- **Strong gpt-5.6-luna:** generic misuse only on 2 inducement tasks; discovery cleans those but exact only **3/8 → 4/8 (+1)** and completion **falls 5/8 → 4/8**. Main failure is abandon, not wrong pick—retrieval cannot fix “no tool call at all.” The weakness scaffolding targets is thin on strong models, so that value fades.

#### Value 2: schema-footprint savings — **persists** regardless of model strength

Full injection is always **11,630 tokens/task** (all 126 schemas in system). On-demand injects a few hundred to ~1k:

- Weak gpt-4o-mini: 93,040 → 7,266, **~12.8×**
- Strong gpt-5.6-luna: 93,040 → 4,733, **~19.7×** (larger partly because abandon skips `discover_tools`)

Schema-footprint savings hold on both models and grow with catalog size. Provider billing requires separate usage accounting across every request.

#### One-line summary

> **Stronger models weaken “help it not pick the wrong tool”** (gpt-4o-mini full 5/8→discovery 8/8 exact, zero completion loss; gpt-5.6-luna only 3/8→4/8 and completion drops, because losses are “abandon” not “wrong pick”). **Schema-footprint savings stay** (~12.8× / ~19.7×). On strong models, the main case for active discovery shifts from “fix instruction-following degradation” to “control context size.”

### Files

- `tools_library.py` — 126 tool defs + `select_tools` + mock execution + 8 eval tasks / scoring
- `discovery.py` — pluggable embedders (`OpenAIEmbedder`) + vector index / similarity search
- `agent.py` — three strategies (full / prefilter / discovery) ReAct loops + token stats
- `offline_backend.py` — `LocalEmbedder` + `MockChatClient` for `--offline`
- `demo.py` — multi-strategy CLI demo
- `requirements.txt` / `requirements-dev.txt` / `env.example`

---
