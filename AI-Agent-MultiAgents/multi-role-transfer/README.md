# Two Ways to Implement Multi-Role Switching (★★)

Companion code for *Deep Understanding of AI Agents*. This is a controlled comparison of two ways to implement
multi-role behavior over the same shared trajectory:

1. **System-prompt transfer**: `transfer_to_agent(target_role, reason)` swaps the current role's system prompt
   and tool set while retaining the conversation history.
2. **Skill loading**: one fixed system prompt and one fixed tool catalog remain in place; `load_skill(name)` appends
   the selected `SKILL.md` to the trajectory through progressive disclosure.

## What This Experiment Illustrates

- Unlike a predefined stage pipeline, both arms let the model decide which cross-domain capability to use next.
- Both arms use the same canonical `SKILL.md` role documents and retain the same user/assistant/tool trajectory. The
  independent variable is where that document lives: a replaced high-priority system message, or an appended Skill
  tool result. The two arms are **not** matched on instruction length: the Transfer arm adds a
  238-character mechanism note to the canonical role document, while the Skill arm's fixed system
  prompt is about 1,983 characters because it also carries the mandatory Skill protocol. That
  difference, and the Skill arm's mandatory `load_skill("triage")` round-trip, are properties of
  these two implementations rather than of the two mechanisms; see Interpretation and Limitations.
- The comparison separates mechanism metrics (prefix stability and transition calls) from target metrics (task success,
  uncached input tokens, latency, and boundary instruction-following), following Chapter 6's evaluation method.
- The core mechanism is **autonomous role handoff**, but every tool used by an accepted run still performs
  real work. In particular, `web_search` calls Tavily and fails closed when `TAVILY_API_KEY` is absent;
  there is no knowledge-base/mock fallback in the current implementation.

## Architecture

| Property | Path 1 · system-prompt transfer | Path 2 · Skill loading |
|---|---|---|
| Role instruction | Replaces the system prompt | Appends a `SKILL.md` tool result |
| Tool exposure | Only the current role's tools | Fixed superset of tools; Skill supplies the behavioral boundary |
| Prefix cache | Diverges at each role boundary | Stable system/tool prefix; new Skill content is appended |
| Harness enforcement | Can make out-of-role tools unavailable | Requires a separate policy/permission gate for hard enforcement |
| Runtime complexity | Role registry + dynamic prompt/tool switching + loop guards | Stable agent loop + Skill catalog/loader |

Path 1:

```text
                        Shared conversation history (user/assistant/tool messages, retained throughout)
                                       ▲   ▲
   On each LLM call:                    │   │
   [ current role's system prompt ] + history ┘   └ only [ current role's tool set + transfer_to_agent ] exposed

   Two model actions:
     ① Call its own dedicated tools (normal function calling)
     ② Call transfer_to_agent(target_role, reason)
        → Orchestrator swaps "system prompt + tool set", history stays unchanged
        → New role inherits all history (shared context)
```

Path 2:

```text
   fixed [ system prompt + all tool schemas ] + shared history
                                                │
                          load_skill(name) ─────┘
                          → SKILL.md is appended as a tool result
                          → the static prefix is not rewritten
```

5 roles (`roles.py`):

The roster and dedicated-tool table below describe Path 1. Path 2 reuses the same five names as Skill directories;
its runtime tool visibility is intentionally fixed as shown above.

| Role | Description | Dedicated Tool Set |
|------|-------------|-------------------|
| `triage` | Front-desk triage / default entry point, decomposes requests and hands off sequentially, final wrap-up | Only `transfer_to_agent` |
| `research` | Information retrieval | `web_search` (real Tavily search with attributable URLs) |
| `coding` | Programming | `execute_python` (real execution with output capture) |
| `data_analysis` | Data analysis / computation | `calculate`, `descriptive_stats` |
| `writing` | Polishing and writing | `count_characters` |

Each role additionally holds `transfer_to_agent`, enabling autonomous handoff of control to colleagues.

Code structure:

- `tools.py` — Implementation of each role's dedicated tools + OpenAI function-calling schema
- `roles.py` — 5 role definitions (system prompts + tool sets) + `transfer_to_agent` schema
- `orchestrator.py` — Handoff orchestrator (shared history + main loop for swapping system prompts/tool sets, with deadlock prevention and self-handoff rejection)
- `skills/*/SKILL.md` — The five role capabilities used by the Skill arm
- `skill_orchestrator.py` — Stable-prefix Skill loader and agent loop
- `evaluation.py` — Deterministic outcome Rubric and trajectory-prefix boundary cases
- `experiment_protocol.json` — Pre-registered controls, strata, metrics and statistical tests
- `tasks.example.json` — Small mixed-strata task-file template for a smoke run
- `tasks.complex.example.json` — Eight multi-stage tasks with branching rules, source conflicts, explicit-stop
  instructions, prompt-injection probes, no-side-effect coding invariants and revision/loop constraints
- `run_comparison.py` — Paired live A/B runner and machine-readable report
- `demo.py` — Single-command demo entry point
- `tests/` — Offline regressions for tool dispatch and local tools

## How to Run

```bash
# From the repository root: use the shared Chapter 10 environment
uv sync --locked --python 3.12 --extra ch10

# Activate it before changing directories:
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Windows cmd: .venv\Scripts\activate.bat

# pip fallback when uv is not installed:
# python -m pip install -e ".[ch10]"

cd AI-Agent-MultiAgents/multi-role-transfer

# Single-project compatibility path, still supported during migration:
# python -m pip install -r requirements.txt

# Configure API key (choose one)
export OPENAI_API_KEY=your-openai-api-key        # Direct export
export TAVILY_API_KEY=your-tavily-key            # Required by research.web_search; no mock fallback
# or: cp env.example .env and fill in

python demo.py
```

`demo.py` remains the single-run illustration of Path 1. Run the paired comparison with the same model in both arms:

```bash
python run_comparison.py \
  --model gpt-5.6-luna \
  --trials 5 \
  --output validation/comparison/luna-YYYYMMDD.json
```

For a formal paired campaign, provide a JSON array of records via `--task-file tasks.json`; each record may include
`id`, `prompt`, `kind`, and observable gates such as `required_capabilities`, `required_tools`, `forbidden_tools`,
`required_tool_order`, `required_output_patterns`, `forbidden_output_patterns`, `min_source_urls`,
`min_output_source_urls` and `max_deliverable_chars`. `kind` can be `cagr`, `coding`, `writing` or `complex`;
`--trials` means repetitions per task.
The built-in `--task` is intentionally a single-task smoke path, not the 30-sample claim.

For example, the three-row template can be repeated ten times as a 30-cell pilot (expand the task file for a real
production decision):

```bash
python run_comparison.py --model gpt-5.6-luna \
  --task-file tasks.example.json --trials 10 \
  --output validation/comparison/luna-pilot.json
```

For the harder rule-following pilot, use the eight-task suite. It intentionally mixes long chains with short
single-role and early-stop cases so that an extra Skill load is not automatically treated as a cost win:

```bash
python run_comparison.py --model gpt-5.6-luna \
  --task-file tasks.complex.example.json --trials 4 \
  --output validation/comparison/luna-complex-pilot.json
```

The complex records are pre-registered task specifications, not fabricated expected answers. Their deterministic
gates check observable tool calls, tool order, source URLs, forbidden actions, uncertainty language and bounded
deliverables; numerical correctness and usefulness still require the blinded quality review described below. A
formal result should expand this suite to at least 30 paired task samples and retain every failed trajectory.

The default run executes five paired end-to-end trials and one pass over each boundary case per arm. It requires
`TAVILY_API_KEY` because the research tool fails closed. To report monetary cost, pass the provider's current prices
explicitly rather than baking volatile prices into the repository:

```bash
python run_comparison.py --model gpt-5.6-luna --trials 5 \
  --input-price-per-million <price> \
  --cached-input-price-per-million <price> \
  --output-price-per-million <price>
```

When a deterministic Rubric changes, rescore saved trajectories without spending another API call:

```bash
python run_comparison.py --replay validation/comparison/previous.json \
  --output validation/comparison/previous-rescored.json
```

### Pre-registered evaluation protocol

Hold the model, provider, task text, temperature, tool implementations, maximum steps and trial count fixed. Alternate
the two arms within each trial, use a fresh conversation for every cell, source both arms' role instructions from the
same `SKILL.md` files, and retain every trajectory including failures.
Use at least 30 paired task samples (or report the five-trial run only as a smoke test), stratified across research →
analysis → writing, coding → writing, single-role tasks and tasks that explicitly stop after an intermediate stage.
This is an architecture comparison, not a one-variable prompt ablation: Path 1 has hard tool isolation while Path 2
keeps a fixed tool superset to preserve the prefix. Add a third fixed-tools/dynamic-prompt arm if a pure prompt-carrier
causal estimate is required.

Report three groups of metrics:

- **Cost**: API calls, input/output tokens, cached and uncached input tokens, wall-clock p50/p95, and price-recomputed
  dollars. Prefix-cache hit tokens are the target measurement; prompt length alone is only a mechanism proxy.
- Distinguish model **KV/prompt cache** from a **KB/Skill document cache**: the former is measured by provider
  `cached_tokens`; the latter needs its own hit/miss, `name@version` version-key and load-latency fields. A Skill cache hit does not
  imply a model-prefix cache hit.
- This protocol defines Skill loading as appending `SKILL.md` through a tool result. A runtime that mutates a
  system/developer message or tool schemas when loading a Skill changes the prefix and belongs in a separate arm.
- **Actual effect**: deterministic outcome gates first (source URL, real calculation call, correct CAGR range, format,
  deliverable length, and required capability-sequence completion), then a blinded pairwise judge from a different model or human reviewer for usefulness and writing quality. If a runtime
  adds a wrap-up envelope, apply the length limit to the text passed to `count_characters`, identically in both arms.
  Apply a hallucination veto.
- **Boundary instruction-following**: use trajectory prefixes frozen before the first error for current-user override, prompt injection in
  retrieved text, missing evidence and transition loops. Score only observable next actions, required evidence, and forbidden actions; do not infer hidden reasoning.

For binary paired outcomes report Pass@1 and Pass-consecutive-k, a paired 95% bootstrap interval, and McNemar's test;
for token/latency deltas report paired medians and bootstrap intervals. Randomize A/B display order for pairwise judging
and judge the swapped order a second time to control position bias. Do not infer a winner from one successful trace.

Environment configuration:
`OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`),
`OPENAI_MODEL` (default `gpt-5.6-luna`), and `TAVILY_API_KEY` for the research role's real web search.

All live model calls require `OPENAI_API_KEY` (or `--api-key` for the comparison runner).
The default endpoint is OpenAI directly; there is no automatic OpenRouter fallback.
If an existing `.env` sets `OPENAI_BASE_URL` to Moonshot or another provider, change it to
`https://api.openai.com/v1`. The official experiment runner always uses the direct OpenAI endpoint.
Choose a different OpenAI model with `OPENAI_MODEL` or `--model`.

### Command-Line Arguments

All arguments are optional; if omitted, behavior is identical to the original version (runs the default `cagr` scenario). Run
`python demo.py --help` to see the full documentation.

| Argument | Effect |
|----------|--------|
| `--list-roles` | **Offline self-check**: Only prints the role roster + built-in scenarios and exits, **no API Key required** |
| `--scenario {cagr,solar,coding}` | Select a built-in scenario (default `cagr`); `coding` routes to the `coding` role to actually run code |
| `--task "..."` | Custom task text, overrides `--scenario` |
| `--role {triage,research,coding,data_analysis,writing}` | Specify the **starting role** (alias `--starting-role`, default `triage`) |
| `--interactive` | **Interactive multi-turn**: Reuses the same orchestrator, roles and shared history persist across turns |
| `--model gpt-5.6-luna` | Temporarily overrides `OPENAI_MODEL` |
| `--max-steps 30` | Hard upper limit on LLM rounds per message (default 20, prevents infinite loops) |

Examples:

```bash
python demo.py --list-roles            # Offline view of roles/scenarios, no API call
python demo.py --scenario coding       # Scenario routed to the coding role
python demo.py --task "Research and summarize…" # Custom task
python demo.py --role research         # Start from the research role
python demo.py --interactive           # Interactive multi-turn, type exit to quit
```

Run the provenance-complete OpenAI + Tavily acceptance campaign with:

```bash
python run_official_experiment.py --model gpt-5.6-luna --run-id exp10-1-openai-tavily-receipts-YYYYMMDD-vN
```

This path requires `OPENAI_API_KEY` and `TAVILY_API_KEY`. It defaults to
`https://api.openai.com/v1` and accepts `--base-url` (or `OPENAI_BASE_URL`) for any
OpenAI-compatible endpoint. The receipt file and the provenance gate names carry the
provider label derived from that URL, so the default writes `openai_receipts.json`
while `--base-url https://api.moonshot.cn/v1` writes `moonshot_receipts.json` -- which
is how the retained `kimi-k2.5` bundle below was produced.
It retains credential-free raw OpenAI requests/responses, response IDs
and usage, raw Tavily HTTP response bodies with the API key removed from the
stored request, current runtime source hashes, artifact hashes, and a combined
behavior/provenance acceptance record.

Three built-in scenarios (`SCENARIOS`): `cagr` (default, new energy vehicle sales → CAGR → investment summary),
`solar` (same chain with a different set of photovoltaic installation data), `coding` (routes to the `coding` role
to actually run a Fibonacci script via `execute_python`, then `writing`/`triage` wraps up).

## Offline Validation

```bash
# From the repository root; include dev tools for pytest.
uv sync --locked --python 3.12 --extra ch10 --extra dev
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

cd AI-Agent-MultiAgents/multi-role-transfer
python -m pytest tests
python -m pytest tests/test_skill_comparison.py
python demo.py --list-roles
```

`tests/` contains offline regressions for `count_characters`, `execute_python` timeouts, and tool-dispatch error handling. They do not require an API key.

## Language

Runtime sources, Skill documents and the task sets are English. The task sets now request
**English deliverables**: `tasks.formal.json`, `tasks.complex.example.json` and `tasks.example.json`
ask for English summaries, their Chinese scoring alternatives have been dropped, and every
`max_deliverable_chars` was scaled by 3 (for example 140 -> 420, 160 -> 480) because a Chinese
character carries roughly three times the content of an English one. `evaluation.py` keeps its
bilingual patterns on purpose: the retained campaigns are Chinese, and replaying them with the
current scorer must still reproduce the recorded numbers.

The retained evidence under `validation/` is Chinese, because that is what the models and Tavily
actually returned. Those files are never rewritten: `moonshot_receipts.json` and
`tavily_receipts.json` carry `request_sha256`, `response_sha256`, `raw_response_bytes` and real
provider response IDs, so translating them in place would both break the manifests and present
translated text as raw provider output. Instead:

```bash
python translate_validation.py            # write validation/translated/
python translate_validation.py --check    # coverage report only
```

[`translate_validation.py`](translate_validation.py) writes English reading copies under
`validation/translated/`, leaving the originals byte-identical. Each copy starts with a
`_translation` block recording its source, the coverage counts, and every JSON path still in
Chinese. Role and Skill system prompts, Skill documents and tool schemas are substituted from
`roles.py`, `skill_orchestrator.py` and `tools.py`, so a copy cannot drift from the code;
everything else comes from [`validation/translation_map.json`](validation/translation_map.json).
`validation/translated/` itself is derived and git-ignored.

The map is keyed by **source id** -- `translate_validation.source_key`, the first 32 hex
characters of the SHA-256 of the Chinese source -- so `translation_map.json` holds no Chinese and
reads as plain English. [`validation/translation_sources.json`](validation/translation_sources.json)
records id -> source so each translation can be checked against what it translates;
`translate_validation.py` hashes the string it finds in a bundle and never reads that file, but it
is the only remaining copy of the strings from the two in-place-translated summaries, so it is not
disposable.

Coverage is deliberately uneven, and the reports say so:

| Bundle | Status |
|---|---|
| `exp10-1-kimi-k2.5-tavily-20260730-v1/v2` (historical summaries) | **English in place, no Chinese at all.** Nothing hashes or references these two files, so they were converted rather than mirrored. Their Chinese strings survive only as entries in [`validation/translation_sources.json`](validation/translation_sources.json), which is why that file is not disposable. |
| `exp10-1-kimi-k2.5-tavily-receipts-20260730-v3` (official run) | **The reading copies contain no Chinese at all**, including every Tavily source excerpt. The bundle itself is byte-identical, because `moonshot_receipts.json` / `tavily_receipts.json` carry `request_sha256`, `response_sha256`, `raw_response_bytes` and real provider response IDs. |
| `judge.json` (comparison campaign) | **No Chinese at all** in the reading copy. |
| `campaign.json` (comparison campaign) | Every model output, reasoning trace, tool result, handoff reason, task prompt and task spec is translated. What remains is **127 verbatim Tavily page excerpts (~100k Chinese characters)** of third-party news articles, PDF-extracted reports and site boilerplate, each identified by its URL. |

Every residual excerpt is reported in its file's `_translation.left_in_chinese` with `kind: "source_excerpt"`, so the remainder is explicit rather than silent. A regression test asserts that **no model output anywhere is left untranslated** -- only `source_excerpt` may remain.

To translate the excerpts too, [`fill_translation_map.py`](fill_translation_map.py) sends whatever the map cannot resolve through an OpenAI-compatible model in batches, caching each result back into `translation_map.json`, never overwriting an existing entry and writing after every batch so an interrupted run resumes:

```bash
python fill_translation_map.py --dry-run     # show the queue and batch plan
python fill_translation_map.py               # translate what is left
python translate_validation.py               # rebuild the reading copies
```

Translations are faithful to the record, so a run that asked for "a Chinese summary of at most 120
characters" still says exactly that in its reading copy. The copies are not evidence and hashes
must not be verified against them.

## Formal v2 evidence

The authoritative package is [`validation/comparison/runs/exp10-1-qwen35flash-20260809-v2/`](validation/comparison/runs/exp10-1-qwen35flash-20260809-v2/), independently checked by [`validate_comparison.py`](validate_comparison.py) (12/12 gates). The campaign uses `qwen/qwen3.5-flash-02-23` through OpenRouter, 30 paired tasks at temperature 0, an eight-round per-cell limit, 60 main trajectories, and 12 boundary trajectories. The Skill arm now requires `load_skill("triage")` before any specialist tool.

For this bounded model/configuration, Skill passes 15/30 deterministic task gates versus Transfer's 2/30. Skill's median delta is +6,855 uncached input tokens (+5,054 once its mandatory `load_skill("triage")` call is
excluded), +4.368 seconds, and +$0.00044304 repriced cost. Replaying `campaign.json` with the current scorer reproduces both pass counts exactly. An independent Gemini 2.5 Flash Lite judge reviewed all 30 pairs twice: Skill 32, Transfer 20, and 8 ties across 60 judgments. These are bounded architecture results, not model-independent superiority claims.

**Known defect in the retained judge evidence.** The judge's position-swap control drew a fresh
random order inside each repeat instead of inverting the first, so the two repeats were
independent rather than opposite. In the retained campaign **17 of 30 pairs happened to show the
judge the same order twice**, meaning those pairs are not position-controlled and the judge
tallies above carry an unquantified position bias. The bundle is kept unmodified as the record of
what ran. `judge_comparison.position_swap_flags` now guarantees opposite orders, and the
`blind_quality_judge_position_swapped` gate compares the recorded `shown_order` values instead of
merely counting two judgments per pair -- so it rejects this retained `judge.json`, and a
re-judged campaign is required before the quality tally can be quoted without this caveat.
The deterministic 15/30 vs 2/30 result does not depend on the judge.

The Skill arm keeps all tool schemas visible to preserve a stable prefix, but the Harness rejects tools before a Skill is loaded or when the current Skill does not authorize them. Visibility is therefore not mistaken for progressive disclosure.

## Path 1 Demo and Historical Evidence

Archived validation receipts and manifests describe the original Chinese-language runs and are
preserved unchanged. Their runtime source hashes describe the Chinese-era code and cannot match the
translated, since-corrected source, so equality there is an assertion that must fail forever.
Verification instead works as follows (see [`provenance.py`](provenance.py)): each bundle's own
artifacts must still recompute exactly, and every recorded runtime source must either still match
its manifest hash or be listed with a reason in
[`validation/source_drift.json`](validation/source_drift.json). Undeclared drift fails, so a file
that everyone believes is pinned cannot change quietly. Both retained bundles currently declare all
of their sources as changed. These historical records do not establish live validation of the
English runtime.

`demo.py` presents a composite task requiring **multiple cross-domain switches**:

> Look up China's new energy vehicle sales for 2021–2023 → Calculate the compound annual growth rate (CAGR) → Write a Chinese summary for investors

Expected autonomous handoff chain:

```text
triage → research → data_analysis → writing
```

- `triage` determines the first step is to look up data, hands off to `research`;
- `research` uses `web_search` to find the three years of sales data, hands off to `data_analysis`;
- `data_analysis` uses `calculate` to compute CAGR ≈ 64.22%, hands off to `writing`;
- `writing` synthesizes the sales data and CAGR from **the prior history** and directly produces the final draft.

`writing` never retrieved or computed anything itself, yet it can reference accurate sales figures and growth rates —
this is evidence of **shared context**. After execution, the full handoff chain, each `from→to` and `reason`,
and a **role-by-role summary** (who called which dedicated tools, who produced the final reply) are printed,
making it clear at a glance how "different specialized roles take turns on the same history."

> Note: Real LLM output has randomness; specific wording or step counts in a given run may vary slightly, but the handoff mechanism is consistent.

### Expected Output Shape

The following excerpt illustrates the console format in English translation. The 101-character count and
120-character limit refer to the original Chinese draft, not its English translation below. The canonical accepted real run is
[`validation/runs/exp10-1-kimi-k2.5-tavily-receipts-20260730-v3/manifest.json`](validation/runs/exp10-1-kimi-k2.5-tavily-receipts-20260730-v3/manifest.json):
it records Moonshot `kimi-k2.5`, three real Tavily searches with source URLs, the complete handoff chain,
the calculation tool call, and the counted draft. All 9 behavior and 6 provenance gates passed. The run
retains nine raw Moonshot requests/responses with unique response IDs and usage, three raw Tavily response
bodies, five runtime source hashes, and four artifact hashes; all declared hashes recompute and the
credential scan found zero hits. The older v2 JSON remains as a sanitized summary-only historical run.

```text
=== Role Roster (5 specialized roles) ===
• triage — Front-desk triage (default entry)
    Tool set: ['transfer_to_agent']
    System prompt (first line): You are the 'front-desk triage' role of the general assistant system, and the default entry point.
• research — Information retrieval specialist
    Tool set: ['web_search', 'transfer_to_agent']
    ...(other roles omitted, see full list in the role table above)

┌── Current role: Information Retrieval Specialist (research)   Tools: ['web_search', 'transfer_to_agent']
└── 🔧 Calling tool web_search args={'query': 'China 2021 2022 2023 new energy vehicle sales CPCA CAAM'}
    → [Search Results · China Passenger Car Association / CAAM]…2021: 3.521 million units / 2022: 6.887 million units / 2023: 9.495 million units
┌── Current role: Data Analysis Specialist (data_analysis)   Tools: ['calculate', 'descriptive_stats', 'transfer_to_agent']
└── 🔧 Calling tool calculate args={'expression': '(9.495/3.521)**(1/2)-1'}
    → (9.495/3.521)**(1/2)-1 = 0.6421562289791105

================ Run Summary ================
Autonomous handoff chain: triage → research → data_analysis → writing → triage
Handoff count: 4
  1. triage → research  |  reason: Need to first retrieve China's 2021, 2022, 2023 new energy vehicle sales and reliable sources, to provide data for subsequent CAGR calculation and investor summary.
  2. research → data_analysis  |  reason: Retrieved 2021, 2022, 2023 NEV sales data; please calculate the two-year CAGR from 2021 to 2023 and provide the result for subsequent writing.
  3. data_analysis → writing  |  reason: Sales data and CAGR completed: 2021: 3.521M, 2022: 6.887M, 2023: 9.495M; 2021–2023 CAGR=(9.495/3.521)^(1/2)-1=64.22%. Please write a Chinese investor summary of no more than 120 characters based on this.
  4. writing → triage  |  reason: Completed investor summary and verified length (101 characters, within 120-char limit)… Please do final wrap-up confirmation.

Role-by-role breakdown (who used which tools, who produced the final reply):
  triage        : (routing/handoff only, no dedicated tools used)  ⇒ Produced final reply
  research      : web_search
  data_analysis : calculate
  writing       : count_characters

Final output:
According to public data from CAAM, China's new energy vehicle sales grew from 3.521 million units in 2021 to 6.887 million in 2022 and 9.495 million in 2023. The two-year CAGR from 2021 to 2023 reached 64.2%, indicating rapid market expansion with significant growth potential.
```

## Interpretation and Limitations

- The default model is `gpt-5.6-luna`; whether the handoff follows the expected chain depends heavily on the selected model's instruction-following ability. Switching models may yield different results. Note that no retained bundle was produced with this default: the official run used Moonshot `kimi-k2.5` and the comparison campaign used `qwen/qwen3.5-flash-02-23` via OpenRouter, so the default is unvalidated here.
- Prefix-cache reuse is provider dependent. Use the provider-reported `cached_tokens` field when available; otherwise
  label any prefix-hash comparison as a mechanism proxy rather than measured cache savings.
- The Skill arm intentionally keeps all tool schemas stable and visible. A Skill is a soft behavioral boundary, not a
  permission boundary. High-risk tools such as deletion, payment, and messaging still need a harness-level allowlist, approval gate, or separate sandbox.
- `load_skill` adds an extra tool round and appends instructions to the trajectory. On short, single-role tasks that
  overhead can outweigh cache savings; the experiment must include such tasks instead of only long handoff chains.
- **The two arms are not perfectly matched, and the mismatch favours Transfer on the cost metrics.** The Skill arm must
  spend a real `load_skill("triage")` round-trip to earn the `triage` capability, while the Transfer arm receives it for
  free from `start_role` and has it prepended unconditionally to `observed_capabilities`. The Skill arm's fixed system
  prompt is also ~1,983 characters against the Transfer arm's ~238-character mechanism note. That round-trip is now
  measured rather than merely noted: every run records `bootstrap_api_calls` and `bootstrap_input_tokens` (the calls
  made before any Skill is loaded), and the paired report carries
  `uncached_input_token_delta_excluding_skill_bootstrap` alongside the raw delta. On the retained campaign the median
  delta falls from **+6,855 to +5,054 uncached input tokens** once that one mandatory call is removed, so roughly a
  quarter of the apparent penalty is the protocol rather than the architecture. The prompt-length difference is not
  adjusted for and remains a property of these two implementations.
- Prefix stability is **measured from the retained request bodies**, not recomputed from the current source. Each run's
  `metrics.prefix_source` records which it is, and a run kept without provider receipts reports `null` metrics and an
  explicit `unavailable` note rather than a number reconstructed from today's code. On the retained campaign this
  confirms the mechanism empirically: the Skill arm shows 1 unique prefix and 0 changes, the Transfer arm 3 unique
  prefixes and 2 changes per trajectory.
- The `research` role requires a live Tavily credential. Missing credentials, HTTP failures, or empty provider results are surfaced explicitly and never replaced with canned facts.
- Real LLM output has randomness: the exact number of handoff steps, the wording of each `reason`, whether the `coding` role is visited, etc., may vary between runs, but the handoff mechanism itself is consistent.
- `orchestrator.py` has a hard `max_steps` limit (default 20) and a correction prompt for "same (role, tool, arguments) called ≥3 times consecutively" to prevent model infinite loops; this is a safety net, not an indication that every run will use all these steps.
