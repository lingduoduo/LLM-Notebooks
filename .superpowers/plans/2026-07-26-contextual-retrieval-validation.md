# Contextual Retrieval Validation and Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and finish the OpenAI-only contextual retrieval implementation under the canonical `AI-Agent-KnowledgeBase/contextual-retrieval` path.

**Architecture:** Treat the existing implementation as the baseline and repair only verified gaps. Validate the repository name boundary first, then the OpenAI configuration and runtime boundaries, and finally the offline retrieval experiment and user-facing entry points.

**Tech Stack:** Python 3, OpenAI Python SDK, pytest, Ruff, `unittest.mock`, BM25

## Global Constraints

- LLM calls authenticate only with `OPENAI_API_KEY`.
- The default LLM model is `gpt-5.6-terra`.
- `LLM_MODEL`, `--model`, and `--llm-model` may override the model.
- Anthropic remains credited as the source of the Contextual Retrieval technique.
- Local, Dify, RAPTOR, and GraphRAG retrieval backends remain available.
- Unit tests must not make network calls.
- `AI-Agent-KnowledgeBase/contextual-retrieval` is the canonical module path.

---

### Task 1: Validate the canonical repository path

**Files:**
- Modify if needed: `.superpowers/plans/2026-07-25-contextual-retrieval-openai-only.md`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/README.md`
- Test: repository-wide static path audit

**Interfaces:**
- Consumes: canonical path `AI-Agent-KnowledgeBase/contextual-retrieval`
- Produces: documentation and runnable commands that reference the canonical path

- [ ] **Step 1: Audit active files for the retired path**

Run:

```bash
rg -n "AI-Agent/contextual-retrieval" \
  .superpowers/plans \
  .superpowers/specs \
  AI-Agent-KnowledgeBase/contextual-retrieval
```

Expected: only historical implementation-plan references may fail the audit initially; active README and design files must already use, or be changed to use, the canonical path.

- [ ] **Step 2: Repair each stale active path**

Replace every active `AI-Agent/contextual-retrieval` path found in Step 1 with `AI-Agent-KnowledgeBase/contextual-retrieval`. Do not alter historical Git objects or review-diff artifacts under `.superpowers/sdd`.

- [ ] **Step 3: Verify the retired directory is absent and canonical directory exists**

Run:

```bash
test ! -d AI-Agent/contextual-retrieval
test -d AI-Agent-KnowledgeBase/contextual-retrieval
rg -n "AI-Agent/contextual-retrieval" \
  .superpowers/plans \
  .superpowers/specs \
  AI-Agent-KnowledgeBase/contextual-retrieval
```

Expected: both directory checks pass and `rg` exits 1 with no matches.

- [ ] **Step 4: Commit canonical-name corrections if files changed**

```bash
git add .superpowers/plans .superpowers/specs AI-Agent-KnowledgeBase/contextual-retrieval/README.md
git commit -m "docs: use contextual retrieval canonical path"
```

If no files changed, record the passing audit and do not create an empty commit.

---

### Task 2: Validate OpenAI-only configuration and clients

**Files:**
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/config.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/agent.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/contextual_chunking.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_config.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_clients.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_only_surfaces.py`

**Interfaces:**
- Consumes: `OPENAI_API_KEY`, optional `LLM_MODEL`
- Produces: `LLMConfig.get_client_config() -> tuple[dict[str, str], str]`
- Produces: `OpenAI(api_key=...)` clients for agent and context generation

- [ ] **Step 1: Run the OpenAI boundary tests**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest -q \
  test_openai_config.py \
  test_openai_clients.py \
  test_openai_only_surfaces.py
```

Expected: all tests pass without network access.

- [ ] **Step 2: Audit executable and setup surfaces for alternate LLM routing**

Run:

```bash
rg -ni \
  "llm_provider|--provider|--llm-provider|kimi|moonshot|doubao|siliconflow|openrouter|groq|together ai|deepseek" \
  config.py agent.py contextual_chunking.py main.py \
  index_local_laws_contextual.py evaluation/evaluate.py \
  env.example README.md README_LEGAL_INDEXING.md
```

Expected: no matches. Anthropic references are deliberately excluded because they credit the retrieval technique.

- [ ] **Step 3: Add a failing regression test for any confirmed gap**

Add the smallest assertion to the relevant existing test module. Examples:

```python
assert client_config == {"api_key": "test-key"}
assert config.llm.model == "gpt-5.6-terra"
assert "--provider" not in source
```

Run the new test alone and confirm it fails for the observed reason before changing production code.

- [ ] **Step 4: Apply the minimal production repair**

Keep the public boundaries exact:

```python
client_config, model = llm_config.get_client_config()
client = OpenAI(**client_config)
```

Do not introduce `base_url`, provider selection, or alternate API-key variables.

- [ ] **Step 5: Re-run the OpenAI boundary tests**

Run:

```bash
python -m pytest -q \
  test_openai_config.py \
  test_openai_clients.py \
  test_openai_only_surfaces.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit verified repairs if files changed**

```bash
git add config.py agent.py contextual_chunking.py \
  test_openai_config.py test_openai_clients.py test_openai_only_surfaces.py
git commit -m "fix: enforce OpenAI-only contextual retrieval"
```

If no files changed, do not create an empty commit.

---

### Task 3: Validate CLIs and runtime configuration

**Files:**
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/main.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/index_local_laws_contextual.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/evaluation/evaluate.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/quickstart.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_cli.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_simple.py`

**Interfaces:**
- Consumes: `Config.from_env()`
- Produces: `--model` and `--llm-model` OpenAI model overrides
- Produces: live smoke tests gated by both `OPENAI_API_KEY` and `RUN_LIVE_SMOKE_TESTS=1`

- [ ] **Step 1: Run focused CLI and smoke-boundary tests**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest -q test_openai_cli.py test_simple.py
```

Expected: all tests pass; no external API request occurs.

- [ ] **Step 2: Exercise each CLI help surface**

Run:

```bash
python main.py --help
python evaluation/evaluate.py --help
python index_local_laws_contextual.py --help
```

Expected: commands exit 0, model options say `OpenAI model ID override`, and no provider option appears.

- [ ] **Step 3: Add a failing regression test for any confirmed CLI gap**

Use `monkeypatch` and stub classes to prevent network or filesystem side effects. Assert exact option behavior, for example:

```python
assert parsed_model == "gpt-4.1-mini"
assert "--provider" not in help_text
```

Run the new test alone and confirm the expected failure.

- [ ] **Step 4: Apply the minimal CLI or configuration repair**

Preserve these supported options:

```text
main.py: --model
evaluation/evaluate.py: --model
index_local_laws_contextual.py: --llm-model
```

Do not add provider flags.

- [ ] **Step 5: Re-run focused CLI tests and help commands**

Run the commands from Steps 1 and 2.

Expected: all tests and help commands pass.

- [ ] **Step 6: Commit verified repairs if files changed**

```bash
git add main.py index_local_laws_contextual.py evaluation/evaluate.py \
  quickstart.py test_openai_cli.py test_simple.py
git commit -m "fix: align contextual retrieval entry points"
```

If no files changed, do not create an empty commit.

---

### Task 4: Validate retrieval behavior and complete verification

**Files:**
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/compare_retrieval.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/contextual_tools.py`
- Modify if needed: `AI-Agent-KnowledgeBase/contextual-retrieval/README.md`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_document_ids.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/test_history_limit_zero.py`
- Test: `AI-Agent-KnowledgeBase/contextual-retrieval/evaluation/retrieval_eval.json`

**Interfaces:**
- Consumes: `document_store.json` and `evaluation/retrieval_eval.json`
- Produces: offline BM25 plain-versus-contextual recall results
- Produces: a complete passing unit-test and static-validation report

- [ ] **Step 1: Run the complete unit-test suite**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest -q
```

Expected: all tests pass; live smoke tests remain skipped unless explicitly enabled.

- [ ] **Step 2: Run the offline retrieval comparison**

Run:

```bash
python compare_retrieval.py
```

Expected: command exits 0 and reports both plain and contextual recall without an API request.

- [ ] **Step 3: Run syntax, formatting, and static checks**

Run:

```bash
python -m compileall -q .
ruff check .
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 4: Diagnose any failure before changing code**

For each failure, identify whether it is a product defect, stale generated artifact, missing local service, or optional live dependency. Add a regression test for product defects; do not weaken assertions or enable network calls to make tests pass.

- [ ] **Step 5: Apply minimal tested repairs and rerun the failing command**

Change only files directly implicated by Step 4. Run the smallest failing test first, then repeat Steps 1–3.

- [ ] **Step 6: Commit verified repairs if files changed**

```bash
git add AI-Agent-KnowledgeBase/contextual-retrieval
git commit -m "fix: finish contextual retrieval validation"
```

If no files changed, do not create an empty commit.

- [ ] **Step 7: Record final evidence**

Report:

```text
canonical path audit: pass/fail
OpenAI-only static audit: pass/fail
pytest: passed/skipped/failed counts
offline comparison: plain and contextual recall@1/@3/@5
compileall: pass/fail
ruff: pass/fail
git diff --check: pass/fail
commits created: hashes or none
```
