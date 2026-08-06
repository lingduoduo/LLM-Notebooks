# Agent Cost Analysis English Validity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the English agent-cost benchmark preserve cache eligibility, currency semantics, valid fixture data, and honest offline-trace provenance.

**Architecture:** Keep the existing four-scenario API and CLI. Add small validation helpers and regression tests around the benchmark's data invariants, then regenerate the bundled offline trace as deterministic synthetic English-workload data with explicit provenance.

**Tech Stack:** Python 3, pytest, standard-library `json`/`ast`, optional `tiktoken`, Markdown documentation.

## Global Constraints

- All module text remains English-only.
- The stable system prompt must contain at least 1,024 tokens under the benchmark tokenizer.
- Scenario commerce values remain CNY/RMB.
- Existing scenario keys and public entry points remain unchanged.
- Offline mode remains deterministic and credential-free.
- Bundled synthetic data must never be described as live-observed usage.

---

### Task 1: Enforce English benchmark invariants

**Files:**
- Create: `AI-Agent-Evaluation/agent-cost-analysis/test_benchmark_invariants.py`
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/agent.py`

**Interfaces:**
- Consumes: `agent.STABLE_SYSTEM_PROMPT`, `agent.TOOL_RESULTS`, `agent.TOOL_SUMMARIES`, `agent._ntok(text)`.
- Produces: `agent.MIN_CACHEABLE_PREFIX_TOKENS: int`, `agent.validate_benchmark() -> None`.

- [ ] **Step 1: Write failing invariant tests**

```python
import json
import re

import agent


def test_stable_prompt_is_cache_eligible():
    assert agent._ntok(agent.STABLE_SYSTEM_PROMPT) >= agent.MIN_CACHEABLE_PREFIX_TOKENS


def test_tool_results_are_valid_json():
    for _, _, result in agent.TOOL_RESULTS:
        assert isinstance(json.loads(result), dict)


def test_scenario_uses_cny_without_dollar_amounts():
    corpus = "\n".join([agent.STABLE_SYSTEM_PROMPT, *agent.TOOL_SUMMARIES.values()])
    assert "CNY" in corpus
    assert not re.search(r"\$\s*\d", corpus)


def test_benchmark_validation_passes():
    agent.validate_benchmark()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `/Users/linghuang/miniconda3/bin/pytest -q test_benchmark_invariants.py`

Expected: failures because `MIN_CACHEABLE_PREFIX_TOKENS` and `validate_benchmark` do not exist and the English prompt is too short.

- [ ] **Step 3: Implement the minimum behavior**

Add `MIN_CACHEABLE_PREFIX_TOKENS = 1024`, expand `STABLE_SYSTEM_PROMPT` with relevant order-state, logistics, refund-policy, risk, notification, escalation, privacy, and tool-contract guidance, and change `$469`/`$500` to `CNY 469`/`CNY 500`. Implement:

```python
def validate_benchmark() -> None:
    import json

    prompt_tokens = _ntok(STABLE_SYSTEM_PROMPT)
    if prompt_tokens < MIN_CACHEABLE_PREFIX_TOKENS:
        raise ValueError(
            f"Stable prefix has {prompt_tokens} tokens; "
            f"at least {MIN_CACHEABLE_PREFIX_TOKENS} are required"
        )
    for step, tool, result in TOOL_RESULTS:
        try:
            json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool-result JSON for {step}/{tool}") from exc
```

Call `validate_benchmark()` at the beginning of `run_scenario` so live runs fail clearly if fixtures regress.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `/Users/linghuang/miniconda3/bin/pytest -q test_benchmark_invariants.py`

Expected: all invariant tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add AI-Agent-Evaluation/agent-cost-analysis/agent.py AI-Agent-Evaluation/agent-cost-analysis/test_benchmark_invariants.py
git commit -m "fix: preserve English benchmark invariants"
```

### Task 2: Regenerate and label the English offline trace

**Files:**
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/sample_trace.json`
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/demo.py`
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/test_trace_offline.py`

**Interfaces:**
- Consumes: trace top-level metadata and existing `collect_offline`, `dump_output` functions.
- Produces: trace metadata fields `provenance: "synthetic" | "observed"` and `language: "en"`.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_bundled_trace_is_labeled_synthetic_english():
    data = json.loads(Path(DEFAULT_TRACE).read_text())
    assert data["provenance"] == "synthetic"
    assert data["language"] == "en"


def test_saved_live_trace_is_labeled_observed(tmp_path):
    path = tmp_path / "trace.json"
    dump_output(path, [], Pricing(0.15, 0.075, 0.6), "test-model")
    data = json.loads(path.read_text())
    assert data["provenance"] == "observed"
    assert data["language"] == "en"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `/Users/linghuang/miniconda3/bin/pytest -q test_trace_offline.py`

Expected: metadata assertions fail because the fields do not exist.

- [ ] **Step 3: Implement trace provenance and deterministic English counts**

Add `"provenance": "synthetic"` and `"language": "en"` to the bundled trace. Regenerate its prompt/tool-context counts from the English fixtures using deterministic assistant placeholders and the four existing message-building strategies. Keep completion counts and latency fixed synthetic values, recompute component summaries, and ensure every cached count is between zero and its prompt count. Add observed metadata to `dump_output`, and print provenance in `collect_offline`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `/Users/linghuang/miniconda3/bin/pytest -q test_trace_offline.py`

Expected: all trace tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add AI-Agent-Evaluation/agent-cost-analysis/sample_trace.json AI-Agent-Evaluation/agent-cost-analysis/demo.py AI-Agent-Evaluation/agent-cost-analysis/test_trace_offline.py
git commit -m "fix: label and regenerate English offline trace"
```

### Task 3: Align documentation and perform full verification

**Files:**
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/README.md`
- Modify: `AI-Agent-Evaluation/agent-cost-analysis/env.example`
- Test: `AI-Agent-Evaluation/agent-cost-analysis/test_benchmark_invariants.py`

**Interfaces:**
- Consumes: implemented cache threshold, CNY semantics, and trace provenance.
- Produces: accurate user documentation for live and offline modes.

- [ ] **Step 1: Add failing documentation assertions**

```python
def test_readme_describes_synthetic_trace_and_cny():
    readme = Path("README.md").read_text()
    assert "synthetic" in readme.lower()
    assert "CNY" in readme
    assert "1,024" in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `/Users/linghuang/miniconda3/bin/pytest -q test_benchmark_invariants.py::test_readme_describes_synthetic_trace_and_cny`

Expected: failure because README does not yet document these guarantees.

- [ ] **Step 3: Update documentation**

Document that live mode yields observed measurements, the bundled offline trace is deterministic synthetic English data, the refund amount is CNY 469, and the stable prefix is tested at 1,024 or more tokens. Keep `env.example` consistent with credential precedence and model override behavior.

- [ ] **Step 4: Run full verification**

```bash
cd AI-Agent-Evaluation/agent-cost-analysis
/Users/linghuang/miniconda3/bin/pytest -q
/Users/linghuang/miniconda3/bin/python -m py_compile agent.py config.py demo.py tracer.py test_*.py
/Users/linghuang/miniconda3/bin/python -m json.tool sample_trace.json >/dev/null
/Users/linghuang/miniconda3/bin/python demo.py --offline --scenario all
rg -n '[\p{Han}]' .
```

Expected: all tests pass, compilation and JSON parsing exit zero, the offline CLI reports four scenarios and synthetic provenance, and the character scan has no matches.

- [ ] **Step 5: Commit Task 3**

```bash
git add AI-Agent-Evaluation/agent-cost-analysis/README.md AI-Agent-Evaluation/agent-cost-analysis/env.example AI-Agent-Evaluation/agent-cost-analysis/test_benchmark_invariants.py
git commit -m "docs: explain English benchmark guarantees"
```
