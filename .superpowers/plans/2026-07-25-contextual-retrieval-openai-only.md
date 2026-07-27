# Contextual Retrieval OpenAI-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every LLM path in `AI-Agent-KnowledgeBase/contextual-retrieval` use the official OpenAI client, `OPENAI_API_KEY`, and the default model `gpt-5.6-terra`.

**Architecture:** Replace the provider router in `LLMConfig` with one OpenAI configuration boundary shared by the agent and contextual chunker. Preserve Chat Completions and all retrieval-backend choices while removing provider selection from CLIs, evaluation metadata, examples, and documentation.

**Tech Stack:** Python 3, `openai` Python SDK, dataclasses, pytest, `unittest.mock`

## Global Constraints

- Authenticate LLM calls only with `OPENAI_API_KEY`.
- Use `gpt-5.6-terra` as the single default LLM model.
- Permit `LLM_MODEL`, `--model`, and `--llm-model` overrides.
- Keep the Chat Completions request flow.
- Keep local, Dify, RAPTOR, and GraphRAG retrieval backends unchanged.
- Do not make network calls in unit tests.

---

### Task 1: Replace provider routing with direct OpenAI configuration

**Files:**
- Create: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_config.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/config.py`

**Interfaces:**
- Produces: `LLMConfig.get_api_key() -> Optional[str]`
- Produces: `LLMConfig.get_client_config() -> tuple[dict[str, str], str]`
- Produces: `LLMConfig.model: str`, default `"gpt-5.6-terra"`
- Produces: `Config.from_env() -> Config`, reading `LLM_MODEL` but not `LLM_PROVIDER`

- [ ] **Step 1: Write failing configuration tests**

```python
import pytest

from config import Config, LLMConfig


def test_llm_config_defaults_to_openai_terra(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config = Config.from_env()
    assert config.llm.model == "gpt-5.6-terra"
    assert not hasattr(config.llm, "provider")


def test_client_config_uses_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client_config, model = LLMConfig().get_client_config()
    assert client_config == {"api_key": "env-key"}
    assert model == "gpt-5.6-terra"


def test_explicit_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client_config, _ = LLMConfig(api_key="explicit-key").get_client_config()
    assert client_config == {"api_key": "explicit-key"}


def test_model_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    assert Config.from_env().llm.model == "gpt-4.1-mini"


def test_missing_openai_api_key_has_actionable_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        LLMConfig().get_client_config()


def test_legacy_provider_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    config = Config.from_env()
    assert not hasattr(config.llm, "provider")
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_config.py -v
```

Expected: failures show the current `kimi` default, `provider` attribute, alternate routing, and provider argument on `get_api_key`.

- [ ] **Step 3: Implement the minimal OpenAI-only configuration**

In `config.py`:

- Delete `_openrouter_model_id`, `Provider`, `PROVIDER_DEFAULTS`, provider-to-key mappings, base URLs, and OpenRouter fallback logic.
- Replace `LLMConfig` with the focused fields and methods below while retaining its existing generation controls:

```python
@dataclass
class LLMConfig:
    """Configuration for OpenAI language-model calls."""

    model: str = "gpt-5.6-terra"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = True

    def get_api_key(self) -> Optional[str]:
        return self.api_key or os.getenv("OPENAI_API_KEY")

    def get_client_config(self) -> tuple[Dict[str, str], str]:
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set the OPENAI_API_KEY environment variable."
            )
        return {"api_key": api_key}, self.model
```

In `Config.from_env()`, delete the `LLM_PROVIDER` branch and retain:

```python
if model := os.getenv("LLM_MODEL"):
    config.llm.model = model
```

- [ ] **Step 4: Run the focused configuration tests**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_config.py -v
```

Expected: all six tests pass.

- [ ] **Step 5: Commit the configuration boundary**

```bash
git add AI-Agent-KnowledgeBase/contextual-retrieval/config.py AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_config.py
git commit -m "refactor: use OpenAI-only LLM configuration"
```

---

### Task 2: Make agent and contextual chunking clients OpenAI-only

**Files:**
- Create: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_clients.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/agent.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/contextual_chunking.py`

**Interfaces:**
- Consumes: `LLMConfig.get_client_config() -> tuple[dict[str, str], str]`
- Produces: `AgenticRAG.client`, initialized with only the OpenAI API key
- Produces: `ContextualChunker.client`, initialized with only the OpenAI API key
- Preserves: `_reasoning_safe_temperature(model: str, requested: float) -> float`

- [ ] **Step 1: Write failing client-construction tests**

```python
from unittest.mock import patch

from config import Config, LLMConfig


def test_agent_constructs_direct_openai_client():
    from agent import AgenticRAG

    config = Config()
    config.llm = LLMConfig(api_key="test-key")
    with patch("agent.OpenAI") as openai:
        agent = AgenticRAG(config)
    openai.assert_called_once_with(api_key="test-key")
    assert agent.model == "gpt-5.6-terra"


def test_contextual_chunker_constructs_direct_openai_client():
    from contextual_chunking import ContextualChunker

    with patch("contextual_chunking.OpenAI") as openai:
        chunker = ContextualChunker(
            llm_config=LLMConfig(api_key="test-key"),
            use_contextual=True,
        )
    openai.assert_called_once_with(api_key="test-key")
    assert chunker.model == "gpt-5.6-terra"


def test_gpt5_temperature_remains_reasoning_safe():
    from agent import _reasoning_safe_temperature as agent_temperature
    from contextual_chunking import _reasoning_safe_temperature as chunk_temperature

    assert agent_temperature("gpt-5.6-terra", 0.3) == 1
    assert chunk_temperature("gpt-5.6-terra", 0.3) == 1
    assert agent_temperature("gpt-4.1-mini", 0.3) == 0.3
```

- [ ] **Step 2: Run the client tests and verify the expected failure**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_clients.py -v
```

Expected: initialization fails because runtime code still reads `provider` and supports `base_url`.

- [ ] **Step 3: Simplify agent client initialization**

In `agent.py`, retain the `OpenAI` import and replace `_init_llm_client` with:

```python
def _init_llm_client(self):
    """Initialize the official OpenAI client."""
    client_config, self.model = self.config.llm.get_client_config()
    self.client = OpenAI(**client_config)
    logger.info("Using OpenAI model: %s", self.model)
```

Change the class docstring to describe an OpenAI-backed ReAct agent. Remove provider-specific initialization logging. Update `_reasoning_safe_temperature` so its docstring discusses GPT-5 only and its return expression no longer checks `kimi-k3`.

- [ ] **Step 4: Simplify contextual chunker client initialization and cost reporting**

In `contextual_chunking.py`, replace `_init_llm_client` with:

```python
def _init_llm_client(self):
    """Initialize the official OpenAI client for context generation."""
    client_config, self.model = self.llm_config.get_client_config()
    self.client = OpenAI(**client_config)
    logger.info("Using OpenAI model %s for context generation", self.model)
```

Update `_reasoning_safe_temperature` to check GPT-5 only. Replace the provider-dependent cost branch in `get_statistics()` with one OpenAI estimate:

```python
cost_per_1k = 0.03
stats["estimated_cost"] = (stats["total_context_tokens"] / 1000) * cost_per_1k
```

Do not change the Chat Completions calls or retrieval behavior.

- [ ] **Step 5: Run focused and existing unit tests**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_clients.py test_openai_config.py test_history_limit_zero.py test_document_ids.py -v
```

Expected: all tests pass without network calls.

- [ ] **Step 6: Commit runtime conversion**

```bash
git add AI-Agent-KnowledgeBase/contextual-retrieval/agent.py AI-Agent-KnowledgeBase/contextual-retrieval/contextual_chunking.py AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_clients.py
git commit -m "refactor: initialize OpenAI clients directly"
```

---

### Task 3: Remove provider selection from CLIs, evaluation, and smoke tests

**Files:**
- Create: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_cli.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/main.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/index_local_laws_contextual.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/evaluation/evaluate.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/test_simple.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/compare_retrieval.py`

**Interfaces:**
- Consumes: `LLMConfig(model: str = "gpt-5.6-terra", api_key: Optional[str] = None)`
- Produces: main CLI `--model`
- Produces: legal-indexing CLI `--llm-model`
- Produces: evaluation result field `config.llm_model`

- [ ] **Step 1: Write a failing static CLI test**

```python
from pathlib import Path


ROOT = Path(__file__).parent


def test_cli_surfaces_do_not_offer_provider_selection():
    files = [
        ROOT / "main.py",
        ROOT / "index_local_laws_contextual.py",
        ROOT / "evaluation" / "evaluate.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "--provider" not in source
    assert "--llm-provider" not in source
    assert ".llm.provider" not in source


def test_evaluation_records_model_without_provider():
    source = (ROOT / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
    assert '"llm_model": self.agent.model' in source
    assert '"llm_provider"' not in source
```

- [ ] **Step 2: Run the static tests and verify they fail**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_cli.py -v
```

Expected: both tests fail on current provider arguments and evaluation metadata.

- [ ] **Step 3: Update executable entry points**

Apply these exact behavior changes:

- In `main.py`, make `setup_environment()` call `config.llm.get_api_key()` and log only `Please set OPENAI_API_KEY`; delete `--provider` and its override branch.
- In `index_local_laws_contextual.py`, delete `--llm-provider`; construct `LLMConfig(model=args.llm_model)` only when `args.llm_model` is present, otherwise pass `None`.
- In `evaluation/evaluate.py`, delete `--provider` and its override branch; remove `llm_provider` from saved configuration and its printed summary.
- In `test_simple.py`, replace provider output with model output and make the live-query condition check only `OPENAI_API_KEY`.
- In `compare_retrieval.py`, replace the multi-provider key instruction with `Set OPENAI_API_KEY in .env`.

- [ ] **Step 4: Run CLI and regression tests**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_cli.py test_openai_config.py test_openai_clients.py test_history_limit_zero.py test_document_ids.py -v
python main.py --help
python index_local_laws_contextual.py --help
python evaluation/evaluate.py --help
```

Expected: tests pass; help output contains model options but no provider option.

- [ ] **Step 5: Commit executable-surface cleanup**

```bash
git add AI-Agent-KnowledgeBase/contextual-retrieval/main.py AI-Agent-KnowledgeBase/contextual-retrieval/index_local_laws_contextual.py AI-Agent-KnowledgeBase/contextual-retrieval/evaluation/evaluate.py AI-Agent-KnowledgeBase/contextual-retrieval/test_simple.py AI-Agent-KnowledgeBase/contextual-retrieval/compare_retrieval.py AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_cli.py
git commit -m "refactor: remove alternate LLM provider options"
```

---

### Task 4: Align environment template and documentation

**Files:**
- Create: `AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_only_surfaces.py`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/env.example`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/README.md`
- Modify: `AI-Agent-KnowledgeBase/contextual-retrieval/README_LEGAL_INDEXING.md`

**Interfaces:**
- Produces: documented setup using `OPENAI_API_KEY`
- Produces: documented default `LLM_MODEL=gpt-5.6-terra`

- [ ] **Step 1: Write a failing audit test**

```python
from pathlib import Path


ROOT = Path(__file__).parent
ACTIVE_SURFACES = [
    ROOT / "config.py",
    ROOT / "agent.py",
    ROOT / "contextual_chunking.py",
    ROOT / "main.py",
    ROOT / "index_local_laws_contextual.py",
    ROOT / "compare_retrieval.py",
    ROOT / "test_simple.py",
    ROOT / "env.example",
    ROOT / "README.md",
    ROOT / "README_LEGAL_INDEXING.md",
    ROOT / "evaluation" / "evaluate.py",
]
REMOVED_TERMS = (
    "kimi",
    "moonshot",
    "doubao",
    "siliconflow",
    "openrouter",
    "groq",
    "together ai",
    "deepseek",
    "llm_provider",
)


def test_active_surfaces_are_openai_only():
    for path in ACTIVE_SURFACES:
        content = path.read_text(encoding="utf-8").lower()
        for term in REMOVED_TERMS:
            assert term not in content, f"{term!r} remains in {path.relative_to(ROOT)}"


def test_environment_template_uses_openai_defaults():
    content = (ROOT / "env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=your_openai_api_key_here" in content
    assert "LLM_MODEL=gpt-5.6-terra" in content
    assert "LLM_PROVIDER=" not in content
```

- [ ] **Step 2: Run the audit and verify it fails**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_only_surfaces.py -v
```

Expected: failures enumerate the remaining alternate-provider references and old environment template.

- [ ] **Step 3: Reduce `env.example` to OpenAI LLM settings**

Replace its LLM section with:

```dotenv
# OpenAI LLM configuration
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-5.6-terra
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=150
```

Retain unrelated knowledge-base, chunking, agent, performance, and cost-control settings.

- [ ] **Step 4: Update documentation**

In both README files:

- Use `OPENAI_API_KEY` as the only LLM credential.
- Use `gpt-5.6-terra` in commands and configuration.
- Remove provider selection examples and provider-relative cost claims.
- Keep the citation and descriptive references to Anthropic's Contextual Retrieval research, because they identify the algorithm being demonstrated rather than an LLM provider.
- Describe `--model` and `--llm-model` as optional OpenAI model overrides.

- [ ] **Step 5: Run the audit and complete regression suite**

Run:

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest test_openai_only_surfaces.py test_openai_cli.py test_openai_clients.py test_openai_config.py test_history_limit_zero.py test_document_ids.py -v
python -m compileall -q .
```

Expected: every test passes and compilation exits successfully.

- [ ] **Step 6: Perform final source audit**

Run:

```bash
rg -n -i "kimi|moonshot|doubao|siliconflow|openrouter|groq|together|deepseek|LLM_PROVIDER|--provider|--llm-provider" \
  AI-Agent-KnowledgeBase/contextual-retrieval/config.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/agent.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/contextual_chunking.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/main.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/index_local_laws_contextual.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/compare_retrieval.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/test_simple.py \
  AI-Agent-KnowledgeBase/contextual-retrieval/env.example \
  AI-Agent-KnowledgeBase/contextual-retrieval/README.md \
  AI-Agent-KnowledgeBase/contextual-retrieval/README_LEGAL_INDEXING.md \
  AI-Agent-KnowledgeBase/contextual-retrieval/evaluation/evaluate.py
```

Expected: no output. A reference to Anthropic's Contextual Retrieval research is allowed and is intentionally not part of this search.

- [ ] **Step 7: Commit documentation and audit coverage**

```bash
git add AI-Agent-KnowledgeBase/contextual-retrieval/env.example AI-Agent-KnowledgeBase/contextual-retrieval/README.md AI-Agent-KnowledgeBase/contextual-retrieval/README_LEGAL_INDEXING.md AI-Agent-KnowledgeBase/contextual-retrieval/test_openai_only_surfaces.py
git commit -m "docs: document OpenAI-only contextual retrieval"
```

---

### Task 5: Final verification

**Files:**
- Verify only; no planned modifications

**Interfaces:**
- Verifies all interfaces and constraints from Tasks 1–4

- [ ] **Step 1: Run the complete local test set**

```bash
cd AI-Agent-KnowledgeBase/contextual-retrieval
python -m pytest -v
```

Expected: all collected tests pass. Tests requiring a live API remain skipped unless `OPENAI_API_KEY` is set.

- [ ] **Step 2: Verify working-tree scope**

```bash
git status --short
git diff --check
git log --oneline -5
```

Expected: no whitespace errors; only intentional pre-existing untracked or user-owned changes remain; recent commits correspond to the OpenAI-only conversion.

- [ ] **Step 3: Review success criteria**

Confirm from test and audit output:

- all LLM clients use `OPENAI_API_KEY`
- all LLM clients default to `gpt-5.6-terra`
- model overrides remain available
- alternate provider selection and routing are absent
- retrieval backends remain unchanged
- Chat Completions remains the request API
