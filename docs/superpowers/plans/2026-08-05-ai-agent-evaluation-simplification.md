# AI Agent Evaluation Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the TTS evaluator's pipeline and CLI code without changing observable behavior.

**Architecture:** Keep the existing flat-module structure and public entry points. Consolidate repeated client and rubric mechanics inside `pipeline.py`, decompose `demo.py` into small selection and orchestration helpers, and organize tests by the module behavior they protect.

**Tech Stack:** Python 3.12, OpenAI Python SDK, standard-library `argparse`/`urllib`/`subprocess`, pytest.

## Global Constraints

- Existing imports of `config`, `pipeline`, and `demo` continue to work.
- CLI options and their semantics remain unchanged.
- Cached audio filenames remain stable.
- Successful and failed evaluation record shapes remain unchanged.
- Scoring, sorting, and summary calculations remain unchanged.
- Provider request payloads and authentication behavior remain unchanged.
- No new runtime dependencies are introduced.
- Do not modify or stage the user-owned deletion of the 2026-07-29 design and plan files.
- Do not overwrite or remove the user-owned untracked `Ai-Agent-Evaluation/tts-quality-eval/.gitignore`.

---

### Task 1: Consolidate pipeline client selection and rubric parsing

**Files:**
- Modify: `Ai-Agent-Evaluation/tts-quality-eval/pipeline.py:29-109,364-421,425-502`
- Modify: `Ai-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py:22-52,158-253`

**Interfaces:**
- Consumes: `config.JUDGE_MODEL`, `config.GEMINI_MODEL_DEFAULT`, environment keys, and current OpenAI-compatible/Gemini response shapes.
- Produces: `_new_openai_client(api_key: str, base_url: str | None = None) -> OpenAI`, `_rubric_from_json(raw: str) -> RubricResult`, and `_gemini_response_text(data: object) -> str`; retains `get_client()`, `get_judge_client_and_model()`, `parse_rubric_response()`, `judge_rubric()`, and `judge_gemini_audio()` signatures.

- [ ] **Step 1: Add characterization tests for client caching and shared judge parsing**

Add tests that replace `pipeline.OpenAI` with a recording fake and reset the module caches. Assert that direct OpenAI clients are reused, switching judge backends constructs the correct client, native OpenRouter IDs remain unchanged, and both judge paths call the same JSON-to-rubric helper:

```python
def test_get_judge_client_reuses_matching_backend(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(pipeline, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(pipeline, "_judge_client", None)
    monkeypatch.setattr(pipeline, "_judge_client_kind", "")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    first, first_model = pipeline.get_judge_client_and_model("gpt-4.1")
    second, second_model = pipeline.get_judge_client_and_model("gpt-4.1-mini")

    assert first is second
    assert (first_model, second_model) == ("gpt-4.1", "gpt-4.1-mini")
    assert created == [{"api_key": "openai-key", "max_retries": 5, "timeout": 60.0}]


def test_rubric_from_json_rejects_non_object_root():
    with pytest.raises(RuntimeError, match="JSON object"):
        pipeline._rubric_from_json("[]")


def test_gemini_response_text_returns_first_text_part():
    data = {"candidates": [{"content": {"parts": [{"text": "rubric-json"}]}}]}
    assert pipeline._gemini_response_text(data) == "rubric-json"
```

- [ ] **Step 2: Run the new focused tests and confirm the helpers do not exist yet**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_judge_robustness.py -k 'client_reuses or rubric_from_json or gemini_response_text'
```

Expected: failures identifying missing `_rubric_from_json` and `_gemini_response_text` helpers; the pre-existing client behavior test may already pass.

- [ ] **Step 3: Extract small internal helpers while retaining public APIs**

Implement a single constructor helper and explicit response-extraction helpers. Keep the existing cache variables so external tests and callers are not disrupted:

```python
def _new_openai_client(api_key: str, base_url: str | None = None) -> OpenAI:
    kwargs = {"api_key": api_key, "max_retries": 5, "timeout": 60.0}
    if base_url is not None:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _rubric_from_json(raw: str) -> RubricResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge returned invalid JSON: {raw[:300]}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Judge returned JSON object expected: {raw[:300]}")
    return parse_rubric_response(data, raw)


def _gemini_response_text(data: object) -> str:
    if isinstance(data, dict):
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            content = candidate.get("content") if isinstance(candidate, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
    raise RuntimeError(f"Gemini did not return evaluation text: {str(data)[:300]}")
```

Use `_new_openai_client()` in `get_client()` and `get_judge_client_and_model()`. Make `judge_rubric()` pass message content to `_rubric_from_json()`. Make `judge_gemini_audio()` extract text with `_gemini_response_text()` and parse it with `_rubric_from_json()`. Do not alter prompts, request payloads, timeouts, model mapping, score validation, or exception ownership.

- [ ] **Step 4: Run pipeline tests and the full baseline suite**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_judge_robustness.py
```

Expected: all tests pass, including null, malformed, blocked, and valid judge-response cases.

- [ ] **Step 5: Commit the pipeline simplification**

```bash
git add Ai-Agent-Evaluation/tts-quality-eval/pipeline.py Ai-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py
git commit -m "refactor: simplify TTS judge pipeline"
```

---

### Task 2: Decompose CLI selection and orchestration

**Files:**
- Modify: `Ai-Agent-Evaluation/tts-quality-eval/demo.py:162-274`
- Create: `Ai-Agent-Evaluation/tts-quality-eval/test_demo.py`

**Interfaces:**
- Consumes: `config.PROVIDER_CONFIGS`, `config.TTS_CONFIGS`, `config.EXTRA_CONFIGS`, `config.CORPUS`, and existing evaluation/report functions.
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace`, `select_configs(provider_names: str | None, include_extra: bool) -> list[config.TTSConfig]`, `select_corpus(text: str | None, quick: bool) -> list[config.Sample]`, and `run_evaluations(configs, corpus, use_gemini, fresh, judge_model) -> list[dict]`; retains `main() -> None`.

- [ ] **Step 1: Characterize configuration and corpus selection**

Create `test_demo.py` with small deterministic tests:

```python
import pytest

import config
import demo


def test_select_configs_uses_defaults_and_optional_extra():
    assert demo.select_configs(None, False) == config.TTS_CONFIGS
    assert demo.select_configs(None, True) == config.TTS_CONFIGS + config.EXTRA_CONFIGS


def test_select_configs_preserves_requested_provider_order():
    configs = demo.select_configs("minimax, openai", False)
    assert [item.provider for item in configs] == ["minimax", "openai"]


def test_select_configs_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider 'missing'"):
        demo.select_configs("openai,missing", False)


def test_select_corpus_supports_custom_and_quick_modes():
    custom = demo.select_corpus("Hello world", quick=True)
    assert custom == [config.Sample("custom", "Hello world", "custom text", "neutral")]
    assert demo.select_corpus(None, quick=True) == config.CORPUS[:2]
    assert demo.select_corpus(None, quick=False) == config.CORPUS
```

- [ ] **Step 2: Run selection tests and verify they fail for missing helpers**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_demo.py
```

Expected: failures because `select_configs()` and `select_corpus()` are not defined.

- [ ] **Step 3: Extract argument parsing and pure selectors**

Move parser construction from `main()` into `parse_args()`, preserving every option, destination, default, description, and epilog. Add pure selection helpers:

```python
def select_configs(provider_names: str | None, include_extra: bool) -> list[config.TTSConfig]:
    if not provider_names:
        return list(config.TTS_CONFIGS) + (list(config.EXTRA_CONFIGS) if include_extra else [])

    selected = []
    for name in (item.strip() for item in provider_names.split(",")):
        if not name:
            continue
        try:
            selected.append(config.PROVIDER_CONFIGS[name])
        except KeyError:
            available = ", ".join(config.PROVIDER_CONFIGS)
            raise ValueError(f"unknown provider {name!r}. Available: {available}") from None
    return selected


def select_corpus(text: str | None, quick: bool) -> list[config.Sample]:
    if text:
        return [config.Sample("custom", text, "custom text", "neutral")]
    return list(config.CORPUS[:2] if quick else config.CORPUS)
```

In `main()`, translate `ValueError` from `select_configs()` to the same stderr message and exit code currently produced. Keep offline commands before credential validation and output-directory creation.

- [ ] **Step 4: Characterize evaluation-loop delegation**

Add a test proving the extracted runner preserves order and forwards judge arguments:

```python
def test_run_evaluations_preserves_grid_order_and_arguments(monkeypatch):
    configs = [config.TTSConfig("a", "m", "v"), config.TTSConfig("b", "m", "v")]
    corpus = [config.Sample("one", "One", "fixture"), config.Sample("two", "Two", "fixture")]
    calls = []

    def fake_evaluate(cfg, sample, use_gemini, fresh, judge_model):
        calls.append((cfg.name, sample.id, use_gemini, fresh, judge_model))
        return {"config": cfg.name, "sample": sample.id, "ok": True}

    monkeypatch.setattr(demo, "evaluate_one", fake_evaluate)
    monkeypatch.setattr(demo, "print_detail", lambda *_: None)

    records = demo.run_evaluations(configs, corpus, True, True, None)

    assert [(r["config"], r["sample"]) for r in records] == [
        ("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")
    ]
    assert calls == [
        ("a", "one", True, True, None), ("a", "two", True, True, None),
        ("b", "one", True, True, None), ("b", "two", True, True, None),
    ]
```

- [ ] **Step 5: Run the runner test and verify it fails for the missing helper**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_demo.py::test_run_evaluations_preserves_grid_order_and_arguments
```

Expected: failure because `run_evaluations()` is not defined.

- [ ] **Step 6: Extract the evaluation runner and shorten `main()`**

Move the nested configuration/sample loop into `run_evaluations()`. Preserve configuration header text, record order, `print_detail()` calls, and the `None` judge model passed in Gemini mode. Leave `main()` responsible only for loading environment variables, handling offline exits, validating credentials, selecting inputs, printing the run header, invoking the runner, summarizing, writing JSON, and printing elapsed/cost information.

```python
def run_evaluations(
    configs: list[config.TTSConfig],
    corpus: list[config.Sample],
    use_gemini: bool,
    fresh: bool,
    judge_model: str | None,
) -> list[dict]:
    records = []
    for cfg in configs:
        print(f"\n### Configuration {cfg.name}  (provider={cfg.provider}, "
              f"model={cfg.model}, voice={cfg.voice}, speed={cfg.speed})")
        for sample in corpus:
            record = evaluate_one(cfg, sample, use_gemini, fresh, judge_model)
            print_detail(record, sample.text)
            records.append(record)
    return records
```

- [ ] **Step 7: Run CLI unit tests and offline compatibility checks**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_demo.py test_judge_robustness.py
python demo.py --help
python demo.py --list-providers
python demo.py --dump-rubric
```

Expected: all pytest tests pass; all three CLI commands exit 0; help still lists all current flags; provider and rubric output remains available without API keys.

- [ ] **Step 8: Commit the CLI simplification**

```bash
git add Ai-Agent-Evaluation/tts-quality-eval/demo.py Ai-Agent-Evaluation/tts-quality-eval/test_demo.py
git commit -m "refactor: simplify TTS evaluation CLI"
```

---

### Task 3: Tighten configuration types and complete regression verification

**Files:**
- Modify: `Ai-Agent-Evaluation/tts-quality-eval/config.py:34-164`
- Modify: `Ai-Agent-Evaluation/tts-quality-eval/test_demo.py`
- Modify only if commands changed: `Ai-Agent-Evaluation/tts-quality-eval/README.md`

**Interfaces:**
- Consumes: all configuration constants and dataclass construction used by `pipeline.py` and `demo.py`.
- Produces: precisely typed `TTSConfig`, `ProviderInfo`, and `Sample` value objects with explicit `tuple[str, ...]` and registry collection types; all existing mutability, field names, and constructor ordering remain unchanged.

- [ ] **Step 1: Add tests for precise configuration types and environment aliases**

Append:

```python
from typing import get_type_hints


def test_provider_environment_names_have_precise_type():
    assert get_type_hints(config.ProviderInfo)["env"] == tuple[str, ...]


def test_env_get_uses_first_nonempty_alias(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "  ")
    monkeypatch.setenv("FISHAUDIO_API_KEY", " legacy-key ")
    assert config.env_get("FISH_API_KEY") == "legacy-key"
```

- [ ] **Step 2: Run the focused tests and confirm the precise type check fails**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q test_demo.py -k 'provider_environment_names or env_get'
```

Expected: alias test passes; the type test fails because `ProviderInfo.env` is currently an unparameterized tuple.

- [ ] **Step 3: Make configuration collection types explicit**

Keep the dataclasses mutable and type provider environment names as `tuple[str, ...]`. Add explicit built-in generic types to registries and lists where the annotation removes ambiguity:

```python
@dataclass
class TTSConfig:
    name: str
    model: str
    voice: str
    speed: float = 1.0
    provider: str = "openai"


@dataclass
class ProviderInfo:
    key: str
    label: str
    env: tuple[str, ...]
    note: str


@dataclass
class Sample:
    id: str
    text: str
    challenge: str
    emotion: str = "neutral"
```

Do not change dataclass mutability, rename fields, reorder constructor arguments, change registry contents, or introduce enums/classes for providers.

- [ ] **Step 4: Run complete automated and static smoke verification**

Run:

```bash
cd Ai-Agent-Evaluation/tts-quality-eval
pytest -q
python -m compileall -q config.py pipeline.py demo.py test_demo.py test_judge_robustness.py
python demo.py --help
python demo.py --list-providers
python demo.py --dump-rubric
git diff --check
```

Expected: all tests pass; compilation and offline CLI checks exit 0; `git diff --check` prints nothing. Do not run live synthesis or judging because it would require credentials and make paid external calls.

- [ ] **Step 5: Confirm documentation still matches behavior**

Compare the README command list, output paths, provider descriptions, judge paths, and robustness notes against the refactored code. If no user-visible behavior changed, leave `README.md` untouched. If an existing statement is proven inaccurate during verification, make only the smallest correction and rerun `git diff --check`.

- [ ] **Step 6: Commit configuration cleanup and any necessary documentation correction**

```bash
git add Ai-Agent-Evaluation/tts-quality-eval/config.py Ai-Agent-Evaluation/tts-quality-eval/test_demo.py
git add Ai-Agent-Evaluation/tts-quality-eval/README.md  # only if Step 5 required a correction
git commit -m "refactor: tighten TTS evaluation configuration"
```
