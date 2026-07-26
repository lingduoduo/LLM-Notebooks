import importlib.util
import sys
import types
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


def test_simple_smoke_query_requires_explicit_opt_in(monkeypatch):
    query_calls = []

    class FakeConfig:
        llm = types.SimpleNamespace(model="gpt-5.6-terra")
        knowledge_base = types.SimpleNamespace(type="local")
        chunking = types.SimpleNamespace(chunk_size=1024)

        @classmethod
        def from_env(cls):
            return cls()

    class FakeAgent:
        def __init__(self, config):
            self.model = config.llm.model

        def query_non_agentic(self, *args, **kwargs):
            query_calls.append((args, kwargs))
            return "not used"

    class FakeKnowledgeBaseTools:
        def __init__(self, config):
            pass

        def add_document(self, *args, **kwargs):
            pass

        def get_document(self, document_id):
            return {"doc_id": document_id}

    class FakeDocumentChunker:
        def __init__(self, config):
            pass

        def chunk_text(self, text, document_id):
            return [{"text": text}]

    monkeypatch.setitem(sys.modules, "config", types.SimpleNamespace(Config=FakeConfig, KnowledgeBaseType=object))
    monkeypatch.setitem(sys.modules, "agent", types.SimpleNamespace(AgenticRAG=FakeAgent))
    monkeypatch.setitem(sys.modules, "tools", types.SimpleNamespace(KnowledgeBaseTools=FakeKnowledgeBaseTools))
    monkeypatch.setitem(
        sys.modules,
        "chunking",
        types.SimpleNamespace(DocumentChunker=FakeDocumentChunker, DocumentIndexer=object),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("RUN_LIVE_SMOKE_TESTS", raising=False)

    spec = importlib.util.spec_from_file_location("test_simple_smoke", ROOT / "test_simple.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.test_basic_functionality() is True
    assert query_calls == []


def test_contextual_indexer_uses_environment_config_by_default(monkeypatch):
    import index_local_laws_contextual as indexing

    captured = {}

    class FakeIndexer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def process_all_documents(self, **kwargs):
            captured["process_args"] = kwargs

    monkeypatch.setattr(indexing, "ContextualLegalIndexer", FakeIndexer)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("LLM_MAX_TOKENS", "150")
    monkeypatch.setattr(sys, "argv", ["index_local_laws_contextual.py"])

    indexing.main()

    llm = captured["llm_config"]
    assert llm.get_api_key() == "test-key"
    assert llm.model == "gpt-4.1-mini"
    assert llm.temperature == 0.3
    assert llm.max_tokens == 150


def test_contextual_indexer_cli_model_overrides_environment(monkeypatch):
    import index_local_laws_contextual as indexing

    captured = {}

    class FakeIndexer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def process_all_documents(self, **kwargs):
            pass

    monkeypatch.setattr(indexing, "ContextualLegalIndexer", FakeIndexer)
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setattr(
        sys,
        "argv",
        ["index_local_laws_contextual.py", "--llm-model", "gpt-5.6-terra"],
    )

    indexing.main()

    assert captured["llm_config"].model == "gpt-5.6-terra"
