#!/usr/bin/env python3
"""Smoke tests for the Agentic RAG system."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def test_basic_functionality(tmp_path):
    """Core components work without writing test data into the worktree."""
    from agent import AgenticRAG
    from chunking import DocumentChunker
    from config import Config
    from tools import KnowledgeBaseTools

    config = Config.from_env()
    config.knowledge_base.document_store_path = str(tmp_path / "document_store.json")
    config.llm.api_key = os.getenv("OPENAI_API_KEY") or "test-key"

    assert config.llm.model
    assert config.chunking.chunk_size > 0

    chunker = DocumentChunker(config.chunking)
    sample_text = """Intentional homicide is the intentional and unlawful deprivation of
    another person's life.

    Under Article 232 of the Criminal Law of the PRC, whoever intentionally kills another
    shall be sentenced to death, life imprisonment or fixed-term imprisonment of ten years
    or more; where the circumstances are relatively minor, to three to ten years.

    Sentencing considers the motive, the means used and the consequences."""
    chunks = chunker.chunk_text(sample_text, "test_doc")
    assert chunks
    assert chunks[0]["text"]

    kb_tools = KnowledgeBaseTools(config.knowledge_base)
    kb_tools.add_document(
        "test_doc_1",
        "Intentional homicide is punishable by death, life imprisonment, or "
        "fixed-term imprisonment of ten years or more.",
        {"source": "test"},
    )
    document = kb_tools.get_document("test_doc_1")
    assert document["doc_id"] == "test_doc_1"

    agent = AgenticRAG(config)
    assert agent.model == config.llm.model

    if os.getenv("OPENAI_API_KEY") and os.getenv("RUN_LIVE_SMOKE_TESTS") == "1":
        kb_tools.add_document(
            "criminal_law_test",
            """Thresholds for filing a theft case:
            1. Relatively large amount: generally 1,000 to 3,000 yuan or more
            2. Repeated theft: three or more thefts within two years
            3. Home-invasion theft, theft while carrying a weapon and pickpocketing count
               regardless of the amount""",
            {"type": "law"},
        )
        response = agent.query_non_agentic("threshold for filing a theft case", stream=False)
        assert response and len(response) > 10


def test_evaluation_dataset():
    """The bundled evaluation data can be constructed."""
    evaluation_dir = str(Path(__file__).parent / "evaluation")
    if evaluation_dir not in sys.path:
        sys.path.append(evaluation_dir)

    from dataset_builder import LegalDatasetBuilder, create_legal_documents

    builder = LegalDatasetBuilder()
    simple_cases = builder.create_simple_cases()
    complex_cases = builder.create_complex_cases()
    documents = create_legal_documents()

    assert simple_cases
    assert complex_cases
    assert documents


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        test_basic_functionality(Path(temp_dir))
    test_evaluation_dataset()
