"""Configuration for Agentic RAG System"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum


class KnowledgeBaseType(str, Enum):
    """Knowledge base backend types"""
    LOCAL = "local"  # Local retrieval pipeline
    DIFY = "dify"    # Dify knowledge base API
    RAPTOR = "raptor"  # RAPTOR tree-based index
    GRAPHRAG = "graphrag"  # GraphRAG graph-based index


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


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base configuration"""
    type: KnowledgeBaseType = KnowledgeBaseType.LOCAL

    # Local retrieval pipeline config
    local_base_url: str = "http://localhost:4242"
    local_top_k: int = 3

    # Dify config
    dify_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DIFY_API_KEY"))
    dify_base_url: str = "https://api.dify.ai/v1"
    dify_dataset_id: Optional[str] = None
    dify_top_k: int = 10

    # RAPTOR tree-based index config
    raptor_base_url: str = "http://localhost:4242"
    raptor_top_k: int = 10
    raptor_search_levels: bool = True  # Search across multiple tree levels

    # GraphRAG graph-based index config
    graphrag_base_url: str = "http://localhost:4242"
    graphrag_top_k: int = 10
    graphrag_search_type: str = "hybrid"  # entity, community, or hybrid

    # Document storage
    document_store_path: str = "document_store.json"


@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    chunk_size: int = 2048  # Characters per chunk
    max_chunk_size: int = 1024  # Max size when respecting paragraph boundaries
    chunk_overlap: int = 200  # Overlap between chunks
    respect_paragraph_boundary: bool = True
    min_chunk_size: int = 100  # Minimum chunk size


@dataclass
class AgentConfig:
    """Agent configuration"""
    max_iterations: int = 10  # Max reasoning iterations
    enable_reasoning_trace: bool = True
    enable_citations: bool = True
    strict_knowledge_base: bool = True  # Only answer from knowledge base
    conversation_history_limit: int = 20  # Max conversation turns to keep
    verbose: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    dataset_path: str = "evaluation/legal_qa_dataset.json"
    results_path: str = "evaluation/results"
    metrics: list = field(default_factory=lambda: ["accuracy", "relevance", "citation_quality"])


@dataclass
class Config:
    """Main configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config = cls()

        # Override from env
        if model := os.getenv("LLM_MODEL"):
            config.llm.model = model
        if temperature := os.getenv("LLM_TEMPERATURE"):
            config.llm.temperature = float(temperature)
        if max_tokens := os.getenv("LLM_MAX_TOKENS"):
            config.llm.max_tokens = int(max_tokens)
        if kb_type := os.getenv("KB_TYPE"):
            config.knowledge_base.type = KnowledgeBaseType(kb_type.lower())

        return config
