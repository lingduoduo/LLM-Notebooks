"""Vector index over CrewAI tool objects for semantic tool retrieval.

Mirrors the LlamaIndex ObjectIndex pattern:
  index = ToolIndex(tools=ALL_TOOLS)
  index.build(client)                     # embed tool name + description
  relevant = index.retrieve(question)     # top-k by cosine similarity

This lets an agent receive only the tools that are semantically closest to the
question rather than the full list, reducing context size and decision errors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import openai


class ToolIndex:
    """Cosine-similarity index over a list of CrewAI tool objects."""

    def __init__(
        self,
        tools: Sequence,
        top_k: int = 3,
        model: str = "text-embedding-3-small",
    ) -> None:
        if not tools:
            raise ValueError("ToolIndex requires at least one tool.")

        self.tools: list[Any] = list(tools)
        self.top_k = max(1, min(top_k, len(self.tools)))
        self._embeddings: list[list[float]] = []
        self._model = model

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, client: openai.OpenAI) -> None:
        """Embed every tool's name + description and cache the vectors."""
        texts = [self._tool_text(t) for t in self.tools]
        response = client.embeddings.create(model=self._model, input=texts)
        self._embeddings = [e.embedding for e in response.data]

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, client: openai.OpenAI) -> list[Any]:
        """Return the top-k tools most similar to *query*."""
        ranked = self.rank(query, client)
        return [self.tools[i] for i in ranked[: self.top_k]]

    def retrieve_and_explain(
        self, query: str, client: openai.OpenAI
    ) -> tuple[list[Any], str]:
        """Return (top-k tools, score table) in a single embedding API call."""
        scores = self.score(query, client)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        retrieved = [self.tools[i] for i in ranked_idx[: self.top_k]]
        explain = self.format_explain([(self.tools[i], scores[i]) for i in ranked_idx])
        return retrieved, explain

    def rank(self, query: str, client: openai.OpenAI) -> list[int]:
        """Return tool indexes ranked by similarity to *query*."""
        scores = self.score(query, client)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def score(self, query: str, client: openai.OpenAI) -> list[float]:
        """Return cosine similarity scores for all indexed tools."""
        if not self._embeddings:
            raise RuntimeError("Call build() before score().")
        q_vec = client.embeddings.create(
            model=self._model, input=[query]
        ).data[0].embedding
        return [_cosine(q_vec, e) for e in self._embeddings]

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def explain(self, query: str, client: openai.OpenAI) -> str:
        """Return a ranked table of tools and their similarity scores."""
        scores = self.score(query, client)
        ranked = sorted(zip(self.tools, scores), key=lambda x: x[1], reverse=True)
        return self.format_explain(ranked)

    def format_explain(self, ranked: Sequence[tuple[object, float]]) -> str:
        lines = ["Tool Retrieval Scores", "─" * 40]
        for rank, (tool, score) in enumerate(ranked, 1):
            marker = " ◀ selected" if rank <= self.top_k else ""
            lines.append(f"  {rank}. {tool.name:<30} {score:.4f}{marker}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_text(tool: Any) -> str:
        return f"{tool.name}: {tool.description}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
