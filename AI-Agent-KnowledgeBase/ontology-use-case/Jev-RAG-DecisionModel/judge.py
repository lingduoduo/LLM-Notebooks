from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from tree import Node


class Judge(Protocol):
    def binary(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]: ...

    def choice(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]: ...


class FakeJudge:
    """Offline judge: fixed scores, or word overlap between description and query."""

    def __init__(self, scores: Mapping[str, float] | None = None):
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def binary(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]:
        self.calls.append(("binary", tuple(n.id for n in candidates)))
        return {n.id: self._score(query, n) for n in candidates}

    def choice(self, query: str, candidates: Sequence[Node], question: str) -> dict[str, float]:
        self.calls.append(("choice", tuple(n.id for n in candidates)))
        if not candidates:
            return {}
        scores = {n.id: self._score(query, n) for n in candidates}
        total = sum(scores.values())
        if total == 0:
            return {i: 1 / len(scores) for i in scores}
        return {i: s / total for i, s in scores.items()}

    def _score(self, query: str, node: Node) -> float:
        if self.scores is not None:
            return self.scores.get(node.id, 0.0)
        words = set(node.description.lower().split())
        return len(words & set(query.lower().split())) / len(words) if words else 0.0
