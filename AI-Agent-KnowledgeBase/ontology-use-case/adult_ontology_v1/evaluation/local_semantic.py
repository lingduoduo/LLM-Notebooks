"""Zero-shot cosine-similarity baseline using taxonomy-only concept prototypes."""

import math
from typing import Iterable, Sequence

from utils import validate_taxonomy

DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_LABELS = 3


def build_prototypes(taxonomy: dict) -> list:
    """Use labels and parent paths only; no vocabulary, benchmark text or gold."""
    validate_taxonomy(taxonomy)
    nodes = {node["id"]: node for node in taxonomy["nodes"]}
    prototypes = []
    for concept_id, node in sorted(nodes.items()):
        if node["level"] == 0 or node["category_id"] == "IDENTITY":
            continue
        path = []
        current = concept_id
        while current is not None:
            parent = nodes[current]
            if parent["level"] > 0:
                path.append(parent["label"])
            current = parent["parent_id"]
        text = f"Room feature: {node['label']}. Taxonomy: {' > '.join(reversed(path))}."
        prototypes.append({"concept_id": concept_id, "text": text})
    return prototypes


def _normalize_vectors(vectors, expected_count, dimension=None):
    vectors = list(vectors)
    if len(vectors) != expected_count:
        raise ValueError("Encoder returned the wrong number of vectors")
    normalized = []
    for vector in vectors:
        values = [float(value) for value in vector]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Encoder returned an empty or non-finite vector")
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension:
            raise ValueError("Encoder dimensions differ")
        length = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(length) or length == 0:
            raise ValueError("Encoder returned an invalid vector norm")
        normalized.append([value / length for value in values])
    return normalized, dimension


def validate_settings(threshold, max_labels) -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not -1 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and between -1 and 1")
    if type(max_labels) is not int or max_labels <= 0:
        raise ValueError("max_labels must be a positive integer")


class SemanticBaseline:
    """Prepared prototypes; candidate similarities are not class probabilities."""

    def __init__(
        self,
        taxonomy: dict,
        encoder,
        *,
        threshold=DEFAULT_THRESHOLD,
        max_labels=DEFAULT_MAX_LABELS,
        prototypes=None,
    ):
        validate_taxonomy(taxonomy)
        validate_settings(threshold, max_labels)
        self.version = taxonomy["version"]
        # Copy metadata so later caller mutations cannot change prepared prototypes.
        self.nodes = {node["id"]: dict(node) for node in taxonomy["nodes"]}
        self.encoder = encoder
        self.threshold = float(threshold)
        self.max_labels = max_labels
        self.prototypes = [
            dict(item)
            for item in (
                build_prototypes(taxonomy) if prototypes is None else prototypes
            )
        ]
        ids = [p.get("concept_id") for p in self.prototypes]
        if (
            not ids
            or any(
                not isinstance(c, str)
                or c not in self.nodes
                or self.nodes[c]["level"] == 0
                or self.nodes[c]["category_id"] == "IDENTITY"
                for c in ids
            )
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(p.get("text"), str) or not p["text"].strip()
                for p in self.prototypes
            )
        ):
            raise ValueError(
                "Prototypes require unique known, non-identity L1–L3 concepts and text"
            )
        self.vectors, self.dimension = _normalize_vectors(
            encoder.encode([p["text"] for p in self.prototypes]), len(self.prototypes)
        )
        self.ancestors = {}
        for concept_id in ids:
            ancestors = set()
            parent = self.nodes[concept_id]["parent_id"]
            while parent is not None:
                ancestors.add(parent)
                parent = self.nodes[parent]["parent_id"]
            self.ancestors[concept_id] = ancestors

    def predict(
        self,
        subject: str,
        tags: Sequence[str],
        *,
        self_declared_concepts: Iterable[str] = (),
    ) -> dict:
        if (
            not isinstance(subject, str)
            or not isinstance(tags, (list, tuple))
            or any(not isinstance(t, str) for t in tags)
        ):
            raise ValueError("subject and tags must contain strings")
        if isinstance(self_declared_concepts, str):
            raise ValueError("self_declared_concepts must be a collection of IDs")
        declarations = list(self_declared_concepts)
        if any(
            not isinstance(c, str)
            or c not in self.nodes
            or self.nodes[c]["category_id"] != "IDENTITY"
            for c in declarations
        ):
            raise ValueError("Declarations must reference known IDENTITY concepts")
        fields = [("subject", subject)] + [
            (f"tag_{i+1}", t) for i, t in enumerate(tags[:5])
        ]
        fields = [(name, text) for name, text in fields if text.strip()]
        scores, evidence = {}, {}
        if fields:
            vectors, _ = _normalize_vectors(
                self.encoder.encode([text for _, text in fields]),
                len(fields),
                self.dimension,
            )
            for prototype, vector in zip(self.prototypes, self.vectors):
                similarities = [
                    sum(a * b for a, b in zip(vector, field_vector))
                    for field_vector in vectors
                ]
                best = max(range(len(fields)), key=similarities.__getitem__)
                score = max(-1.0, min(1.0, similarities[best]))
                if score >= self.threshold:
                    concept_id = prototype["concept_id"]
                    scores[concept_id] = score
                    evidence[concept_id] = {
                        "field": fields[best][0],
                        "text": fields[best][1],
                        "source": "embedding_similarity",
                        "prototype": prototype["text"],
                    }
        # Prefer a qualified specific concept over its redundant ancestors.
        ancestors = (
            set().union(*(self.ancestors[c] for c in scores)) if scores else set()
        )
        selected = sorted(
            (c for c in scores if c not in ancestors), key=lambda c: (-scores[c], c)
        )[: self.max_labels]
        for concept_id in declarations:
            evidence[concept_id] = {
                "field": "self_declared_concepts",
                "text": concept_id,
                "source": "self_declared",
            }
        concepts = sorted(set(selected) | set(declarations))
        return {
            "concept_ids": concepts,
            "scores": {c: scores[c] for c in selected},
            "evidence": {c: evidence[c] for c in concepts},
            "requires_context_review": bool(selected),
        }
