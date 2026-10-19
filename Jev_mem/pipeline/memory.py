"""Standalone typed memory graph with bounded retrieval and JSON persistence.

This is a local implementation inspired by the notebooks, not upstream Jev-Mem.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from .data import normalize_observations
from .providers import LexicalEncoder, LocalDecisions, cosine

KINDS = ("semantic", "temporal", "causal", "entity")


@dataclass(frozen=True)
class RetrievalConfig:
    anchor_count: int = 5
    max_nodes: int = 15
    max_depth: int = 3
    graph_budget: int = 20
    candidate_count: int = 10
    relation_threshold: float = 0.6

    def __post_init__(self):
        for name in ("anchor_count", "max_nodes", "candidate_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("max_depth", "graph_budget"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.anchor_count > self.max_nodes:
            raise ValueError("anchor_count exceeds max_nodes")
        if (
            type(self.relation_threshold) not in (int, float)
            or not math.isfinite(self.relation_threshold)
            or not 0 <= self.relation_threshold <= 1
        ):
            raise ValueError("relation_threshold must be in [0,1]")


def allocate_budget(needs, total):
    """Largest-remainder allocation; deterministic ties; never exceeds total."""
    weights = [max(0, needs.get(k, 0)) for k in KINDS]
    scale = sum(weights)
    if not scale:
        return dict.fromkeys(KINDS, 0)
    quotas = [total * w / scale for w in weights]
    values = [math.floor(v) for v in quotas]
    order = sorted(range(len(KINDS)), key=lambda i: (-(quotas[i] - values[i]), i))
    for i in order[: total - sum(values)]:
        values[i] += 1
    return dict(zip(KINDS, values))


class Memory:
    version = 1

    def __init__(self, encoder=None, decisions=None, config=None, audit_path=None):
        self.encoder = encoder or LexicalEncoder()
        self.decisions = decisions or LocalDecisions()
        self.config = config or RetrievalConfig()
        self.audit_path = Path(audit_path) if audit_path else None
        self.nodes = []
        self.edges = []

    def _audit(self, event, **fields):
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, **fields}, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                )

    def _decide(self, operation, state, questions):
        result = self.decisions.decide(operation, state, questions)
        if set(result) != set(questions) or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in result.values()
        ):
            raise ValueError(
                "Decision provider must return every requested probability in [0,1]"
            )
        record = (
            self.decisions.calls[-1] if getattr(self.decisions, "calls", []) else {}
        )
        self._audit(
            "decision",
            operation=operation,
            source=self.decisions.source,
            decisions=result,
            usage=record.get("usage", {}),
        )
        return result

    def build(self, observations):
        items = normalize_observations(
            observations
        )  # Validate the full batch before any provider call.
        nodes = list(self.nodes)
        edges = [dict(e) for e in self.edges if e["kind"] != "temporal"]
        vectors = self.encoder.encode([o["content"] for o in items])
        if len(vectors) != len(items):
            raise ValueError("Encoder returned the wrong number of vectors")
        for item, vector in zip(items, vectors):
            if not vector or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in vector
            ):
                raise ValueError("Encoder returned an invalid vector")
            index = len(nodes)
            entities = item["metadata"].get("entities", [])
            if not entities:
                entities = sorted(
                    set(re.findall(r"\b[A-Z][a-z]{2,}\b", item["content"]))
                )
            memory_type = self._decide(
                "memory_type",
                {
                    "observation": item,
                    "heuristics": {
                        "episodic": 0.8 if item["timestamp"] else 0.3,
                        "semantic": 0.8,
                        "procedural": 0.8
                        if re.search(r"步骤|先.*再|first.*then", item["content"])
                        else 0.1,
                        "preference": 0.8
                        if re.search(r"喜欢|倾向|偏好|like|prefer", item["content"])
                        else 0.1,
                    },
                },
                {
                    "episodic": "Does this describe an event or experience?",
                    "semantic": "Does this contain factual information?",
                    "procedural": "Does this describe steps or instructions?",
                    "preference": "Does this state a preference?",
                },
            )
            node = {
                **item,
                "id": index,
                "vector": list(vector),
                "entities": entities,
                "memory_type": memory_type,
            }
            candidates = sorted(
                nodes, key=lambda n: (-cosine(vector, n["vector"]), n["id"])
            )[: self.config.candidate_count]
            questions = {}
            hints = {}
            for i, other in enumerate(candidates):
                similarity = max(0, cosine(vector, other["vector"]))
                shared = bool(set(entities) & set(other["entities"]))
                hints.update(
                    {
                        f"{i}_semantic": similarity,
                        f"{i}_entity": 0.9 if shared else 0.05,
                        f"{i}_causes": 0.75
                        if similarity > 0.2
                        and re.search(r"因为|导致|because|caused", item["content"])
                        else 0.05,
                        f"{i}_caused_by": 0.05,
                    }
                )
                questions.update(
                    {
                        f"{i}_semantic": f"Is candidate {i} semantically related to the new observation?",
                        f"{i}_entity": f"Do candidate {i} and the new observation refer to the same entity?",
                        f"{i}_causes": f"Does the new observation cause candidate {i}?",
                        f"{i}_caused_by": f"Does candidate {i} cause the new observation?",
                    }
                )
            if candidates:
                relations = self._decide(
                    "relations",
                    {
                        "observation": item,
                        "candidates": [self._public(n) for n in candidates],
                        "heuristics": hints,
                    },
                    questions,
                )
                for i, other in enumerate(candidates):
                    for kind in ("semantic", "entity", "causes", "caused_by"):
                        score = relations[f"{i}_{kind}"]
                        if score < self.config.relation_threshold:
                            continue
                        source, target = (
                            (other["id"], index)
                            if kind == "caused_by"
                            else (index, other["id"])
                        )
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "kind": "causal"
                                if kind in ("causes", "caused_by")
                                else kind,
                                "confidence": score,
                            }
                        )
            nodes.append(node)
        ordered = sorted(
            (n for n in nodes if n["timestamp"]),
            key=lambda n: (n["timestamp"], n["id"]),
        )
        for a, b in zip(ordered, ordered[1:]):
            edges.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "kind": "temporal",
                    "confidence": 1.0,
                }
            )
        self.nodes, self.edges = nodes, edges
        self._audit(
            "memory_built", observations=len(items), nodes=len(nodes), edges=len(edges)
        )
        return {"admitted": len(items), "nodes": len(nodes), "edges": len(edges)}

    @staticmethod
    def _public(node):
        return {k: v for k, v in node.items() if k != "vector"}

    @staticmethod
    def format_evidence(nodes):
        lines = []
        for node in nodes:
            speaker = node["metadata"].get("speaker", node["role"])
            source = node["metadata"].get("dia_id") or node.get("id", "")
            lines.append(
                f"[{node['timestamp'] or 'undated'}] [{source}] {speaker}: {node['content']}"
            )
        return "\n".join(lines)

    def retrieve(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be nonempty")
        if not self.nodes:
            return {
                "context": "",
                "evidence": [],
                "diagnostics": {
                    "expanded_edges": 0,
                    "depth": 0,
                    "stop_reason": "empty",
                },
            }
        query_vector = self.encoder.encode(question)[0]
        scores = {
            n["id"]: max(0, cosine(query_vector, n["vector"])) for n in self.nodes
        }
        ranked = sorted(self.nodes, key=lambda n: (-scores[n["id"]], n["id"]))
        selected = {
            n["id"] for n in ranked[: self.config.anchor_count] if scores[n["id"]] > 0
        }
        route = self._decide(
            "routing",
            {
                "question": question,
                "heuristics": {
                    "semantic": 0.9,
                    "entity": 0.7,
                    "temporal": 0.9
                    if re.search(
                        r"when|before|after|earliest|latest|哪天|时间|先|最早",
                        question,
                        re.I,
                    )
                    else 0.1,
                    "causal": 0.9
                    if re.search(r"why|cause|为什么|原因", question, re.I)
                    else 0.1,
                },
            },
            {
                k: f"How useful is {k} graph traversal for answering this question?"
                for k in KINDS
            },
        )
        budgets = allocate_budget(route, self.config.graph_budget)
        used = dict.fromkeys(KINDS, 0)
        visited_edges = set()
        frontier = set(selected)
        depth = 0
        stop_reason = "depth_limit"
        for depth in range(1, self.config.max_depth + 1):
            if not frontier or len(selected) >= self.config.max_nodes:
                stop_reason = "no_frontier" if not frontier else "node_limit"
                break
            candidates = {}
            for i, edge in enumerate(self.edges):
                kind = edge["kind"]
                if i in visited_edges or used[kind] >= budgets[kind]:
                    continue
                a, b = edge["source"], edge["target"]
                if a not in frontier and b not in frontier:
                    continue
                target = b if a in frontier else a
                visited_edges.add(i)
                used[kind] += 1
                if target not in selected:
                    candidates[target] = max(
                        candidates.get(target, 0), edge["confidence"]
                    )
            if not candidates:
                stop_reason = "budget_or_graph_exhausted"
                break
            relevance = self._decide(
                "traversal",
                {
                    "question": question,
                    "candidates": [self._public(self.nodes[i]) for i in candidates],
                    "heuristics": {
                        str(i): min(1, 0.8 * scores[i] + 0.2 * candidates[i])
                        for i in candidates
                    },
                },
                {
                    str(
                        i
                    ): f"Is memory {i} relevant and useful for answering the question?"
                    for i in candidates
                },
            )
            order = sorted(
                candidates, key=lambda i: (-relevance[str(i)], -scores[i], i)
            )
            frontier = set(order[: self.config.max_nodes - len(selected)])
            frontier = {i for i in frontier if relevance[str(i)] >= 0.15}
            selected.update(frontier)
            stop = self._decide(
                "stopping",
                {
                    "question": question,
                    "evidence": [self._public(self.nodes[i]) for i in sorted(selected)],
                    "depth": depth,
                    "heuristics": {
                        "sufficient": 0.9
                        if not frontier or len(selected) >= self.config.max_nodes
                        else 0.2
                    },
                },
                {
                    "sufficient": "Is the retrieved evidence sufficient to answer without further traversal?"
                },
            )
            if stop["sufficient"] >= 0.85:
                stop_reason = "sufficient"
                break
        evidence = [
            self._public(self.nodes[i])
            for i in sorted(
                selected, key=lambda i: (self.nodes[i]["timestamp"] or "", i)
            )
        ]
        diagnostics = {
            "expanded_edges": sum(used.values()),
            "budgets": budgets,
            "used_budgets": used,
            "depth": depth,
            "stop_reason": stop_reason,
            "decision_source": self.decisions.source,
        }
        self._audit(
            "retrieval",
            question=question,
            evidence_ids=[n["id"] for n in evidence],
            **diagnostics,
        )
        return {
            "context": self.format_evidence(evidence),
            "evidence": evidence,
            "diagnostics": diagnostics,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "encoder": self.encoder.fingerprint,
            "dimension": self.encoder.dimension,
            "config": asdict(self.config),
            "nodes": self.nodes,
            "edges": self.edges,
            "decision_source": self.decisions.source,
        }
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        envelope = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "memory": payload,
        }
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, encoding="utf-8", delete=False
        ) as stream:
            json.dump(envelope, stream, ensure_ascii=False, allow_nan=False)
            tmp = Path(stream.name)
        tmp.replace(directory / "memory.json")

    @classmethod
    def load(cls, directory, encoder=None, decisions=None, audit_path=None):
        envelope = json.loads(
            (Path(directory) / "memory.json").read_text(encoding="utf-8")
        )
        payload = envelope["memory"]
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        if hashlib.sha256(text.encode()).hexdigest() != envelope["sha256"]:
            raise ValueError("Memory checksum mismatch")
        if payload["version"] != cls.version:
            raise ValueError("Unsupported memory version")
        encoder = encoder or LexicalEncoder()
        if encoder.fingerprint != payload["encoder"]:
            raise ValueError("Incompatible encoder for stored memory")
        if encoder.dimension is not None and encoder.dimension != payload["dimension"]:
            raise ValueError("Incompatible encoder dimension")
        encoder.dimension = payload["dimension"]
        result = cls(
            encoder, decisions, RetrievalConfig(**payload["config"]), audit_path
        )
        nodes = payload["nodes"]
        edges = payload["edges"]
        normalize_observations(nodes)
        for i, node in enumerate(nodes):
            if (
                node["id"] != i
                or len(node["vector"]) != encoder.dimension
                or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in node["vector"]
                )
            ):
                raise ValueError("Invalid stored memory vector or ID")
        for edge in edges:
            if edge["kind"] not in KINDS or any(
                type(edge[k]) is not int or not 0 <= edge[k] < len(nodes)
                for k in ("source", "target")
            ):
                raise ValueError("Invalid stored graph edge")
        result.nodes, result.edges = nodes, edges
        return result
