"""Adapter: auto-built keyword tree (JSON) -> engine Node tree, plus concept scoring."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from navigate import NavigationConfig, Result
from taxonomy_corpus import ONTOLOGY_DIR, taxonomy_config
from tree import Node, walk
from tree_builder import read_tree

HERE = Path(__file__).resolve().parent
KEYWORD_TREE_PATH = HERE / "trees" / "keyword_tree.json"
KEYWORDS_CSV = ONTOLOGY_DIR / "keywords.csv"


def _to_node(d: dict) -> Node:
    children = d.get("children")
    if children is None and "keyword" in d:
        return Node(d["id"], d["keyword"],
                    payload={"keyword": d["keyword"], "concept_id": d.get("concept_id", "")})
    return Node(d["id"], d["description"], tuple(_to_node(c) for c in children or ()))


def load_keyword_tree(path: Path = KEYWORD_TREE_PATH) -> tuple[dict, Node]:
    data = read_tree(path)  # FileNotFoundError propagates
    return data["meta"], _to_node(data["tree"])


def is_stale(meta: dict, csv_path: Path = KEYWORDS_CSV) -> bool:
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    return digest != meta.get("keywords_sha256")


def keyword_config(**overrides) -> NavigationConfig:
    base = replace(taxonomy_config(), returnable=lambda n: n.is_leaf)
    return replace(base, **overrides)


def to_concepts(result: Result, root: Node) -> tuple[list[str], int]:
    leaves = {n.id: n for n in walk(root) if n.is_leaf}
    concepts: list[str] = []
    unscorable = 0
    for r in result.ranked:
        if r.source == "self_declared":
            concepts.append(r.id)
            continue
        node = leaves.get(r.id)
        cid = node.payload.get("concept_id", "") if node else ""
        if cid:
            concepts.append(cid)
        else:
            unscorable += 1
    return list(dict.fromkeys(concepts)), unscorable
