"""Adapter: adult-content room taxonomy -> generic Node tree, plus room glue."""
from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "adult_ontology_v1"
if str(ONTOLOGY_DIR) not in sys.path:
    sys.path.append(str(ONTOLOGY_DIR))

import utils  # noqa: E402

from navigate import NavigationConfig, Ranked, Result, navigate, score_units  # noqa: E402
from tree import Node, walk  # noqa: E402

ROOT_ID = "__root__"


def build_tree(taxonomy: dict, keyword_rows: Iterable[dict], max_examples: int = 8) -> Node:
    examples: dict[str, list[str]] = {}
    for row in keyword_rows:
        if row["review_status"] == "excluded":
            continue
        examples.setdefault(row["concept_id"], []).append(row["keyword"])

    by_id = {n["id"]: n for n in taxonomy["nodes"]}
    kids: dict[str | None, list[dict]] = {}
    for n in taxonomy["nodes"]:
        kids.setdefault(n["parent_id"], []).append(n)

    def path(n: dict) -> str:
        labels = []
        while n is not None and n["level"] >= 1:
            labels.append(n["label"])
            n = by_id.get(n["parent_id"])
        return " > ".join(reversed(labels))

    def make(n: dict) -> Node:
        desc = f"{n['label']} — {path(n)}"
        ex = examples.get(n["id"], [])[:max_examples]
        if ex:
            desc += f" (e.g. {', '.join(ex)})"
        return Node(n["id"], desc, tuple(make(c) for c in kids.get(n["id"], [])),
                    {"level": n["level"]})

    top = [c for n in taxonomy["nodes"] if n["level"] == 0 for c in kids.get(n["id"], [])]
    top += [n for n in kids.get(None, []) if n["level"] == 1]
    order = {n["id"]: i for i, n in enumerate(taxonomy["nodes"])}
    top.sort(key=lambda n: order[n["id"]])
    return Node(ROOT_ID, "root", tuple(make(n) for n in top))


def is_returnable(node: Node) -> bool:
    return node.is_leaf or node.payload.get("level") == 2


def room_query(subject: str, tags: Sequence[str]) -> str:
    t = ", ".join(tags[:5]) if tags else "(none)"
    return f"Subject: {subject}\nTags: {t}"


def taxonomy_config(**overrides) -> NavigationConfig:
    base = NavigationConfig(
        returnable=is_returnable,
        binary_question="Does this room involve Candidate {n}?",
        choice_question="Which candidate is most relevant to this room?",
    )
    return replace(base, **overrides)


def tag_room(root: Node, judge, config: NavigationConfig, subject: str, tags: Sequence[str],
             self_declared: Iterable[str] = (), mode: str = "navigate") -> Result:
    query = room_query(subject, tags)
    if mode == "navigate":
        res = navigate(root, query, judge, config)
    elif mode in ("flat", "separate"):
        units = [n for n in walk(root) if config.returnable(n)]
        res = score_units(units, query, judge, config, separate=(mode == "separate"))
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    declared = list(dict.fromkeys(self_declared))
    res.ranked = ([Ranked(c, None, source="self_declared") for c in declared]
                  + [r for r in res.ranked if r.id not in declared])
    return res


def load_tree() -> tuple[dict, Node]:
    taxonomy = utils.load_taxonomy()
    return taxonomy, build_tree(taxonomy, utils.load_keyword_rows())
