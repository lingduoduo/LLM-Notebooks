"""Room taxonomy library: the definition, validation, keyword vocabulary,
matching, room classification, mapping review and visuals behind main.py.

Standard library only. The mapping-review helpers import openpyxl and the chart
imports matplotlib/numpy when they are called, so neither is needed otherwise.
"""

import csv
import json
import math
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent


# ---- Taxonomy definition and validation --------------------------------------


TAXONOMY_VERSION = "1.3.0"


# Independent dimensions: only "content" categories sit under the L0 root, so a
# format, interaction, identity, commerce or generic label never implies adult
# content. GENERIC holds non-specific terms (generic explicitness, profanity,
# brands). Safety/policy labels belong to a separate system and have no axis here.
CONTENT_AXIS = "content"


AXES = (CONTENT_AXIS, "format", "interaction", "identity", "commerce", "generic")


CATEGORIES = {
    "body_appearance": {
        "code": "BODY",
        "label": "Body & Appearance",
        "axis": "content",
    },
    "sexual_activities": {"code": "ACT", "label": "Sexual Activity", "axis": "content"},
    "fetish_kink": {"code": "KINK", "label": "Fetish & Kink", "axis": "content"},
    "performance_format": {
        "code": "FORMAT",
        "label": "Performance Format",
        "axis": "format",
    },
    "products_props": {
        "code": "PRODUCT",
        "label": "Products & Props",
        "axis": "content",
    },
    "social_interaction": {
        "code": "SOCIAL",
        "label": "Social Interaction",
        "axis": "interaction",
    },
    "identity_audience": {
        "code": "IDENTITY",
        "label": "Identity & Audience",
        "axis": "identity",
    },
    "commercial_context": {
        "code": "COMMERCE",
        "label": "Monetization",
        "axis": "commerce",
    },
    "generic_reference": {"code": "GENERIC", "label": "Generic", "axis": "generic"},
}


CATEGORY_CODES = tuple(category["code"] for category in CATEGORIES.values())


CATEGORY_AXES = {category["code"]: category["axis"] for category in CATEGORIES.values()}


LABELS = {
    "bdsm": "BDSM",
    "audience_engagement": "Interaction",
    "participation": "Engagement",
    "audience_description": "Audience",
    "self_description": "Self-description",
    "monetization": "Payments",
    "promotion": "Promotions",
    "material_interest": "Material Interests",
}


PREDICATES = (
    "IS_A",
    "ALIAS_OF",
    "RELATED_TO",
    "CO_OCCURS_WITH",
    "USES_PRODUCT",
    "HAS_FORMAT",
)


def taxonomy_schema() -> dict:
    """JSON Schema describes structure; validate_taxonomy checks graph semantics."""
    node = {
        "type": "object",
        "required": ["id", "label", "parent_id", "axis", "level", "category_id"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "label": {"type": "string", "minLength": 1},
            "parent_id": {"type": ["string", "null"]},
            "axis": {"enum": list(AXES)},
            "level": {"type": "integer", "minimum": 0, "maximum": 3},
            "category_id": {"enum": [None, *CATEGORY_CODES]},
        },
        "allOf": [
            {
                "if": {"properties": {"level": {"const": 0}}},
                "then": {
                    "properties": {
                        "parent_id": {"type": "null"},
                        "category_id": {"type": "null"},
                    }
                },
                "else": {"properties": {"category_id": {"enum": list(CATEGORY_CODES)}}},
            }
        ],
        "additionalProperties": False,
    }
    edge = {
        "type": "object",
        "required": ["source", "predicate", "target"],
        "properties": {
            "source": {"type": "string"},
            "predicate": {"enum": list(PREDICATES)},
            "target": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/ontology/taxonomy-v1.3.schema.json",
        "type": "object",
        "required": ["version", "nodes", "edges"],
        "properties": {
            "version": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
            "nodes": {"type": "array", "items": node},
            "edges": {"type": "array", "items": edge},
        },
        "additionalProperties": False,
    }


def validate_taxonomy(taxonomy: dict) -> None:
    """Reject invalid levels, categories, axes, parents and inconsistent IS_A edges."""
    if (
        not isinstance(taxonomy, dict)
        or not isinstance(taxonomy.get("version"), str)
        or not taxonomy["version"].strip()
    ):
        raise ValueError("Taxonomy requires a non-empty version")
    if not isinstance(taxonomy.get("nodes"), list) or not isinstance(
        taxonomy.get("edges"), list
    ):
        raise ValueError("Taxonomy nodes and edges must be lists")
    nodes = {}
    for node in taxonomy["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not node["id"].strip()
        ):
            raise ValueError("Concept IDs must be non-empty strings")
        if node["id"] in nodes:
            raise ValueError(f"Duplicate concept ID: {node['id']}")
        if not isinstance(node.get("label"), str) or not node["label"].strip():
            raise ValueError(f"Missing concept label: {node['id']}")
        if type(node.get("level")) is not int or node["level"] not in range(4):
            raise ValueError(f"Invalid hierarchy level: {node['id']}")
        if (
            "parent_id" not in node
            or "category_id" not in node
            or not isinstance(node.get("axis"), str)
        ):
            raise ValueError(f"Missing concept metadata: {node['id']}")
        if node["parent_id"] is not None and not isinstance(node["parent_id"], str):
            raise ValueError(f"Invalid parent ID: {node['id']}")
        if node["category_id"] is not None and not isinstance(node["category_id"], str):
            raise ValueError(f"Invalid category code: {node['id']}")
        if node["axis"] not in AXES:
            raise ValueError(f"Unknown axis: {node['id']}")
        nodes[node["id"]] = node
    roots = [node for node in nodes.values() if node["level"] == 0]
    categories = [node["category_id"] for node in nodes.values() if node["level"] == 1]
    if (
        len(roots) > 1
        or len(categories) != len(CATEGORY_CODES)
        or set(categories) != set(CATEGORY_CODES)
    ):
        raise ValueError(
            "Expected one L1 node per category code and at most one L0 root"
        )
    expected_edges = set()
    for node in nodes.values():
        level, parent_id, category = (
            node["level"],
            node["parent_id"],
            node["category_id"],
        )
        if level == 0:
            if parent_id is not None or category is not None:
                raise ValueError("L0 root must have no parent or category")
            if node["axis"] != CONTENT_AXIS:
                raise ValueError("L0 root must be on the content axis")
            continue
        if category not in CATEGORY_CODES:
            raise ValueError(f"Unknown category: {node['id']}")
        if node["axis"] != CATEGORY_AXES[category]:
            raise ValueError(f"Axis differs from category: {node['id']}")
        # Only content categories attach to the root; others stand alone.
        if level == 1 and (not roots or node["axis"] != CONTENT_AXIS):
            if parent_id is not None:
                raise ValueError(f"L1 category must have no parent: {node['id']}")
            continue
        parent = nodes.get(parent_id)
        if parent is None or parent["level"] != level - 1:
            raise ValueError(f"Invalid parent level or reference: {node['id']}")
        if level > 1 and category != parent["category_id"]:
            raise ValueError(f"Category differs from parent: {node['id']}")
        expected_edges.add((node["id"], parent_id))
    actual_edges = set()
    for edge in taxonomy["edges"]:
        if not isinstance(edge, dict) or edge.get("predicate") not in PREDICATES:
            raise ValueError("Invalid taxonomy edge")
        if (
            not isinstance(edge.get("source"), str)
            or not isinstance(edge.get("target"), str)
            or edge["source"] not in nodes
            or edge["target"] not in nodes
        ):
            raise ValueError("Unknown edge endpoint")
        if edge["predicate"] == "IS_A":
            pair = (edge["source"], edge["target"])
            if pair in actual_edges:
                raise ValueError("Duplicate IS_A edge")
            actual_edges.add(pair)
    if actual_edges != expected_edges:
        raise ValueError("IS_A edges must exactly match parent_id relationships")


TREE = {
    "body_appearance": {
        "nudity": ["full_nudity", "partial_nudity", "implied_nudity"],
        "clothing": ["lingerie", "costume", "swimwear"],
        "appearance": ["tattoos", "hair_style"],
    },
    "sexual_activities": {
        "solo": ["solo_performance", "self_stimulation"],
        "partnered": ["partnered_activity"],
        "intimacy": ["partnered_intimacy"],
    },
    "fetish_kink": {
        "bdsm": ["bondage", "dominance", "submission"],
        "roleplay": ["costume_roleplay", "fantasy_roleplay"],
        "material_interest": ["latex", "leather", "footwear"],
    },
    "performance_format": {
        "broadcast_mode": ["public_live", "private_show", "group_show"],
        "audience_engagement": ["interactive_show", "request_show", "goal_show"],
    },
    "products_props": {
        "devices": ["adult_toy", "interactive_device"],
        "accessories": ["restraints", "costume_props"],
    },
    "social_interaction": {
        "conversation": ["flirting", "companionship", "chat"],
        "participation": ["audience_requests", "games"],
    },
    "identity_audience": {
        "self_description": ["gender_identity", "orientation"],
        "audience_description": ["audience_interest"],
    },
    "commercial_context": {
        "monetization": ["tips", "paid_session", "subscription", "tip_goal"],
        "promotion": ["special_offer", "scheduled_event"],
    },
    "generic_reference": {
        "explicit_reference": ["explicit_general"],
        "crude_language": ["profanity", "crude_slang"],
        "media_reference": ["brand_reference", "shock_reference"],
        "unrelated": ["other_reference"],
    },
}


# The single keywords file. Each row is exactly one kind:
# - a keyword: concept_id set, policy_route empty (loaded by load());
# - a policy candidate: policy_route set, concept_id empty (never matched);
# - an excluded term: both empty, review_status "excluded" (a reviewed drop,
#   kept so imports skip it; never matched).
# SEEDS rows are regenerated by --init; all other rows are preserved verbatim.
VOCABULARY_FILE = "keywords.csv"


FIELDS = [
    "keyword",
    "language",
    "concept_id",
    "source",
    "context_required",
    "review_status",
    "policy_route",
]


POLICY_ROUTES = (
    "minor_safety",
    "non_consent",
    "incest",
    "bestiality",
    "violence_extremism",
    "solicitation",
    "hate_slur",
)


SEED_SOURCE = "curated_seed"


EXCLUDED_STATUS = "excluded"


SEEDS = [
    ("nude", "full_nudity", False),
    ("nudity", "full_nudity", False),
    ("topless", "partial_nudity", False),
    ("lingerie", "lingerie", False),
    ("cosplay", "costume_roleplay", True),
    ("bdsm", "bdsm", False),
    ("bondage", "bondage", False),
    ("domination", "dominance", True),
    ("submission", "submission", True),
    ("latex", "latex", True),
    ("leather", "leather", True),
    ("private show", "private_show", True),
    ("group show", "group_show", True),
    ("interactive show", "interactive_show", True),
    ("tips", "tips", True),
    ("tip goal", "tip_goal", True),
    ("adult toy", "adult_toy", True),
    ("flirting", "flirting", True),
    ("roleplay", "roleplay", True),
    ("solo show", "solo_performance", True),
    ("subscription", "subscription", True),
    ("costume", "costume", True),
    ("chat", "chat", True),
    ("partnered intimacy", "partnered_intimacy", True),
    ("public live", "public_live", True),
]


# ---- Keyword matching --------------------------------------------------------


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


class Matcher:
    """Prepare once, then reuse for multiple subjects with stable row ordering."""

    def __init__(self, parents, rows):
        # Resolve iteratively so deep trees do not hit Python's recursion limit.
        self._ancestors = {}
        for concept in parents:
            path = []
            visiting = set()
            current = concept
            while current is not None and current not in self._ancestors:
                if current not in parents:
                    raise ValueError(f"Unknown parent concept: {current}")
                if current in visiting:
                    raise ValueError(f"Cycle in taxonomy at: {current}")
                visiting.add(current)
                path.append(current)
                current = parents[current]
            ancestors = (
                self._ancestors[current] | {current}
                if current is not None
                else frozenset()
            )
            for node in reversed(path):
                self._ancestors[node] = frozenset(ancestors)
                ancestors = ancestors | {node}

        self._keywords = []
        for row in rows:
            keyword = norm(row["keyword"])
            concept = row["concept_id"]
            if not keyword:
                raise ValueError("Keyword must not be empty")
            if concept not in parents:
                raise ValueError(f"Unknown keyword concept: {concept}")
            if row["context_required"] not in ("true", "false"):
                raise ValueError(f"Invalid context_required for: {keyword}")
            pattern = re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)")
            self._keywords.append(
                (
                    pattern,
                    keyword,
                    concept,
                    row["context_required"] == "true",
                    row["source"],
                )
            )

    def match(self, subject, tags):
        tags = tags[:5]
        texts = [("subject", subject)] + [
            (f"tag_{i + 1}", tag) for i, tag in enumerate(tags)
        ]
        hits = []
        for field, raw in texts:
            text = norm(raw)
            for pattern, keyword, concept, needs_context, source in self._keywords:
                # Substring presence is necessary for a match and far cheaper.
                if keyword in text and pattern.search(text):
                    hits.append(
                        {
                            "field": field,
                            "term": keyword,
                            "concept_id": concept,
                            "context_required": needs_context,
                            "source": source,
                        }
                    )
        direct = {hit["concept_id"] for hit in hits}
        inferred = set()
        for concept in direct:
            inferred.update(self._ancestors[concept])
        return {
            "subject": subject,
            "tags": tags,
            "matches": hits,
            "direct_concepts": sorted(direct),
            "inferred_ancestors": sorted(inferred - direct),
            "requires_context_review": any(hit["context_required"] for hit in hits),
        }


def match(subject, tags, parents, rows):
    """Compatibility wrapper; use Matcher directly when classifying a batch."""
    return Matcher(parents, rows).match(subject, tags)


# ---- Building and loading the taxonomy and keywords --------------------------


def generate():
    nodes = []
    edges = []
    nodes.append(
        dict(
            id="adult_content",
            label="Adult Content",
            parent_id=None,
            axis=CONTENT_AXIS,
            level=0,
            category_id=None,
        )
    )
    for a, branches in TREE.items():
        axis = CATEGORIES[a]["axis"]
        is_content = axis == CONTENT_AXIS
        nodes.append(
            dict(
                id=a,
                label=CATEGORIES[a]["label"],
                parent_id="adult_content" if is_content else None,
                axis=axis,
                level=1,
                category_id=CATEGORIES[a]["code"],
            )
        )
        if is_content:
            edges.append(dict(source=a, predicate="IS_A", target="adult_content"))
        for b, leaves in branches.items():
            nodes.append(
                dict(
                    id=b,
                    label=LABELS.get(b, b.replace("_", " ").title()),
                    parent_id=a,
                    axis=axis,
                    level=2,
                    category_id=CATEGORIES[a]["code"],
                )
            )
            edges.append(dict(source=b, predicate="IS_A", target=a))
            for c in leaves:
                nodes.append(
                    dict(
                        id=c,
                        label=c.replace("_", " ").title(),
                        parent_id=b,
                        axis=axis,
                        level=3,
                        category_id=CATEGORIES[a]["code"],
                    )
                )
                edges.append(dict(source=c, predicate="IS_A", target=b))
    taxonomy = {
        "version": TAXONOMY_VERSION,
        "nodes": nodes,
        "edges": edges,
        "notes": "Identity labels are descriptive, not inherently NSFW. Safety policy is separate.",
    }
    validate_taxonomy(taxonomy)
    (ROOT / "taxonomy_v1.json").write_text(
        json.dumps(taxonomy, indent=2), encoding="utf-8"
    )
    path = ROOT / VOCABULARY_FILE
    imported = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            imported = [r for r in csv.DictReader(stream) if r["source"] != SEED_SOURCE]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for term, concept, amb in SEEDS:
            w.writerow(
                dict(
                    keyword=term,
                    language="en",
                    concept_id=concept,
                    source=SEED_SOURCE,
                    context_required=str(amb).lower(),
                    review_status="seed_review_required",
                    policy_route="",
                )
            )
        w.writerows(imported)
    generate_schema()


def generate_schema():
    (ROOT / "taxonomy_schema.json").write_text(
        json.dumps(taxonomy_schema(), indent=2), encoding="utf-8"
    )


def load_taxonomy() -> dict:
    """Load and validate the versioned taxonomy at the compatibility file path."""
    taxonomy = json.loads((ROOT / "taxonomy_v1.json").read_text(encoding="utf-8"))
    validate_taxonomy(taxonomy)
    return taxonomy


def load():
    """Load parent mappings and vocabulary for the existing matcher API."""
    taxonomy = load_taxonomy()
    parents = {node["id"]: node["parent_id"] for node in taxonomy["nodes"]}
    keywords, _ = split_keyword_rows(load_keyword_rows())
    return parents, keywords


def load_policy_candidates():
    """Policy-review rows from the keywords file; these are never matched."""
    return split_keyword_rows(load_keyword_rows())[1]


def load_keyword_rows():
    with (ROOT / VOCABULARY_FILE).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def split_keyword_rows(rows):
    """Return (keyword rows, policy rows); every row must be exactly one kind.

    Excluded rows are validated and then left out of both lists.
    """
    keywords, policy, excluded = [], [], []
    for row in rows:
        concept, route = row.get("concept_id") or "", row.get("policy_route") or ""
        if row.get("review_status") == EXCLUDED_STATUS:
            if concept or route:
                raise ValueError(
                    f"Excluded row must have no concept_id or policy_route: {row['keyword']!r}"
                )
            excluded.append(row)
            continue
        if bool(concept) == bool(route):
            raise ValueError(
                f"Row needs exactly one of concept_id or policy_route: {row['keyword']!r}"
            )
        if route and route not in POLICY_ROUTES:
            raise ValueError(f"Unknown policy_route {route!r}: {row['keyword']!r}")
        (policy if route else keywords).append(row)
    check_vocabulary(keywords)
    term = lambda r: (norm(r["keyword"]), r["language"])  # noqa: E731
    matched = {term(r) for r in keywords}
    for row in policy:
        if term(row) in matched:
            raise ValueError(
                f"Term is both a keyword and a policy candidate: {row['keyword']!r}"
            )
    routed = {term(r) for r in policy}
    for row in excluded:
        if term(row) in matched or term(row) in routed:
            raise ValueError(f"Term is excluded but also kept: {row['keyword']!r}")
    return keywords, policy


def check_vocabulary(rows):
    """Reject a keyword mapped to the same concept twice."""
    seen = set()
    for row in rows:
        key = (norm(row["keyword"]), row["language"], row["concept_id"])
        if key in seen:
            raise ValueError(f"Duplicate vocabulary row: {key}")
        seen.add(key)


def import_words(path, language="en"):
    """Stream dictionary lines into unmapped candidates, deduplicated per language.

    Terms already in keywords.csv (as a keyword or a policy candidate) for the
    same language are skipped, so reviewed terms are not offered again.
    """
    source = Path(path)
    dest = ROOT / "candidate_terms.csv"
    if dest.exists() and source.samefile(dest):
        raise ValueError("Import source and candidate destination are the same file")
    known = set()
    if (ROOT / VOCABULARY_FILE).exists():
        known = {(norm(r["keyword"]), r["language"]) for r in load_keyword_rows()}
    existing = set()
    skipped = 0
    if dest.exists():
        with dest.open(encoding="utf-8", newline="") as f:
            existing = {(r["keyword"], r["language"]) for r in csv.DictReader(f)}
    with source.open(encoding="utf-8") as stream:
        with dest.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["keyword", "language", "source", "review_status"]
            )
            if dest.stat().st_size == 0:
                w.writeheader()
            # Retain splitlines() support for Unicode separators within a line.
            entries = (entry for line in stream for entry in line.splitlines())
            for entry in entries:
                word = norm(entry)
                if word and (word, language) in known:
                    skipped += 1
                elif word and (word, language) not in existing:
                    w.writerow(
                        {
                            "keyword": word,
                            "language": language,
                            "source": str(path),
                            "review_status": "unmapped_unreviewed",
                        }
                    )
                    existing.add((word, language))
    print(
        "Candidates:",
        len(existing),
        f"(not automatically classified); skipped {skipped} already in {VOCABULARY_FILE}",
    )


# ---- Room classification -----------------------------------------------------


class RoomClassifier:
    """Prepare once; classify many rooms without inferring identity from keywords."""

    def __init__(self, taxonomy: dict, rows: Iterable[dict]) -> None:
        validate_taxonomy(taxonomy)
        self.version = taxonomy["version"]
        self.nodes = {node["id"]: deepcopy(node) for node in taxonomy["nodes"]}
        parents = {node_id: node["parent_id"] for node_id, node in self.nodes.items()}
        rows = list(rows)
        self.matcher = Matcher(parents, rows)
        if any(self.nodes[row["concept_id"]]["level"] == 0 for row in rows):
            raise ValueError(
                "Room vocabulary must reference L1–L3 concepts, not the L0 root"
            )

    def classify_room(
        self,
        room_id: str,
        subject: str,
        tags: Sequence[str],
        *,
        self_declared_concepts: Iterable[str] = (),
    ) -> dict:
        """Group matched evidence by concept and derive ancestors, L1 and axes.

        Evidence retains the complete original field and the normalized matched
        term, without claiming a precise raw-text span after Unicode normalization.
        Identity concepts are accepted only through explicit self-declaration.
        """
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        if (
            not isinstance(subject, str)
            or not isinstance(tags, (list, tuple))
            or any(not isinstance(tag, str) for tag in tags)
        ):
            raise ValueError("subject and tags must contain strings")
        if isinstance(self_declared_concepts, str):
            raise ValueError(
                "self_declared_concepts must be a collection of concept IDs"
            )
        raw_fields = {
            "subject": subject,
            **{f"tag_{i + 1}": tag for i, tag in enumerate(tags[:5])},
        }
        assignments = {}

        def assignment(concept_id):
            node = self.nodes[concept_id]
            return assignments.setdefault(
                concept_id,
                {
                    "concept_id": concept_id,
                    "category_id": node["category_id"],
                    "axis": node["axis"],
                    "level": node["level"],
                    "evidence": [],
                    "requires_context_review": False,
                },
            )

        for hit in self.matcher.match(subject, tags)["matches"]:
            if self.nodes[hit["concept_id"]]["category_id"] == "IDENTITY":
                continue
            item = assignment(hit["concept_id"])
            item["evidence"].append(
                {
                    "field": hit["field"],
                    "text": raw_fields[hit["field"]],
                    "matched_term": hit["term"],
                    "source": hit["source"],
                }
            )
            item["requires_context_review"] |= hit["context_required"]
        for concept_id in self_declared_concepts:
            if (
                not isinstance(concept_id, str)
                or concept_id not in self.nodes
                or self.nodes[concept_id]["category_id"] != "IDENTITY"
            ):
                raise ValueError(
                    f"Self-declaration requires a known IDENTITY concept: {concept_id}"
                )
            item = assignment(concept_id)
            if not item["evidence"]:
                item["evidence"].append(
                    {
                        "field": "self_declared_concepts",
                        "text": concept_id,
                        "matched_term": None,
                        "source": "self_declared",
                    }
                )
        ancestors = set()
        for concept_id in assignments:
            parent = self.nodes[concept_id]["parent_id"]
            while parent is not None:
                ancestors.add(parent)
                parent = self.nodes[parent]["parent_id"]
        return {
            "room_id": room_id,
            "taxonomy_version": self.version,
            "subject": subject,
            "tags": list(tags[:5]),
            "assignments": [
                assignments[concept_id] for concept_id in sorted(assignments)
            ],
            "inferred_ancestors": sorted(ancestors - assignments.keys()),
            "category_ids": sorted(
                {item["category_id"] for item in assignments.values()}
            ),
            "axes": sorted({item["axis"] for item in assignments.values()}),
            "requires_context_review": any(
                item["requires_context_review"] for item in assignments.values()
            ),
        }


def count_rooms_by_category(rooms: Iterable[dict]) -> dict:
    """Count distinct room IDs with evidence, including assignments needing review.

    Repeated records for a room are unioned; callers should select the desired
    snapshot before aggregation. Reject mixed taxonomy versions to avoid silently
    combining memberships defined by different releases.
    """
    members = {code: set() for code in CATEGORY_CODES}
    version = None
    for room in rooms:
        room_id = room["room_id"]
        if not isinstance(room_id, str) or not room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        current_version = room.get("taxonomy_version")
        if not isinstance(current_version, str) or not current_version.strip():
            raise ValueError("Every room requires a non-empty taxonomy_version")
        if version is not None and current_version != version:
            raise ValueError("Cannot aggregate rooms from different taxonomy versions")
        version = current_version
        for code in room["category_ids"]:
            if code not in members:
                raise ValueError(f"Unknown room category: {code}")
            members[code].add(room_id)
    return {code: len(room_ids) for code, room_ids in members.items()}


# ---- Mapping review (reads the openpyxl workbook lazily) ---------------------


DECISIONS = ("accept", "change", "move to policy", "drop")


COLUMNS = {
    "keyword": "Keyword",
    "concept": "Mapped concept_id",
    "decision": "Decision",
    "corrected": "Corrected concept_id",
    "route": "Policy route",
    "context": "Set context_required",
}


# Optional: the original flag, used when a keyword has to be re-created.
OPTIONAL_COLUMNS = {"original_context": "context_required"}


def read_decisions(workbook_path):
    from openpyxl import load_workbook

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        rows = workbook["Review"].iter_rows(values_only=True)
        header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
        missing = [name for name in COLUMNS.values() if name not in header]
        if missing:
            raise ValueError(f"Review sheet is missing columns: {missing}")
        index = {key: header.index(name) for key, name in COLUMNS.items()}
        index.update(
            (key, header.index(name))
            for key, name in OPTIONAL_COLUMNS.items()
            if name in header
        )
        decided = []
        for row in rows:
            record = {
                key: "" if i >= len(row) or row[i] is None else str(row[i]).strip()
                for key, i in index.items()
            }
            if record["decision"]:
                decided.append(record)
        return decided
    finally:
        workbook.close()


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def apply_review(workbook_path, status, dry_run=False, source="ldnoobw_en"):
    """Write the review workbook's decisions to keywords.csv.

    The workbook is the source of truth for every keyword it decides, so
    re-applying it changes nothing and an override is always reversible:

    - accept / change: a keyword row under the mapped / corrected concept;
    - move to policy: a policy-candidate row with the chosen route;
    - drop: an excluded row, so future imports skip the term.

    "Set context_required" overrides the flag; a re-created row otherwise keeps
    the workbook's "context_required" column, or "true". Every decision is
    validated before anything is written.
    """
    taxonomy = load_taxonomy()
    nodes = {node["id"]: node for node in taxonomy["nodes"]}
    path = ROOT / VOCABULARY_FILE
    fields, rows = _read_csv(path)
    current = {norm(r["keyword"]): r for r in rows if r["source"] == source}

    outcomes = {}  # normalized keyword -> its new row (keyword, policy or excluded)
    summary = {decision: 0 for decision in DECISIONS}
    for d in read_decisions(workbook_path):
        keyword = norm(d["keyword"])
        where = f"{d['keyword']!r} ({d['concept']})"
        if not keyword:
            raise ValueError("Decided rows need a keyword")
        if keyword in outcomes:
            raise ValueError(f"Keyword decided more than once: {where}")
        if d["decision"] not in DECISIONS:
            raise ValueError(f"Unknown decision {d['decision']!r} for {where}")
        if d["context"] not in ("", "true", "false"):
            raise ValueError(f"Set context_required must be true or false for {where}")
        concept = d["corrected"] if d["decision"] == "change" else d["concept"]
        if d["decision"] in ("accept", "change"):
            target = nodes.get(concept)
            if target is None or target["level"] == 0:
                raise ValueError(f"Concept must be an L1-L3 concept for {where}")
            if target["category_id"] == "IDENTITY":
                raise ValueError(f"Keywords may not map to IDENTITY concepts: {where}")
            previous = current.get(keyword, {})
            context = (
                d["context"]
                or previous.get("context_required")
                or d.get("original_context")
                or "true"
            )
            outcomes[keyword] = {
                "keyword": previous.get("keyword", keyword),
                "language": previous.get("language", "en"),
                "concept_id": concept,
                "source": source,
                "context_required": context,
                "review_status": status,
                "policy_route": "",
            }
        elif d["decision"] == "move to policy":
            if d["route"] not in POLICY_ROUTES:
                raise ValueError(
                    f"Policy route must be one of {POLICY_ROUTES} for {where}"
                )
            outcomes[keyword] = {
                "keyword": current.get(keyword, {}).get("keyword", keyword),
                "language": current.get(keyword, {}).get("language", "en"),
                "concept_id": "",
                "source": source,
                "context_required": "",
                "review_status": "policy_review_required",
                "policy_route": d["route"],
            }
        else:
            outcomes[keyword] = {
                "keyword": current.get(keyword, {}).get("keyword", keyword),
                "language": current.get(keyword, {}).get("language", "en"),
                "concept_id": "",
                "source": source,
                "context_required": "",
                "review_status": EXCLUDED_STATUS,
                "policy_route": "",
            }
        summary[d["decision"]] += 1

    # Replace decided rows in place, or append ones not yet present.
    settled, seen = [], set()
    for row in rows:
        key = norm(row["keyword"])
        if row["source"] != source or key not in outcomes:
            settled.append(row)
            continue
        seen.add(key)
        settled.append(outcomes[key])
    settled += [new for key, new in outcomes.items() if key not in seen]
    keywords, _ = split_keyword_rows(settled)
    Matcher({i: n["parent_id"] for i, n in nodes.items()}, keywords)
    print(
        ("Would apply" if dry_run else "Applied"),
        ", ".join(f"{count} {decision}" for decision, count in summary.items()),
    )
    if dry_run:
        return summary
    # Same dialect as the generator (CRLF).
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(settled)
    return summary


# ---- Visuals: offline explorer page and semicircle chart (matplotlib, lazily) ----


# Axis colors match the taxonomy map page (validated categorical slots 1-6).
AXIS_COLORS = {
    "content": "#2a78d6",
    "format": "#eb6834",
    "interaction": "#1baf7a",
    "identity": "#eda100",
    "commerce": "#e87ba4",
    "generic": "#008300",
}


INK = "#14171c"


EMPTY_NOTES = {"identity": "self-declared only"}


def taxonomy_rows(taxonomy, vocabulary):
    """Return (category, group, concept, count, axis) rows in taxonomy order."""
    counts = Counter(row["concept_id"] for row in vocabulary)
    nodes = taxonomy["nodes"]
    children = {}
    for node in nodes:
        children.setdefault(node["parent_id"], []).append(node)
    rows = []
    for category in (n for n in nodes if n["level"] == 1):
        name, axis = category["category_id"], category["axis"]
        if counts[category["id"]]:
            rows.append(
                (
                    name,
                    "Category-level",
                    f"Direct to {name}",
                    counts[category["id"]],
                    axis,
                )
            )
        for group in children.get(category["id"], []):
            if counts[group["id"]]:
                rows.append(
                    (
                        name,
                        group["label"],
                        f"Direct to {group['label']}",
                        counts[group["id"]],
                        axis,
                    )
                )
            for concept in children.get(group["id"], []):
                if counts[concept["id"]]:
                    rows.append(
                        (
                            name,
                            group["label"],
                            concept["label"],
                            counts[concept["id"]],
                            axis,
                        )
                    )
    return rows


def category_order(taxonomy):
    """Return [(category code, axis)] for every L1 category in taxonomy order."""
    return [(n["category_id"], n["axis"]) for n in taxonomy["nodes"] if n["level"] == 1]


def draw_taxonomy(rows, title, output_prefix="taxonomy_semicircle", categories=None):
    """Draw L1 -> L2 -> L3 rings sized by keyword rows; write PNG and SVG.

    Angles stay proportional: a category too thin to label gets an outside
    callout, and one with no keywords (IDENTITY) is a dashed marker.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Wedge

    tree = {}
    axes = {}
    for category, group, concept, count, axis in rows:
        if count <= 0:
            continue
        words = tree.setdefault(category, {}).setdefault(group, {})
        words[concept] = words.get(concept, 0) + count
        axes[category] = axis
    totals = {
        cat: sum(sum(words.values()) for words in groups.values())
        for cat, groups in tree.items()
    }
    grand_total = sum(totals.values())
    if grand_total == 0:
        raise ValueError("No positive counts to plot")
    if categories is None:
        categories = list(axes.items())

    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("white")
    ring_width = 0.95
    inner_radius = 1.55
    max_radius = inner_radius + 3 * ring_width
    ax.set_xlim(-max_radius - 0.25, max_radius + 0.25)
    # Headroom above the arc for callouts.
    ax.set_ylim(-0.2, max_radius + 1.4)
    # Data units per point, to decide whether a label fits its wedge.
    origin, unit = ax.transData.transform([(0, 0), (1, 0)])
    units_per_pt = fig.dpi / 72 / (unit[0] - origin[0])
    # Start at 180° (left) and traverse clockwise to 0° (right).
    current = 180.0

    def segment(theta_start, theta_end, ring, color, label, fontsize=9, bold=False):
        radius = inner_radius + (ring + 1) * ring_width
        ax.add_patch(
            Wedge(
                (0, 0),
                radius,
                theta_end,
                theta_start,
                width=ring_width,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
            )
        )
        sweep = theta_start - theta_end
        mid = (theta_start + theta_end) / 2
        theta = math.radians(mid)
        text_radius = inner_radius + (ring + 0.5) * ring_width
        arc = text_radius * math.radians(sweep)
        # Try the full label, then just its first line (e.g. the category code).
        for label in dict.fromkeys((label, label.split("\n")[0])):
            lines = label.split("\n")
            width = max(map(len, lines)) * fontsize * 0.6 * units_per_pt
            height = len(lines) * fontsize * 1.25 * units_per_pt
            if width <= arc * 0.92:
                # Tangential text, kept upright.
                rotation = mid - 90
                break
            if height <= arc and width <= ring_width * 0.95:
                # Narrow wedge: radial text along the wedge.
                rotation = mid
                break
        else:
            return False
        if rotation < -90:
            rotation += 180
        if rotation > 90:
            rotation -= 180
        ax.text(
            text_radius * math.cos(theta),
            text_radius * math.sin(theta),
            label,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=fontsize,
            weight="bold" if bold else "normal",
            color=INK,
            clip_on=True,
        )
        return True

    def tint(base, white):
        return tuple(base * (1 - white) + np.ones(3) * white)

    callouts = []  # (angle, text)
    for cat, axis in categories:
        if cat not in tree:
            # Zero keywords: keep the category's place with a dashed marker.
            rad = math.radians(current)
            ax.plot(
                [inner_radius * math.cos(rad), max_radius * math.cos(rad)],
                [inner_radius * math.sin(rad), max_radius * math.sin(rad)],
                color=AXIS_COLORS[axis],
                linewidth=1.6,
                linestyle=(0, (3, 2)),
            )
            note = f" ({EMPTY_NOTES[axis]})" if axis in EMPTY_NOTES else ""
            callouts.append((current, f"{cat} · 0 keywords{note}"))
            continue
        groups = tree[cat]
        base = np.array(plt.matplotlib.colors.to_rgb(AXIS_COLORS[axis]))
        cat_end = current - 180 * totals[cat] / grand_total
        placed = segment(
            current,
            cat_end,
            0,
            tint(base, 0.15),
            f"{cat}\n{totals[cat] / grand_total:.0%}",
            10,
            True,
        )
        if not placed:
            n = totals[cat]
            callouts.append(
                ((current + cat_end) / 2, f"{cat} · {n} keyword{'' if n == 1 else 's'}")
            )
        group_start = current
        for group, words in groups.items():
            group_end = group_start - 180 * sum(words.values()) / grand_total
            segment(group_start, group_end, 1, tint(base, 0.45), group, 9, True)
            word_start = group_start
            for concept, count in words.items():
                word_end = word_start - 180 * count / grand_total
                segment(word_start, word_end, 2, tint(base, 0.72), concept, 8)
                word_start = word_end
            group_start = group_end
        current = cat_end

    # Callouts: one label column per side, leftmost anchor gets the top label so
    # leader lines fan out without crossing.
    for side in (1, -1):
        items = sorted(
            (c for c in callouts if (c[0] < 90) == (side == 1)),
            key=lambda c: -c[0] * side,
        )
        if not items:
            continue
        anchors = [
            (
                max_radius * math.cos(math.radians(a)),
                max_radius * math.sin(math.radians(a)),
            )
            for a, _ in items
        ]
        column = (
            max(x for x, _ in anchors) + 0.6
            if side == 1
            else min(x for x, _ in anchors) - 0.6
        )
        for i, ((x, y), (_, text)) in enumerate(zip(anchors, items)):
            label_y = max_radius + 0.3 + (len(items) - 1 - i) * 0.32
            ax.plot([x, column], [y, label_y], color=INK, linewidth=0.8)
            ax.plot([x], [y], marker="o", markersize=3, color=INK)
            ax.text(
                column + 0.05 * side,
                label_y,
                text,
                ha="left" if side == 1 else "right",
                va="center",
                fontsize=9,
                color=INK,
            )

    ax.axis("off")
    ax.set_title(title, fontsize=20, pad=22, color=INK)
    ax.text(
        0,
        0.35,
        f"{grand_total:,}\nkeyword rows",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    shown = [a for a in AXIS_COLORS if a in {axis for _, axis in categories}]
    ax.legend(
        handles=[Patch(color=AXIS_COLORS[a], label=f"{a} axis") for a in shown],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(shown),
        frameon=False,
    )
    plt.tight_layout()
    outputs = []
    for ext in ("png", "svg"):
        path = f"{output_prefix}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def build_explorer(output, taxonomy_path=None):
    """Write a single-file, offline, searchable explorer of the validated taxonomy."""
    path = Path(taxonomy_path) if taxonomy_path else ROOT / "taxonomy_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_taxonomy(data)
    template = (ROOT / "visuals/explorer.template.html").read_text(encoding="utf-8")
    # Embedded JSON cannot terminate its script element; labels render via textContent.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__TAXONOMY_JSON__", payload), encoding="utf-8")
    return output
