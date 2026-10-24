"""Generate and match a small, curated ontology using only the standard library."""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TREE = {
    "body_appearance": {
        "nudity": ["full_nudity", "partial_nudity", "implied_nudity"],
        "clothing": ["lingerie", "costume", "swimwear"],
        "appearance": ["tattoos", "hair_style"],
    },
    "sexual_activities": {
        "solo": ["solo_performance", "self_stimulation"],
        "partnered": ["partnered_intimacy", "partnered_activity"],
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
        "conversation": ["flirting", "companionship"],
        "participation": ["audience_requests", "games"],
    },
    "identity_audience": {
        "self_description": ["gender_identity", "orientation"],
        "audience_description": ["audience_interest"],
    },
    "commercial_context": {
        "monetization": ["tips", "paid_session", "subscription"],
        "promotion": ["special_offer", "scheduled_event"],
    },
}
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
    ("tip goal", "goal_show", True),
    ("adult toy", "adult_toy", True),
    ("flirting", "flirting", True),
    ("roleplay", "roleplay", True),
    ("solo show", "solo_performance", True),
    ("subscription", "subscription", True),
    ("costume", "costume", True),
]


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def generate():
    nodes = []
    edges = []
    nodes.append(
        dict(
            id="adult_content",
            label="Adult Content",
            parent_id=None,
            axis="topic",
            level=0,
        )
    )
    for a, branches in TREE.items():
        nodes.append(
            dict(
                id=a,
                label=a.replace("_", " ").title(),
                parent_id="adult_content",
                axis="topic",
                level=1,
            )
        )
        edges.append(dict(source=a, predicate="IS_A", target="adult_content"))
        for b, leaves in branches.items():
            nodes.append(
                dict(
                    id=b,
                    label=b.replace("_", " ").title(),
                    parent_id=a,
                    axis="topic",
                    level=2,
                )
            )
            edges.append(dict(source=b, predicate="IS_A", target=a))
            for c in leaves:
                nodes.append(
                    dict(
                        id=c,
                        label=c.replace("_", " ").title(),
                        parent_id=b,
                        axis="topic",
                        level=3,
                    )
                )
                edges.append(dict(source=c, predicate="IS_A", target=b))
    (ROOT / "taxonomy_v1.json").write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "nodes": nodes,
                "edges": edges,
                "notes": "Identity labels are descriptive, not inherently NSFW. Safety policy is separate.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (ROOT / "keywords_v1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "keyword",
                "language",
                "concept_id",
                "source",
                "context_required",
                "review_status",
            ],
        )
        w.writeheader()
        for term, concept, amb in SEEDS:
            w.writerow(
                dict(
                    keyword=term,
                    language="en",
                    concept_id=concept,
                    source="curated_seed",
                    context_required=str(amb).lower(),
                    review_status="seed_review_required",
                )
            )
    generate_schema()


def generate_schema():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.org/ontology/taxonomy-v1.schema.json",
        "type": "object",
        "required": ["version", "nodes", "edges"],
        "properties": {
            "version": {"type": "string"},
            "notes": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "label", "parent_id", "axis", "level"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "parent_id": {"type": ["string", "null"]},
                        "axis": {"type": "string"},
                        "level": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source", "predicate", "target"],
                    "properties": {
                        "source": {"type": "string"},
                        "predicate": {
                            "enum": [
                                "IS_A",
                                "ALIAS_OF",
                                "RELATED_TO",
                                "CO_OCCURS_WITH",
                                "USES_PRODUCT",
                                "HAS_FORMAT",
                            ]
                        },
                        "target": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    (ROOT / "taxonomy_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


def load():
    """Load editable ontology files without keeping file handles open."""
    taxonomy = json.loads((ROOT / "taxonomy_v1.json").read_text(encoding="utf-8"))
    parents = {}
    for node in taxonomy["nodes"]:
        if node["id"] in parents:
            raise ValueError(f"Duplicate concept ID: {node['id']}")
        parents[node["id"]] = node["parent_id"]
    with (ROOT / "keywords_v1.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return parents, rows


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


def import_words(path, language="en"):
    """Stream dictionary lines into unmapped candidates, deduplicated per language."""
    source = Path(path)
    dest = ROOT / "candidate_terms.csv"
    if dest.exists() and source.samefile(dest):
        raise ValueError("Import source and candidate destination are the same file")
    existing = set()
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
                if word and (word, language) not in existing:
                    w.writerow(
                        {
                            "keyword": word,
                            "language": language,
                            "source": str(path),
                            "review_status": "unmapped_unreviewed",
                        }
                    )
                    existing.add((word, language))
    print("Candidates:", len(existing), "(not automatically classified)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true")
    p.add_argument("--subject", default="Private show with cosplay and latex")
    p.add_argument("--tags", nargs="*", default=["bdsm", "lingerie"])
    p.add_argument("--import-words")
    p.add_argument("--language", default="en")
    args = p.parse_args()
    if args.init or not (ROOT / "taxonomy_v1.json").exists():
        generate()
    if args.import_words:
        import_words(args.import_words, args.language)
    parents, rows = load()
    print(json.dumps(match(args.subject, args.tags, parents, rows), indent=2))
