"""Versioned room assignments kept separate from taxonomy and keyword vocabulary."""

from copy import deepcopy
from typing import Iterable, Sequence

from matching import Matcher
from taxonomy import CATEGORY_CODES, validate_taxonomy


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
