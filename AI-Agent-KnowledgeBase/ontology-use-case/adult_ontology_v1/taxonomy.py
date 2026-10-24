"""Three-level taxonomy metadata, JSON Schema, and cross-reference validation."""

TAXONOMY_VERSION = "1.1.0"
CATEGORIES = {
    "body_appearance": {"code": "BODY", "label": "Body & Appearance"},
    "sexual_activities": {"code": "ACT", "label": "Sexual Activity"},
    "fetish_kink": {"code": "KINK", "label": "Fetish & Kink"},
    "performance_format": {"code": "FORMAT", "label": "Performance Format"},
    "products_props": {"code": "PRODUCT", "label": "Products & Props"},
    "social_interaction": {"code": "SOCIAL", "label": "Social Interaction"},
    "identity_audience": {"code": "IDENTITY", "label": "Identity & Audience"},
    "commercial_context": {"code": "COMMERCE", "label": "Monetization"},
}
CATEGORY_CODES = tuple(category["code"] for category in CATEGORIES.values())
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
            "axis": {"type": "string"},
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
        "$id": "https://example.org/ontology/taxonomy-v1.1.schema.json",
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
    """Reject invalid levels, categories, parents and inconsistent IS_A edges."""
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
        nodes[node["id"]] = node
    roots = [node for node in nodes.values() if node["level"] == 0]
    categories = [node["category_id"] for node in nodes.values() if node["level"] == 1]
    if (
        len(roots) > 1
        or len(categories) != len(CATEGORY_CODES)
        or set(categories) != set(CATEGORY_CODES)
    ):
        raise ValueError("Expected eight unique L1 categories and at most one L0 root")
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
            continue
        if category not in CATEGORY_CODES:
            raise ValueError(f"Unknown category: {node['id']}")
        if level == 1 and not roots:
            if parent_id is not None:
                raise ValueError(
                    f"Rootless L1 category must have no parent: {node['id']}"
                )
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
