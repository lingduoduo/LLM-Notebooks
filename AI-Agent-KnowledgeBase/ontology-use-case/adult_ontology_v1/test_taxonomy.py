"""Three-level taxonomy and evidence-backed, multi-label room classification."""

import copy
import csv
import json
import unittest

import build


CATEGORIES = {
    "BODY",
    "ACT",
    "KINK",
    "FORMAT",
    "PRODUCT",
    "SOCIAL",
    "IDENTITY",
    "COMMERCE",
    "GENERIC",
}


class TaxonomyTests(unittest.TestCase):
    def test_category_codes_and_three_levels(self):
        taxonomy = build.load_taxonomy()
        nodes = {node["id"]: node for node in taxonomy["nodes"]}
        self.assertEqual(
            {node["category_id"] for node in nodes.values() if node["level"] == 1},
            CATEGORIES,
        )
        self.assertEqual(nodes["adult_content"]["level"], 0)
        self.assertIsNone(nodes["adult_content"]["category_id"])
        self.assertEqual(nodes["lingerie"]["parent_id"], "clothing")
        self.assertEqual(nodes["lingerie"]["category_id"], "BODY")
        self.assertEqual(nodes["partnered_intimacy"]["parent_id"], "intimacy")
        self.assertEqual(nodes["chat"]["level"], 3)
        for node in nodes.values():
            self.assertIn(node["level"], (0, 1, 2, 3))
            if node["parent_id"] is not None:
                parent = nodes[node["parent_id"]]
                self.assertEqual(node["level"], parent["level"] + 1)
                if parent["level"]:
                    self.assertEqual(node["category_id"], parent["category_id"])

    def test_only_content_categories_sit_under_the_adult_content_root(self):
        nodes = {n["id"]: n for n in build.load_taxonomy()["nodes"]}
        self.assertEqual(nodes["adult_content"]["axis"], "content")
        for node in nodes.values():
            if node["level"] == 1:
                expected = "adult_content" if node["axis"] == "content" else None
                self.assertEqual(node["parent_id"], expected, node["id"])
        self.assertEqual(
            {n["category_id"] for n in nodes.values() if n["axis"] == "content"}
            - {None},
            {"BODY", "ACT", "KINK", "PRODUCT"},
        )
        self.assertEqual(nodes["private_show"]["axis"], "format")
        self.assertEqual(nodes["orientation"]["axis"], "identity")
        self.assertEqual(nodes["tip_goal"]["parent_id"], "monetization")

    def test_axes_are_validated_so_policy_labels_cannot_be_mixed_in(self):
        original = build.load_taxonomy()
        changes = (
            ("lingerie", {"axis": "policy"}),
            ("private_show", {"axis": "content"}),
            ("performance_format", {"axis": "content"}),
        )
        for node_id, change in changes:
            taxonomy = copy.deepcopy(original)
            next(n for n in taxonomy["nodes"] if n["id"] == node_id).update(change)
            with self.subTest(node=node_id, change=change), self.assertRaises(
                ValueError
            ):
                build.validate_taxonomy(taxonomy)
        # A non-content category may not be attached under the content root.
        taxonomy = copy.deepcopy(original)
        next(n for n in taxonomy["nodes"] if n["id"] == "performance_format")[
            "parent_id"
        ] = "adult_content"
        taxonomy["edges"].append(
            {
                "source": "performance_format",
                "predicate": "IS_A",
                "target": "adult_content",
            }
        )
        with self.assertRaises(ValueError):
            build.validate_taxonomy(taxonomy)

    def test_relabeling_and_reparenting_do_not_require_new_ids(self):
        taxonomy = copy.deepcopy(build.load_taxonomy())
        node = next(n for n in taxonomy["nodes"] if n["id"] == "lingerie")
        node["label"] = "Intimate clothing"
        build.validate_taxonomy(taxonomy)
        classifier = build.RoomClassifier(taxonomy, build.load()[1])
        result = classifier.classify_room("one", "lingerie", [])
        self.assertEqual(result["assignments"][0]["concept_id"], "lingerie")

    def test_invalid_taxonomy_levels_categories_and_edges_rejected(self):
        original = build.load_taxonomy()
        for change in ({"level": 4}, {"category_id": "KINK"}, {"parent_id": "missing"}):
            taxonomy = copy.deepcopy(original)
            node = next(n for n in taxonomy["nodes"] if n["id"] == "lingerie")
            node.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build.validate_taxonomy(taxonomy)
        taxonomy = copy.deepcopy(original)
        taxonomy["edges"] = [
            edge for edge in taxonomy["edges"] if edge["source"] != "lingerie"
        ]
        with self.assertRaises(ValueError):
            build.validate_taxonomy(taxonomy)

    def test_l0_root_is_optional(self):
        taxonomy = copy.deepcopy(build.load_taxonomy())
        taxonomy["nodes"] = [n for n in taxonomy["nodes"] if n["level"] != 0]
        for node in taxonomy["nodes"]:
            if node["level"] == 1:
                node["parent_id"] = None
        taxonomy["edges"] = [
            e for e in taxonomy["edges"] if e["target"] != "adult_content"
        ]
        build.validate_taxonomy(taxonomy)
        result = build.RoomClassifier(taxonomy, build.load()[1]).classify_room(
            "one", "latex", []
        )
        self.assertEqual(result["category_ids"], ["KINK"])
        self.assertNotIn("adult_content", result["inferred_ancestors"])


class RoomTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = build.load_taxonomy()
        self.parents, self.rows = build.load()
        self.classifier = build.RoomClassifier(self.taxonomy, self.rows)

    def test_multiple_branches_and_grouped_raw_evidence(self):
        result = self.classifier.classify_room(
            "one", "Private show with cosplay and LATEX", ["latex"]
        )
        assignments = {a["concept_id"]: a for a in result["assignments"]}
        self.assertEqual(
            set(assignments), {"private_show", "costume_roleplay", "latex"}
        )
        self.assertEqual(result["category_ids"], ["FORMAT", "KINK"])
        self.assertEqual(result["taxonomy_version"], self.taxonomy["version"])
        self.assertEqual(len(assignments["latex"]["evidence"]), 2)
        self.assertEqual(
            assignments["latex"]["evidence"][0]["text"],
            "Private show with cosplay and LATEX",
        )
        self.assertEqual(assignments["latex"]["evidence"][1]["field"], "tag_1")
        self.assertTrue(assignments["latex"]["requires_context_review"])
        self.assertIn("material_interest", result["inferred_ancestors"])
        self.assertNotIn("full_nudity", assignments)

    def test_independent_dimensions_do_not_imply_adult_content(self):
        for subject, category, axis in (
            ("Private show", "FORMAT", "format"),
            ("tip goal tonight", "COMMERCE", "commerce"),
            ("chat", "SOCIAL", "interaction"),
        ):
            with self.subTest(subject=subject):
                result = self.classifier.classify_room("one", subject, [])
                self.assertEqual(result["category_ids"], [category])
                self.assertEqual(result["axes"], [axis])
                self.assertEqual(result["assignments"][0]["axis"], axis)
                self.assertNotIn("adult_content", result["inferred_ancestors"])
        result = self.classifier.classify_room("one", "latex", [])
        self.assertEqual(result["axes"], ["content"])
        self.assertIn("adult_content", result["inferred_ancestors"])

    def test_tip_goal_is_monetization_not_a_format(self):
        result = self.classifier.classify_room("one", "tip goal", [])
        self.assertEqual([a["concept_id"] for a in result["assignments"]], ["tip_goal"])

    def test_self_declared_identity_does_not_imply_adult_content(self):
        result = self.classifier.classify_room(
            "one", "", [], self_declared_concepts=["orientation"]
        )
        self.assertEqual(result["axes"], ["identity"])
        self.assertNotIn("adult_content", result["inferred_ancestors"])

    def test_broad_assignments_do_not_invent_specific_descendants(self):
        result = self.classifier.classify_room("one", "bdsm", [])
        self.assertEqual([a["concept_id"] for a in result["assignments"]], ["bdsm"])
        self.assertEqual(result["assignments"][0]["level"], 2)
        self.assertEqual(result["category_ids"], ["KINK"])

    def test_identity_requires_explicit_self_declaration(self):
        rows = self.rows + [
            dict(self.rows[0], keyword="identity phrase", concept_id="orientation")
        ]
        classifier = build.RoomClassifier(self.taxonomy, rows)
        self.assertEqual(
            classifier.classify_room("one", "identity phrase", [])["assignments"], []
        )
        result = classifier.classify_room(
            "one", "", [], self_declared_concepts=["orientation"]
        )
        self.assertEqual(result["category_ids"], ["IDENTITY"])
        self.assertEqual(
            result["assignments"][0]["evidence"][0]["source"], "self_declared"
        )
        with self.assertRaises(ValueError):
            classifier.classify_room("one", "", [], self_declared_concepts=["latex"])

    def test_category_counts_deduplicate_rooms_and_multiple_concepts(self):
        rooms = [
            self.classifier.classify_room("one", "bondage latex", []),
            self.classifier.classify_room("one", "lingerie", []),
            self.classifier.classify_room("two", "latex", []),
        ]
        counts = build.count_rooms_by_category(rooms)
        self.assertEqual(counts["KINK"], 2)
        self.assertEqual(counts["BODY"], 1)
        self.assertEqual(counts["FORMAT"], 0)
        self.assertEqual(set(counts), CATEGORIES)

    def test_root_is_not_a_room_classification_concept(self):
        rows = [dict(self.rows[0], concept_id="adult_content")]
        with self.assertRaises(ValueError):
            build.RoomClassifier(self.taxonomy, rows)

    def test_mixed_taxonomy_versions_are_not_silently_aggregated(self):
        room = self.classifier.classify_room("one", "latex", [])
        old_room = dict(room, taxonomy_version="1.0.0")
        with self.assertRaises(ValueError):
            build.count_rooms_by_category([room, old_room])

    def test_aggregation_requires_a_version_on_every_record(self):
        room = self.classifier.classify_room("one", "latex", [])
        for version in (None, "", 4):
            with self.subTest(version=version), self.assertRaises(ValueError):
                build.count_rooms_by_category(
                    [dict(room, taxonomy_version=version), room]
                )

    def test_sample_room_output(self):
        expected = json.loads(
            (build.ROOT / "sample_room_output.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            self.classifier.classify_room(
                expected["room_id"], expected["subject"], expected["tags"]
            ),
            expected,
        )

    def test_room_ids_are_required_and_tags_still_limited_to_five(self):
        with self.assertRaises(ValueError):
            self.classifier.classify_room("", "latex", [])
        result = self.classifier.classify_room("one", "", ["", "", "", "", "", "latex"])
        self.assertEqual(result["assignments"], [])
        self.assertEqual(len(result["tags"]), 5)


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = build.load_taxonomy()
        self.nodes = {node["id"]: node for node in self.taxonomy["nodes"]}
        self.parents, self.rows = build.load()
        self.classifier = build.RoomClassifier(self.taxonomy, self.rows)

    def test_generic_category_is_standalone_three_level_branch(self):
        root = self.nodes["generic_reference"]
        self.assertEqual(
            (root["category_id"], root["axis"], root["parent_id"]),
            ("GENERIC", "generic", None),
        )
        for leaf in ("explicit_general", "profanity", "brand_reference"):
            self.assertEqual(self.nodes[leaf]["level"], 3)
            self.assertEqual(self.nodes[leaf]["category_id"], "GENERIC")

    def test_ldnoobw_terms_map_to_related_categories_or_generic(self):
        for subject, concept, review in (
            ("dildo", "adult_toy", False),
            ("shibari", "bondage", False),
            ("ass", "nudity", True),
            ("porn", "explicit_general", False),
            ("bullshit", "profanity", False),
        ):
            with self.subTest(subject=subject):
                result = self.classifier.classify_room("one", subject, [])
                (assignment,) = result["assignments"]
                self.assertEqual(assignment["concept_id"], concept)
                self.assertEqual(assignment["requires_context_review"], review)
                self.assertEqual(assignment["evidence"][0]["source"], "ldnoobw_en")
        result = self.classifier.classify_room("one", "porn", [])
        self.assertNotIn("adult_content", result["inferred_ancestors"])

    def test_policy_terms_are_kept_out_of_the_matching_vocabulary(self):
        path = build.ROOT / "policy_candidates_ldnoobw_en.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            policy = list(csv.DictReader(stream))
        self.assertTrue(policy)
        keywords = {build.norm(row["keyword"]) for row in self.rows}
        for row in policy:
            self.assertEqual(row["review_status"], "policy_review_required")
            self.assertNotIn(build.norm(row["keyword"]), keywords)
        self.assertEqual(
            self.classifier.classify_room("one", "jailbait", [])["assignments"], []
        )

    def test_vocabulary_has_no_duplicates_or_identity_mappings(self):
        keys = [
            (build.norm(r["keyword"]), r["language"], r["concept_id"])
            for r in self.rows
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(
            [
                r
                for r in self.rows
                if self.nodes[r["concept_id"]]["category_id"] == "IDENTITY"
            ]
        )

    def test_duplicate_vocabulary_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate vocabulary row"):
            build.check_vocabulary(self.rows + self.rows[:1])


if __name__ == "__main__":
    unittest.main()
