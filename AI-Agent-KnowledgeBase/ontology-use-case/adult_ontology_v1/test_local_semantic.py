"""Small-vector baseline tests; model weights are used only by the explicit CLI."""

import math
import unittest

import build
from local_semantic import SemanticBaseline, build_prototypes


class SmallEncoder:
    """Known semantic vectors: labels come from meanings, never evaluation gold."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "private show" in lower or "one paying viewer" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "lingerie" in lower:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = build.load_taxonomy()
        self.prototypes = [
            {"concept_id": "private_show", "text": "Private Show"},
            {"concept_id": "lingerie", "text": "Lingerie"},
        ]

    def baseline(self, **kwargs):
        return SemanticBaseline(
            self.taxonomy, SmallEncoder(), prototypes=self.prototypes, **kwargs
        )

    def test_metadata_prototypes_are_deterministic_and_exclude_identity_and_root(self):
        prototypes = build_prototypes(self.taxonomy)
        self.assertEqual(prototypes, build_prototypes(self.taxonomy))
        nodes = {node["id"]: node for node in self.taxonomy["nodes"]}
        self.assertTrue(
            all(
                nodes[p["concept_id"]]["level"] > 0
                and nodes[p["concept_id"]]["category_id"] != "IDENTITY"
                for p in prototypes
            )
        )
        private = next(p for p in prototypes if p["concept_id"] == "private_show")
        self.assertIn("Private Show", private["text"])
        self.assertIn("Broadcast Mode", private["text"])

    def test_paraphrase_and_multilabel_tags_with_raw_score_evidence(self):
        result = self.baseline().predict(
            "A session for one paying viewer", ["lingerie"]
        )
        self.assertEqual(result["concept_ids"], ["lingerie", "private_show"])
        self.assertEqual(result["scores"]["private_show"], 1)
        self.assertEqual(result["evidence"]["lingerie"]["field"], "tag_1")
        self.assertTrue(result["requires_context_review"])

    def test_empty_and_below_threshold_inputs_abstain(self):
        model = self.baseline()
        self.assertEqual(model.predict("", [])["concept_ids"], [])
        self.assertEqual(model.predict("Gardening weather", [])["concept_ids"], [])

    def test_top_k_and_five_tag_limit(self):
        model = self.baseline(max_labels=1)
        self.assertEqual(
            len(model.predict("Private show", ["lingerie"])["concept_ids"]), 1
        )
        self.assertEqual(
            model.predict("", ["", "", "", "", "", "lingerie"])["concept_ids"], []
        )

    def test_identity_is_never_predicted_from_text(self):
        model = self.baseline()
        self.assertEqual(model.predict("Orientation", [])["concept_ids"], [])
        result = model.predict("", [], self_declared_concepts=["orientation"])
        self.assertEqual(result["concept_ids"], ["orientation"])
        self.assertEqual(result["evidence"]["orientation"]["source"], "self_declared")
        self.assertNotIn("orientation", result["scores"])
        with self.assertRaises(ValueError):
            model.predict("", [], self_declared_concepts=["private_show"])

    def test_invalid_configuration_and_vectors_fail(self):
        for threshold in (-1.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.baseline(threshold=threshold)
        for maximum in (0, -1, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.baseline(max_labels=maximum)

        class BrokenEncoder:
            def encode(self, texts):
                return [[math.nan]] * len(texts)

        with self.assertRaises(ValueError):
            SemanticBaseline(self.taxonomy, BrokenEncoder(), prototypes=self.prototypes)

    def test_redundant_ancestors_are_removed_but_other_branches_remain(self):
        prototypes = [
            *self.prototypes,
            {"concept_id": "broadcast_mode", "text": "Private Show"},
        ]
        model = SemanticBaseline(self.taxonomy, SmallEncoder(), prototypes=prototypes)
        self.assertEqual(
            model.predict("Private show", ["lingerie"])["concept_ids"],
            ["lingerie", "private_show"],
        )


if __name__ == "__main__":
    unittest.main()
