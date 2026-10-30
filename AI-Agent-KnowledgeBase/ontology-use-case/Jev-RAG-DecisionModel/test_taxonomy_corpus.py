import unittest

from judge import FakeJudge
from navigate import Ranked, Result
from taxonomy_corpus import (is_returnable, load_tree, room_query, tag_room,
                             taxonomy_config)
from tree import walk
import utils


class TaxonomyCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy, cls.root = load_tree()
        cls.nodes = {n.id: n for n in walk(cls.root)}

    def test_top_level_is_nine_categories(self):
        self.assertEqual(len(self.root.children), 9)
        self.assertFalse(any("adult_content" in i for i in self.nodes))

    def test_fifty_leaves_and_returnable_groups(self):
        self.assertEqual(sum(1 for n in self.nodes.values() if n.is_leaf), 50)
        self.assertTrue(is_returnable(self.nodes["bdsm"]))
        self.assertTrue(is_returnable(self.nodes["roleplay"]))
        for c in self.root.children:
            self.assertFalse(is_returnable(c))

    def test_description_has_path_and_examples(self):
        d = self.nodes["lingerie"].description
        self.assertTrue(d.startswith("Lingerie — Body & Appearance > "), d)
        self.assertIn("e.g.", d)
        excluded = {r["keyword"] for r in utils.load_keyword_rows()
                    if r["review_status"] == "excluded"}
        self.assertTrue(excluded)
        for n in self.nodes.values():
            for kw in excluded:
                self.assertNotIn(f" {kw},", n.description + ",")
                self.assertFalse(n.description.endswith(f" {kw})"))

    def test_room_query_empty(self):
        self.assertEqual(room_query("", []), "Subject: \nTags: (none)")
        self.assertIsInstance(
            tag_room(self.root, FakeJudge(), taxonomy_config(), "", []), Result)

    def test_room_query_first_five_tags(self):
        q = room_query("s", list("abcdefg"))
        self.assertIn("a, b, c, d, e", q)
        self.assertNotIn("f", q.split("Tags:")[1])

    def test_self_declared_first_and_deduped(self):
        scores = {"identity_audience": 1, "self_description": 1, "orientation": 1,
                  "body_appearance": 1, "clothing": 1, "lingerie": 1}
        res = tag_room(self.root, FakeJudge(scores), taxonomy_config(), "x", [],
                       self_declared=["orientation"])
        self.assertEqual(res.ranked[0], Ranked("orientation", None, None, "self_declared"))
        self.assertEqual([r.id for r in res.ranked].count("orientation"), 1)
        self.assertIn("lingerie", [r.id for r in res.ranked])

    def test_flat_and_separate_modes(self):
        for mode in ("flat", "separate"):
            res = tag_room(self.root, FakeJudge({"lingerie": 1}), taxonomy_config(),
                           "x", [], mode=mode)
            self.assertEqual(res.ranked[0].id, "lingerie")


if __name__ == "__main__":
    unittest.main()
