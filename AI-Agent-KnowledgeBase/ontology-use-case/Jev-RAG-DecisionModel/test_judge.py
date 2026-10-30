import unittest

from judge import FakeJudge
from tree import Node, walk


class TreeTest(unittest.TestCase):
    def test_walk_is_preorder(self):
        r = Node("r", "root", (Node("a", "A", (Node("a1", "A1"), Node("a2", "A2"))), Node("b", "B")))
        self.assertEqual([n.id for n in walk(r)], ["a", "a1", "a2", "b"])

    def test_is_leaf(self):
        self.assertTrue(Node("x", "X").is_leaf)
        self.assertFalse(Node("r", "R", (Node("x", "X"),)).is_leaf)


class FakeJudgeTest(unittest.TestCase):
    a = Node("a", "Latex")
    b = Node("b", "Gaming stream")

    def test_fake_scores_lookup(self):
        out = FakeJudge({"a": 0.9}).binary("q", [self.a, self.b], "?")
        self.assertEqual(out, {"a": 0.9, "b": 0.0})

    def test_fake_overlap(self):
        out = FakeJudge().binary("latex outfit", [self.a, self.b], "?")
        self.assertEqual(out, {"a": 1.0, "b": 0.0})

    def test_fake_choice_sums_to_one(self):
        out = FakeJudge({"a": 0.6, "b": 0.2}).choice("q", [self.a, self.b], "?")
        self.assertAlmostEqual(sum(out.values()), 1.0)
        self.assertGreater(out["a"], out["b"])

    def test_fake_choice_uniform_when_all_zero(self):
        out = FakeJudge({}).choice("q", [self.a, self.b], "?")
        self.assertEqual(out, {"a": 0.5, "b": 0.5})

    def test_fake_records_calls(self):
        j = FakeJudge()
        j.binary("q", [self.a, self.b], "?")
        self.assertEqual(j.calls, [("binary", ("a", "b"))])


if __name__ == "__main__":
    unittest.main()
