import unittest

from judge import FakeJudge
from navigate import NavigationConfig, navigate, score_units
from tree import Node


def leaf(i):
    return Node(i, i)


def tree():
    return Node("root", "root", (
        Node("A", "A", (leaf("a1"), leaf("a2"))),
        Node("B", "B", (Node("b1", "b1", (leaf("b1x"), leaf("b1y"))),)),
        leaf("c"),
    ))


class ChoiceStub(FakeJudge):
    def __init__(self, scores, answer):
        super().__init__(scores)
        self.answer = answer

    def choice(self, query, candidates, question):
        self.calls.append(("choice", tuple(n.id for n in candidates)))
        return dict(self.answer)


def ids(result):
    return [r.id for r in result.ranked]


class NavigateTest(unittest.TestCase):
    def test_pruned_branch_never_judged(self):
        j = FakeJudge({"A": .9, "a1": .8, "B": .05})
        navigate(tree(), "q", j, NavigationConfig())
        seen = {i for _, c in j.calls for i in c}
        self.assertFalse(seen & {"b1", "b1x", "b1y"})

    def test_uneven_depth_returns_leaves_at_any_level(self):
        j = FakeJudge({"A": .9, "a1": .8, "c": .7, "B": .9, "b1": .9, "b1y": .6})
        r = navigate(tree(), "q", j, NavigationConfig(use_choice=False))
        self.assertEqual(set(ids(r)), {"a1", "c", "b1y"})

    def test_keep_cap_k(self):
        kids = tuple(Node(f"n{i}", "x", (leaf(f"l{i}"),)) for i in range(5))
        j = FakeJudge({f"n{i}": .9 for i in range(5)})
        r = navigate(Node("root", "r", kids), "q", j, NavigationConfig(k=4))
        self.assertEqual(len(r.trace[0]["kept"]), 4)

    def test_floor_keeps_top_one(self):
        r = navigate(tree(), "q", FakeJudge({"A": .2, "B": .15}), NavigationConfig())
        self.assertEqual(r.trace[0]["kept"], ["A"])

    def test_below_floor_stops(self):
        j = FakeJudge({"A": .05, "B": .05, "c": .05})
        r = navigate(tree(), "q", j, NavigationConfig())
        self.assertEqual(r.ranked, [])
        self.assertNotIn("choice", [m for m, _ in j.calls])
        self.assertEqual(len(j.calls), 1)

    def test_internal_node_returnable_when_no_child_passes(self):
        cfg = NavigationConfig(returnable=lambda n: n.is_leaf or n.id == "A", use_choice=False)
        r = navigate(tree(), "q", FakeJudge({"A": .9, "a1": .2, "a2": .1}), cfg)
        self.assertEqual(ids(r), ["A"])
        r = navigate(tree(), "q", FakeJudge({"A": .9, "a1": .8, "a2": .1}), cfg)
        self.assertEqual(ids(r), ["a1"])

    def test_unjudged_children_do_not_block_tagging_at_zero_threshold(self):
        cfg = NavigationConfig(k=1, tag_threshold=0.0, use_choice=False,
                               returnable=lambda n: n.is_leaf or n.id in ("A", "B"))
        j = FakeJudge({"A": .9, "B": .8, "a1": .5, "a2": .5})
        r = navigate(tree(), "q", j, cfg)
        self.assertIn("B", ids(r))
        self.assertNotIn("A", ids(r))
        self.assertFalse({i for _, c in j.calls for i in c} & {"b1"})

    def test_choice_reorders_only_top_m(self):
        j = ChoiceStub({"A": .9, "a1": .9, "a2": .8, "c": .7}, {"a2": .7, "a1": .3})
        r = navigate(tree(), "q", j, NavigationConfig(m=2))
        self.assertEqual(ids(r), ["a2", "a1", "c"])
        self.assertIsNone(r.ranked[2].p_mul)
        self.assertEqual(r.ranked[0].p_mul, .7)
        self.assertIn("choice", r.trace[-1])

    def test_choice_skipped_for_single_tag(self):
        j = FakeJudge({"A": .9, "a1": .8})
        r = navigate(tree(), "q", j, NavigationConfig())
        self.assertEqual(ids(r), ["a1"])
        self.assertNotIn("choice", [m for m, _ in j.calls])

    def test_choice_missing_option_is_zero(self):
        j = ChoiceStub({"A": .9, "a1": .9, "a2": .8, "c": .7}, {"a2": 1.0})
        r = navigate(tree(), "q", j, NavigationConfig())
        self.assertEqual(ids(r), ["a2", "a1", "c"])
        self.assertEqual(r.ranked[1].p_mul, 0.0)

    def test_score_units_separate_calls_once_per_unit(self):
        units = [leaf("x"), leaf("y"), leaf("z")]
        j = FakeJudge({"x": .9, "y": .8, "z": .7})
        score_units(units, "q", j, NavigationConfig(), separate=True)
        binaries = [c for m, c in j.calls if m == "binary"]
        self.assertEqual([len(c) for c in binaries], [1, 1, 1])
        j = FakeJudge({"x": .9, "y": .8, "z": .7})
        score_units(units, "q", j, NavigationConfig())
        self.assertEqual([c for m, c in j.calls if m == "binary"], [("x", "y", "z")])

    def test_top_r_caps_units(self):
        units = [leaf(i) for i in "wxyz"]
        j = FakeJudge({"w": .9, "x": .8, "y": .7, "z": .6})
        r = score_units(units, "q", j, NavigationConfig(r=2, use_choice=False))
        self.assertEqual(ids(r), ["w", "x"])

    def test_trace_has_latency_and_requests(self):
        j = FakeJudge({"A": .9, "a1": .9, "a2": .8})
        r = navigate(tree(), "q", j, NavigationConfig())
        for stage in r.trace:
            self.assertGreaterEqual(stage["latency_s"], 0)
            self.assertEqual(stage["requests"], 1)

    def test_requests_attribute_delta(self):
        class Counting(FakeJudge):
            requests = 10

            def binary(self, *a):
                self.requests += 3
                return super().binary(*a)

        r = navigate(tree(), "q", Counting({"A": .9, "a1": .9}), NavigationConfig(use_choice=False))
        self.assertEqual(r.trace[0]["requests"], 3)

    def test_paper_nolan_example(self):
        root = Node("root", "r", (
            Node("nolan", "n", (leaf("n1"), leaf("n2"))),
            Node("memento", "m", (leaf("m1"), leaf("m2"), leaf("m3"))),
            Node("gladiator", "g", (leaf("g1"), leaf("g2"))),
        ))
        j = ChoiceStub({"nolan": .81, "memento": .96, "gladiator": .07,
                        "m1": .9, "n2": .6, "m2": .4}, {"m1": .8, "n2": .2})
        cfg = NavigationConfig(k=2, tag_threshold=0.0, r=3, m=2)
        r = navigate(root, "q", j, cfg)
        seen = {i for _, c in j.calls for i in c}
        self.assertFalse(seen & {"g1", "g2"})
        self.assertEqual(len(r.ranked), 3)
        self.assertEqual(r.ranked[0].id, "m1")
        self.assertTrue(all(i[0] in "nm" for i in ids(r)))


if __name__ == "__main__":
    unittest.main()
