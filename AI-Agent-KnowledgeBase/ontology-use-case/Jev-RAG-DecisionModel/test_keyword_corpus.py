import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from judge import FakeJudge
from keyword_corpus import is_stale, keyword_config, load_keyword_tree, to_concepts
from navigate import Ranked, Result
from taxonomy_corpus import is_returnable, load_tree, tag_room, taxonomy_config
from tree import walk
from tree_builder import build_keyword_tree

CSV_BYTES = b"keyword,concept_id\na,c1\n"


def leaf(i, kw, cid):
    return {"id": f"kw{i:03d}", "keyword": kw, "concept_id": cid}


TREE = {
    "meta": {"keywords_sha256": hashlib.sha256(CSV_BYTES).hexdigest()},
    "tree": {"id": "__root__", "description": "All keywords", "description_source": "fallback",
             "children": [
                 {"id": "c0", "description": "Alpha things", "description_source": "llm",
                  "children": [leaf(0, "a", "c1"), leaf(1, "b", "c1"), leaf(2, "x", "")]},
                 {"id": "c1", "description": "Gamma things", "description_source": "llm",
                  "children": [leaf(3, "c", "c2")]},
             ]},
}


def res(*ids):
    return Result([Ranked(i, 0.9) for i in ids], [], 0.0)


class KeywordCorpusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "t.json"
        self.path.write_text(json.dumps(TREE), encoding="utf-8")
        self.meta, self.root = load_keyword_tree(self.path)

    def test_load_tree_nodes_and_payload(self):
        nodes = {n.id: n for n in walk(self.root)}
        self.assertEqual(self.root.id, "__root__")
        self.assertEqual(nodes["c0"].description, "Alpha things")
        self.assertEqual(nodes["kw000"].description, "a")
        self.assertEqual(dict(nodes["kw000"].payload), {"keyword": "a", "concept_id": "c1"})
        self.assertIn("keywords_sha256", self.meta)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_keyword_tree(Path(self.tmp.name) / "nope.json")

    def test_is_stale(self):
        csv = Path(self.tmp.name) / "k.csv"
        csv.write_bytes(CSV_BYTES)
        self.assertFalse(is_stale(self.meta, csv))
        csv.write_bytes(CSV_BYTES + b"b,c2\n")
        self.assertTrue(is_stale(self.meta, csv))

    def test_to_concepts_maps_dedups_and_counts_unscorable(self):
        out = to_concepts(res("kw000", "kw001", "kw002", "kw003"), self.root)
        self.assertEqual(out, (["c1", "c2"], 1))

    def test_to_concepts_self_declared_first(self):
        r = res("kw003")
        r.ranked.insert(0, Ranked("c1", None, source="self_declared"))
        self.assertEqual(to_concepts(r, self.root), (["c1", "c2"], 0))

    def test_all_unscorable_gives_empty_list(self):
        self.assertEqual(to_concepts(res("kw002"), self.root), ([], 1))

    def test_tag_room_flat_uses_config_returnable(self):
        r = tag_room(self.root, FakeJudge({"kw000": 0.9, "c0": 0.9}), keyword_config(),
                     "alpha", [], mode="flat")
        self.assertTrue(r.ranked)
        self.assertTrue(all(x.id.startswith("kw") for x in r.ranked))

    def test_taxonomy_flat_unchanged(self):
        _, troot = load_tree()
        units = {n.id for n in walk(troot) if is_returnable(n)}
        r = tag_room(troot, FakeJudge({"bdsm": 0.9, "lingerie": 0.8, "roleplay": 0.7}),
                     taxonomy_config(), "x", [], mode="flat")
        self.assertTrue(r.ranked)
        self.assertTrue({x.id for x in r.ranked} <= units)

    def test_builder_records_csv_sha(self):
        rows = [{"keyword": "a", "concept_id": "c1"}]
        t = build_keyword_tree(rows, lambda ts: [[1.0]] * len(ts),
                               lambda v, k, s: [0] * len(v), lambda k: "d",
                               csv_sha256="abc")
        self.assertEqual(t["meta"]["keywords_sha256"], "abc")
        t = build_keyword_tree(rows, lambda ts: [[1.0]] * len(ts),
                               lambda v, k, s: [0] * len(v), lambda k: "d")
        self.assertIsNone(t["meta"]["keywords_sha256"])


if __name__ == "__main__":
    unittest.main()
