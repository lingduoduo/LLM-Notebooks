import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (prediction_row, rank_metrics, run_baselines, run_decisionrag,
                      run_keyword_tree, write_report)
from judge import FakeJudge
from navigate import Ranked, Result
from keyword_corpus import keyword_config
from taxonomy_corpus import taxonomy_config


class EvaluateTest(unittest.TestCase):
    def test_prediction_rows_pass_existing_evaluator(self):
        run = run_decisionrag(FakeJudge(), taxonomy_config())
        self.assertEqual(run["report"]["count"], 28)
        self.assertEqual(len(run["rows"]), 28)
        self.assertEqual(run["cost_usd"], 0.0)

    def test_prediction_row_shape(self):
        res = Result([Ranked("x", 0.7), Ranked("y", None, source="self_declared")], [], 0.1)
        row = prediction_row({"id": "c1"}, res, "1.0")
        self.assertEqual(row["concept_ids"], ["x", "y"])
        self.assertEqual(row["scores"], {"x": 0.7})
        self.assertEqual(row["cost_usd"], 0.0)

    def test_rank_metrics(self):
        cases = [{"id": "1", "expected_concepts": ["a"]},
                 {"id": "2", "expected_concepts": []}]
        rows = [{"case_id": "1", "concept_ids": ["b", "a"]},
                {"case_id": "2", "concept_ids": ["z"]}]
        m = rank_metrics(cases, rows)
        self.assertEqual(m["hit_at_1"], 0)
        self.assertEqual(m["mrr"], 0.5)
        self.assertEqual(m["count"], 1)

    def test_baselines_load(self):
        runs = run_baselines()
        self.assertEqual([r["name"] for r in runs], ["keyword", "minilm"])
        for r in runs:
            self.assertEqual(r["report"]["count"], 28)
        self.assertIsNone(runs[0]["rank"])

    def test_write_report_markdown(self):
        runs = run_baselines() + [run_decisionrag(FakeJudge(), taxonomy_config())]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "results.json"
            write_report(runs, out)
            self.assertEqual(len(json.loads(out.read_text())), 3)
            self.assertIn("Hit@1", out.with_suffix(".md").read_text())


    def test_write_report_latency_and_cost_columns(self):
        report = run_baselines()[0]["report"]
        runs = [{"name": "dr", "report": report, "rank": None, "cost_usd": 0.0123, "requests": 3,
                 "rows": [{"latency_s": 1.0}, {"latency_s": 2.0}]},
                {"name": "kw", "report": report, "rank": None, "cost_usd": 0.0, "requests": 0,
                 "rows": []}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            write_report(runs, out)
            lines = out.with_suffix(".md").read_text().splitlines()
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        self.assertIn("latency (s/room)", header)
        self.assertIn("nominal cost (¢)", header)
        col = header.index("latency (s/room)")
        dr = [c.strip() for c in lines[2].strip("|").split("|")]
        kw = [c.strip() for c in lines[3].strip("|").split("|")]
        self.assertEqual(dr[col], "1.500")
        self.assertEqual(kw[col], "-")
        self.assertEqual(dr[header.index("nominal cost (¢)")], "1.23")

    def test_write_report_estimated_cost_and_models(self):
        report = run_baselines()[0]["report"]
        runs = [{"name": "dr", "report": report, "rank": None, "cost_usd": 0.01, "requests": 3,
                 "rows": [], "provider": "typesafe", "models_seen": ["jev-1.13.0"],
                 "cost_estimated": True}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            write_report(runs, out)
            md = out.with_suffix(".md").read_text()
        self.assertIn("estimated cost (¢)", md.splitlines()[0])
        self.assertIn("jev-1.13.0", md)

    def _tree_file(self, d):
        leaf = lambda i, k, c: {"id": i, "keyword": k, "concept_id": c}
        tree = {"id": "root", "description": "root", "children": [
            {"id": "c1", "description": "worn items", "children": [
                leaf("k1", "lingerie", "lingerie"), leaf("k2", "latex", "latex")]},
            {"id": "c2", "description": "shows", "children": [
                leaf("k3", "private show", "private_show"), leaf("k4", "misc", "")]}]}
        p = Path(d) / "tree.json"
        p.write_text(json.dumps({"meta": {"keywords_sha256": "x"}, "tree": tree}))
        return p

    def test_run_keyword_tree_fake(self):
        with tempfile.TemporaryDirectory() as d:
            run = run_keyword_tree(FakeJudge(), keyword_config(), tree_path=self._tree_file(d))
        self.assertEqual(run["report"]["count"], 28)
        self.assertEqual(run["name"], "decisionrag-keyword-tree-navigate")
        self.assertEqual(run["tree"], "keywords")
        self.assertIsInstance(run["unscorable_hits"], int)
        self.assertIsInstance(run["rooms_with_unscorable"], int)
        self.assertEqual(len(run["rows"]), 28)
        for row in run["rows"]:
            self.assertIn("keywords", row)
            self.assertIn("unscorable", row)

    def test_report_has_hierarchical_column_and_unscorable_note(self):
        report = run_baselines()[0]["report"]
        runs = [{"name": "kw", "report": report, "rank": None, "cost_usd": 0.0, "requests": 0,
                 "rows": [], "unscorable_hits": 5, "rooms_with_unscorable": 2}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            write_report(runs, out)
            md = out.with_suffix(".md").read_text()
        header = [c.strip() for c in md.splitlines()[0].strip("|").split("|")]
        self.assertEqual(header[header.index("micro-F1") + 1], "hier. micro-F1")
        row = [c.strip() for c in md.splitlines()[2].strip("|").split("|")]
        self.assertEqual(row[header.index("hier. micro-F1")],
                         f"{report['hierarchical']['micro_f1']:.3f}")
        self.assertIn("Unscorable keyword hits (no concept): kw=5 in 2 rooms", md)


if __name__ == "__main__":
    unittest.main()
