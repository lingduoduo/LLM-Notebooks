import json
import tempfile
import unittest
from pathlib import Path

import taxonomy_corpus  # noqa: F401  (puts ontology dir on sys.path)
from evaluate import prediction_row, rank_metrics, run_baselines, run_decisionrag, write_report
from judge import FakeJudge
from navigate import Ranked, Result
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


if __name__ == "__main__":
    unittest.main()
