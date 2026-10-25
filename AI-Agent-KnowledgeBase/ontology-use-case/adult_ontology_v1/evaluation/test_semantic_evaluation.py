"""Evaluator contracts; challenge gold is not generated from matcher predictions."""

import copy
import unittest

import utils
from evaluation.semantic_evaluation import evaluate, load_benchmark, score_labels


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = utils.load_taxonomy()
        self.benchmark = load_benchmark()

    def test_label_metrics_count_false_positives_and_misses(self):
        result = score_labels([{"a", "b"}, set()], [{"a", "c"}, {"c"}])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["micro_precision"], 1 / 3)
        self.assertEqual(result["micro_recall"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 0.4)
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(score_labels([set()], [set()])["exact_match"], 1)

    def test_perfect_predictions_and_slices(self):
        predictions = [
            {
                "case_id": case["id"],
                "taxonomy_version": self.taxonomy["version"],
                "concept_ids": case["expected_concepts"],
            }
            for case in self.benchmark["cases"]
        ]
        report = evaluate(self.benchmark, self.taxonomy, predictions=predictions)
        self.assertEqual(report["concepts"]["exact_match"], 1)
        self.assertEqual(report["hierarchical"]["micro_f1"], 1)
        self.assertEqual(report["categories"]["micro_f1"], 1)
        self.assertEqual(report["taxonomy_negative_false_positive_rate"], 0)
        self.assertEqual(
            sum(s["count"] for s in report["slices"].values()), report["count"]
        )

    def test_parent_prediction_gets_partial_hierarchical_credit(self):
        benchmark = copy.deepcopy(self.benchmark)
        benchmark["cases"] = [
            dict(benchmark["cases"][0], expected_concepts=["lingerie"])
        ]
        predictions = [
            {
                "case_id": benchmark["cases"][0]["id"],
                "taxonomy_version": self.taxonomy["version"],
                "concept_ids": ["clothing"],
            }
        ]
        report = evaluate(benchmark, self.taxonomy, predictions=predictions)
        self.assertEqual(report["concepts"]["micro_f1"], 0)
        self.assertGreater(report["hierarchical"]["micro_f1"], 0)
        self.assertLess(report["hierarchical"]["micro_f1"], 1)
        self.assertEqual(report["categories"]["micro_f1"], 1)

    def test_missing_duplicate_unknown_and_wrong_version_predictions_fail(self):
        cases = self.benchmark["cases"]
        good = [
            {
                "case_id": c["id"],
                "taxonomy_version": self.taxonomy["version"],
                "concept_ids": c["expected_concepts"],
            }
            for c in cases
        ]
        for predictions in (
            good[:-1],
            good + [good[0]],
            [dict(good[0], concept_ids=["sex"])] + good[1:],
            [dict(good[0], taxonomy_version="old")] + good[1:],
        ):
            with self.subTest(predictions=predictions[:1]), self.assertRaises(
                ValueError
            ):
                evaluate(self.benchmark, self.taxonomy, predictions=predictions)

    def test_gold_version_labels_and_cross_split_leakage_fail(self):
        for field, value in [("taxonomy_version", "old")]:
            benchmark = dict(self.benchmark, **{field: value})
            with self.assertRaises(ValueError):
                evaluate(benchmark, self.taxonomy, predictions=[])
        for change in (
            {"expected_concepts": ["not_a_concept"]},
            {"expected_concepts": ["adult_content"]},
        ):
            benchmark = copy.deepcopy(self.benchmark)
            benchmark["cases"][0].update(change)
            with self.assertRaises(ValueError):
                evaluate(benchmark, self.taxonomy, predictions=[])
        for shared_group in (True, False):
            benchmark = copy.deepcopy(self.benchmark)
            extra = dict(benchmark["cases"][0], id="leaked", split="train")
            if not shared_group:
                extra["group_id"] = "different-group-same-text"
            benchmark["cases"].append(extra)
            with self.assertRaisesRegex(ValueError, "split"):
                evaluate(benchmark, self.taxonomy, predictions=[])

    def test_baseline_runs_without_model_downloads(self):
        classifier = utils.RoomClassifier(self.taxonomy, utils.load()[1])
        report = evaluate(self.benchmark, self.taxonomy, classifier=classifier)
        self.assertEqual(report["count"], len(self.benchmark["cases"]))
        self.assertEqual(report["prediction_source"], "keyword_baseline")
        self.assertLessEqual(report["concepts"]["micro_f1"], 1)
        self.assertGreaterEqual(report["taxonomy_negative_false_positive_rate"], 0)


if __name__ == "__main__":
    unittest.main()
