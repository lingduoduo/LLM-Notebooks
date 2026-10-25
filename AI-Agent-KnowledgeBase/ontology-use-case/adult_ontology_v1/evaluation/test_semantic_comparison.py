"""Compare real pipeline behavior using a deterministic encoder, no model weights."""

import copy
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

import utils
from evaluation.compare_semantic import main, run_comparison
from evaluation.semantic_evaluation import load_benchmark


class FixedEncoder:
    def encode(self, texts):
        return [
            [1.0, 0.0] if "private" in text.lower() else [0.0, 1.0] for text in texts
        ]


class ComparisonTests(unittest.TestCase):
    def test_predictions_do_not_depend_on_gold_labels(self):
        taxonomy = utils.load_taxonomy()
        benchmark = load_benchmark()
        benchmark["cases"] = benchmark["cases"][:1]
        first, predictions, prototypes = run_comparison(
            benchmark, taxonomy, utils.load()[1], FixedEncoder()
        )
        changed = copy.deepcopy(benchmark)
        changed["cases"][0]["expected_concepts"] = []
        second, other_predictions, other_prototypes = run_comparison(
            changed, taxonomy, utils.load()[1], FixedEncoder()
        )
        self.assertEqual(predictions, other_predictions)
        self.assertEqual(prototypes, other_prototypes)
        self.assertEqual(
            first["prototypes_sha256_canonical"], second["prototypes_sha256_canonical"]
        )
        self.assertNotEqual(
            first["semantic"]["concepts"], second["semantic"]["concepts"]
        )

    def test_invalid_settings_fail_before_model_loading(self):
        with patch(
            "evaluation.compare_semantic.CachedMiniLMEncoder"
        ) as encoder, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--threshold", "nan", "--output-dir", "/unused"])
        self.assertEqual(error.exception.code, 2)
        encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
