import unittest
from benchmark import seed_examples
from judge import DemoJudge, exact_score_agreement
from generation import DemoGenerator, generate_with_guardrail
from monitoring import check_drift


class WorkflowTests(unittest.TestCase):
    def test_metrics_reject_truncated_pairs(self):
        with self.assertRaises(ValueError):
            exact_score_agreement(seed_examples(), [])

    def test_empty_monitoring_is_not_healthy(self):
        with self.assertRaises(ValueError):
            check_drift([], [])

    def test_false_accepts_trigger_alert(self):
        examples = [seed_examples()[2]]
        result = DemoJudge().evaluate(seed_examples()[0])
        self.assertTrue(check_drift(examples, [result]).drifted)

    def test_negative_retry_budget_rejected(self):
        with self.assertRaises(ValueError):
            generate_with_guardrail(seed_examples()[0], DemoGenerator(), DemoJudge(), -1)

    def test_retry_and_fallback(self):
        outcome = generate_with_guardrail(seed_examples()[0], DemoGenerator(), DemoJudge())
        self.assertTrue(outcome.passed)
        self.assertEqual(len(outcome.traces), 2)
        outcome = generate_with_guardrail(seed_examples()[0], DemoGenerator(), DemoJudge(), 0)
        self.assertTrue(outcome.used_fallback)
        self.assertFalse(outcome.passed)


if __name__ == '__main__':
    unittest.main()
