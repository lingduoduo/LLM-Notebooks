from copy import deepcopy
import unittest
from benchmark import seed_examples
from generation import DemoGenerator, generate_with_guardrail
from judge import DemoJudge
from llm_backend import context, LLMJudge
from models import Item


class SimilarityTests(unittest.TestCase):
    def example(self):
        ex = deepcopy(seed_examples()[0])
        ex.user.recently_watched = ['My Secret Santa']
        ex.item = Item('A Winter Beginning', [], genres=['holiday romance'],
                       tones=['funny', 'heartfelt'], themes=['love', 'new beginnings'])
        ex.reference = Item('My Secret Santa', [], genres=['holiday romance'],
                            tones=['funny', 'heartfelt'], themes=['love', 'new beginnings'])
        ex.explanation = 'A funny, heartfelt holiday romance about love and new beginnings, much like “My Secret Santa.”'
        return ex

    def test_supported_similarity(self):
        ex = self.example()
        self.assertTrue(DemoJudge().evaluate(ex).passed)
        outcome = generate_with_guardrail(ex, DemoGenerator(), DemoJudge())
        self.assertTrue(outcome.passed)
        self.assertIn('My Secret Santa', outcome.served_text)
        self.assertNotIn('NBA', outcome.served_text)

    def test_unwatched_reference_rejected(self):
        ex = self.example()
        ex.user.recently_watched = []
        self.assertFalse(DemoJudge().evaluate(ex).passed)
        self.assertTrue(generate_with_guardrail(ex, DemoGenerator(), DemoJudge()).used_fallback)

    def test_unshared_attribute_rejected(self):
        ex = self.example()
        ex.reference.tones = ['heartfelt']
        self.assertFalse(DemoJudge().evaluate(ex).passed)

    def test_no_overlap_falls_back(self):
        ex = self.example()
        ex.reference = Item('My Secret Santa', [], genres=['horror'])
        self.assertTrue(generate_with_guardrail(ex, DemoGenerator(), DemoJudge()).used_fallback)

    def test_missing_or_multiple_reference_rejected(self):
        for text in ['A funny holiday romance.', 'A funny holiday romance like “My Secret Santa” and “Other Film”.']:
            ex = self.example()
            ex.explanation = text
            self.assertFalse(DemoJudge().evaluate(ex).passed)

    def test_llm_receives_evidence_and_structural_gate(self):
        ex = self.example()
        self.assertEqual(context(ex)['reference']['title'], 'My Secret Santa')
        self.assertIn('shared_attributes', context(ex))
        class Client:
            def complete(self, system, payload):
                return dict(groundedness=2, relevance=2, privacy_safety=2, clarity=2,
                            unsupported_claims=[], rationale='Good')
        ex.user.recently_watched = []
        self.assertFalse(LLMJudge(Client()).evaluate(ex).passed)

    def test_similarity_cli(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(__file__).with_name('demo.py')
            run = subprocess.run([sys.executable, str(script), '--scenario', 'similarity',
                                  '--output-dir', tmp], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((Path(tmp) / 'report.json').read_text())
            self.assertTrue(report['generation']['passed'])
            self.assertIn('My Secret Santa', report['generation']['served_text'])
            self.assertEqual(report['scenario'], 'similarity')
