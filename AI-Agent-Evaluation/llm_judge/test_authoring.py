from copy import deepcopy
import unittest
from quality import TaskSpec, Alignment, TaskJudge, align, validate_record


class AuthoringTests(unittest.TestCase):
    def task(self):
        return TaskSpec('summary', 'Check evidence.', {'accuracy': '0=wrong, 1=partial, 2=supported'}, {'accuracy': 2})

    def candidates(self):
        from benchmark_authoring import import_experts
        return import_experts([{'id': 'expert-1', 'context': {'source': 'A fact'}, 'output': 'A fact',
                               'metadata': {}, 'expert_id': 'expert-a', 'difficulty': 'boundary',
                               'boundary_reason': 'A minimal paraphrase near the support boundary.'}], self.task())

    def rating(self, candidate, rater='rater-a', passed=True):
        from benchmark_authoring import candidate_hash
        return {'id': candidate['id'], 'candidate_hash': candidate_hash(candidate),
                'rater_id': rater, 'passed': passed, 'rationale': 'All claims follow from the evidence.'}

    def test_only_reviewed_examples_enter_benchmark(self):
        from benchmark_authoring import build_benchmark
        candidates = self.candidates()
        accepted, pending = build_benchmark(candidates, [], self.task())
        self.assertEqual(accepted, [])
        self.assertEqual(len(pending), 1)
        ratings = [self.rating(candidates[0]), self.rating(candidates[0], 'rater-b')]
        accepted, pending = build_benchmark(candidates, ratings, self.task())
        self.assertEqual(pending, [])
        self.assertTrue(accepted[0]['human_decision'])
        self.assertNotIn('human', accepted[0])
        validate_record(accepted[0], self.task())

    def test_conflicting_ratings_require_adjudication(self):
        from benchmark_authoring import build_benchmark
        candidates = self.candidates()
        ratings = [self.rating(candidates[0]), self.rating(candidates[0], 'rater-b', False)]
        accepted, pending = build_benchmark(candidates, ratings, self.task())
        self.assertEqual(accepted, [])
        self.assertEqual(pending[0]['reason'], 'disagreement')
        adjudication = self.rating(candidates[0], 'adjudicator', False)
        accepted, _ = build_benchmark(candidates, ratings, self.task(), adjudications=[adjudication])
        self.assertFalse(accepted[0]['human_decision'])

    def test_stale_empty_and_duplicate_ratings_rejected(self):
        from benchmark_authoring import build_benchmark
        candidates = self.candidates()
        for change in [{'candidate_hash': 'stale'}, {'rationale': ''}, {'passed': 'pass'}]:
            rating = {**self.rating(candidates[0]), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_benchmark(candidates, [rating], self.task())
        rating = self.rating(candidates[0])
        with self.assertRaises(ValueError):
            build_benchmark(candidates, [rating, rating], self.task())

    def test_llm_candidates_are_unlabeled_and_keep_source_context(self):
        from benchmark_authoring import augment, rating_queue
        class Client:
            model = 'author-model'
            def complete(self, system, payload):
                self.payload = payload
                return {'candidates': [{'output': 'A likely fact', 'boundary_reason': 'Adds uncertain wording.'}]}
        candidates = self.candidates()
        client = Client()
        generated = augment(candidates, self.task(), client, count=1)
        self.assertEqual(generated[0]['context'], candidates[0]['context'])
        self.assertEqual(generated[0]['group_id'], candidates[0]['group_id'])
        self.assertEqual(generated[0]['provenance']['source'], 'llm')
        self.assertNotIn('human', generated[0])
        queue = rating_queue(generated, self.task())
        self.assertNotIn('provenance', queue[0])
        self.assertNotIn('boundary_reason', queue[0])
        self.assertIsNone(queue[0]['passed'])

    def test_binary_labels_support_alignment_without_inventing_scores(self):
        task = self.task()
        stats = Alignment(task)
        stats.add(None, {'scores': {'accuracy': 2}, 'passed': True}, decision=False)
        report = stats.report()
        self.assertEqual(report['false_accept_rate'], 1)
        self.assertIsNone(report['macro_agreement'])
        self.assertEqual(report['decision_agreement'], 0)
        class Client:
            def complete(self, system, payload):
                if 'disagreements' in payload:
                    return {'accuracy': 'FIXED: 0=wrong, 1=partial, 2=supported'}
                return {'scores': {'accuracy': 2 if 'FIXED' in system else 0}, 'rationale': 'Evidence'}
        a = {'id': 'a', 'context': {}, 'output': 'a', 'metadata': {},
             'human_decision': True, 'human_rationale': 'Supported'}
        b = {**a, 'id': 'b', 'output': 'b'}
        _, report = align(TaskJudge(Client(), task), [a], [b])
        self.assertTrue(report['promoted'])

    def test_related_examples_never_cross_splits(self):
        from benchmark_authoring import split_groups
        records = [{'id': str(i), 'metadata': {'benchmark_group': str(i//2)}} for i in range(8)]
        calibration, validation = split_groups(records, 0.25, 7)
        self.assertTrue(calibration and validation)
        self.assertFalse({r['metadata']['benchmark_group'] for r in calibration} &
                         {r['metadata']['benchmark_group'] for r in validation})

    def test_cli_prepare_and_build(self):
        import json
        from dataclasses import asdict
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = Path(__file__).with_name('create_benchmark.py')
            (root/'task.json').write_text(json.dumps(asdict(self.task())))
            expert = {'id': 'e1', 'context': {'source': 'A fact'}, 'output': 'A fact', 'metadata': {},
                      'expert_id': 'expert', 'difficulty': 'hard', 'boundary_reason': 'Minimal evidence'}
            (root/'experts.jsonl').write_text(json.dumps(expert) + '\n')
            run = subprocess.run([sys.executable, str(script), 'prepare', '--task', str(root/'task.json'),
                                  '--experts', str(root/'experts.jsonl'), '--output-dir', str(root/'prepared')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            candidate = json.loads((root/'prepared/candidates.jsonl').read_text())
            ratings = [self.rating(candidate), self.rating(candidate, 'rater-b')]
            (root/'ratings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ratings))
            run = subprocess.run([sys.executable, str(script), 'build', '--task', str(root/'task.json'),
                                  '--candidates', str(root/'prepared/candidates.jsonl'), '--ratings', str(root/'ratings.jsonl'),
                                  '--output-dir', str(root/'built')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            record = json.loads((root/'built/benchmark.jsonl').read_text())
            self.assertTrue(record['human_decision'])
            self.assertEqual(len(record['annotation']['ratings']), 2)

    def test_binary_monitoring_and_score_consistency(self):
        from quality import evaluate_stream
        from pathlib import Path
        import tempfile
        record = {'id': 'x', 'context': {}, 'output': 'x', 'metadata': {},
                  'human_decision': False, 'human_rationale': 'Unsupported'}
        class Client:
            def complete(self, system, payload):
                return {'scores': {'accuracy': 2}, 'rationale': 'Accepted'}
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_stream([record], TaskJudge(Client(), self.task()), Path(tmp))
            self.assertEqual(report['human_alignment']['labeled_count'], 1)
            self.assertIsNone(report['human_alignment']['macro_agreement'])
            self.assertTrue(report['alerts'])
        record['human'] = {'accuracy': 2}
        with self.assertRaises(ValueError):
            validate_record(record, self.task())

    def test_model_cannot_submit_labels_in_augmentation(self):
        from benchmark_authoring import augment
        class Client:
            model = 'model'
            def complete(self, system, payload):
                return {'candidates': [{'output': 'Maybe a fact', 'boundary_reason': 'Ambiguous', 'passed': True}]}
        with self.assertRaises(ValueError):
            augment(self.candidates(), self.task(), Client(), 1)

    def test_related_alignment_families_are_rejected(self):
        class Client:
            def complete(self, *args):
                raise AssertionError('No model call should occur on leaking splits')
        a = {'id': 'a', 'context': {}, 'output': 'a', 'metadata': {'benchmark_group': 'family'},
             'human_decision': True, 'human_rationale': 'Supported'}
        b = {**a, 'id': 'b', 'output': 'b'}
        with self.assertRaisesRegex(ValueError, 'families'):
            align(TaskJudge(Client(), self.task()), [a], [b])
