import unittest
from quality import TaskSpec, TaskJudge


class TrainingTests(unittest.TestCase):
    def task(self):
        return TaskSpec('support', 'Check source support.', {'accuracy': 'initial rubric'}, {'accuracy': 2})

    def records(self, split):
        return [{'id': f'{split}-{i}', 'context': {'kind': kind}, 'output': f'{split} text {i}',
                 'metadata': {'benchmark_group': f'{split}-{i}'}, 'human_decision': passed,
                 'human_rationale': 'Supported by source.' if passed else 'Not supported by source.'}
                for i, (kind, passed) in enumerate([('good', True), ('bad-a', False), ('bad-b', False)])]

    def config(self, **overrides):
        from training import TrainingConfig
        return TrainingConfig(min_examples=3, min_per_class=1, target_agreement=1,
                              max_false_accept_rate=0, min_recall=1, **overrides)

    def client(self):
        class Client:
            model, base_url = 'test-judge', 'local'
            def __init__(self):
                self.reflections = []
                self.evaluations = []
            def complete(self, system, payload):
                if 'mismatches' in payload:
                    self.reflections.append(payload)
                    round_number = len(self.reflections)
                    return {'analysis': [{'id': row['id'], 'cause': 'Overaccepts unsupported outputs.',
                                          'change': 'Require source evidence.'} for row in payload['mismatches']],
                            'rubric': {'accuracy': f'revision-{round_number}'}}
                self.evaluations.append(payload)
                kind = payload['context']['kind']
                passed = kind == 'good' or (kind == 'bad-a' and 'revision-' not in system) or (
                    kind == 'bad-b' and 'revision-2' not in system)
                return {'scores': {'accuracy': 2 if passed else 0}, 'rationale': 'Based on evidence.'}
        return Client()

    def test_iterates_and_holdout_is_never_reflection_input(self):
        from training import train
        client = self.client()
        judge, report = train(TaskJudge(client, self.task()), self.records('cal'),
                              self.records('dev'), self.records('test'), self.config())
        self.assertTrue(report['aligned'])
        self.assertEqual(report['rounds_completed'], 2)
        self.assertIn('revision-2', judge.task.rubric['accuracy'])
        self.assertEqual(len([p for p in client.evaluations if p['output'].startswith('test')]), 3)
        for reflection in client.reflections:
            self.assertTrue(reflection['anchors'])
            self.assertTrue(all(r['id'].startswith('cal') for r in reflection['anchors'] + reflection['mismatches']))
            self.assertTrue(all('human_rationale' in r for r in reflection['mismatches']))
        self.assertTrue(all(set(p) == {'context', 'output'} for p in client.evaluations))

    def test_budget_exhaustion_is_not_success(self):
        from training import train
        _, report = train(TaskJudge(self.client(), self.task()), self.records('cal'),
                          self.records('dev'), self.records('test'), self.config(max_rounds=1))
        self.assertFalse(report['aligned'])
        self.assertEqual(report['stop_reason'], 'max_rounds')

    def test_unchanged_rubric_stops_with_patience(self):
        from training import train
        client = self.client()
        original = client.complete
        def complete(system, payload):
            result = original(system, payload)
            if 'mismatches' in payload:
                result['rubric'] = {'accuracy': 'initial rubric'}
            return result
        client.complete = complete
        _, report = train(TaskJudge(client, self.task()), self.records('cal'),
                          self.records('dev'), self.records('test'), self.config(patience=1))
        self.assertFalse(report['aligned'])
        self.assertEqual(report['stop_reason'], 'stagnation')

    def test_holdout_failure_prevents_alignment_claim(self):
        from training import train
        heldout = self.records('test')
        heldout[0]['human_decision'] = False
        heldout[1]['human_decision'] = True
        _, report = train(TaskJudge(self.client(), self.task()), self.records('cal'),
                          self.records('dev'), heldout, self.config())
        self.assertFalse(report['aligned'])
        self.assertEqual(report['stop_reason'], 'holdout_targets_not_met')

    def test_family_leakage_and_insufficient_labels_rejected(self):
        from training import train
        cal, dev, test = self.records('cal'), self.records('dev'), self.records('test')
        dev[0]['metadata']['benchmark_group'] = cal[0]['metadata']['benchmark_group']
        with self.assertRaises(ValueError):
            train(TaskJudge(self.client(), self.task()), cal, dev, test, self.config())
        dev = self.records('dev')
        with self.assertRaises(ValueError):
            train(TaskJudge(self.client(), self.task()), cal, dev[:1], test, self.config())

    def test_regression_and_invalid_reflection_do_not_replace_judge(self):
        from training import train
        client = self.client()
        original = client.complete
        def complete(system, payload):
            result = original(system, payload)
            if 'mismatches' not in payload and 'revision-' in system:
                result['scores'] = {'accuracy': 0}  # Rejects even known human-pass anchors.
            return result
        client.complete = complete
        judge, report = train(TaskJudge(client, self.task()), self.records('cal'),
                              self.records('dev'), self.records('test'), self.config(patience=1))
        self.assertEqual(judge.task.rubric['accuracy'], 'initial rubric')
        self.assertTrue(report['history'][1]['rejection_reasons'])
        self.assertFalse(report['aligned'])
        client = self.client()
        original = client.complete
        def invalid(system, payload):
            result = original(system, payload)
            if 'mismatches' in payload:
                result['rubric'] = {'unexpected': 'new criterion'}
            return result
        client.complete = invalid
        with self.assertRaises(ValueError):
            train(TaskJudge(client, self.task()), self.records('cal'), self.records('dev'), self.records('test'), self.config())

    def test_training_cli_over_http(self):
        import json
        from dataclasses import asdict
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        client = self.client()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                answer = client.complete(body['messages'][0]['content'], json.loads(body['messages'][1]['content']))
                response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(response)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'task.json').write_text(json.dumps(asdict(self.task())))
                command = [sys.executable, str(Path(__file__).with_name('train_judge.py')), '--task', str(root/'task.json'),
                           '--output-dir', str(root/'run'), '--judge-model', 'test-judge',
                           '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                           '--min-examples', '3', '--min-per-class', '1', '--target-agreement', '1']
                for split in ('calibration', 'development', 'holdout'):
                    path = root/f'{split}.jsonl'
                    path.write_text(''.join(json.dumps(r)+'\n' for r in self.records(split)))
                    command += ['--'+split, str(path)]
                run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((root/'run/report.json').read_text())
                self.assertTrue(report['aligned'])
                self.assertEqual(report['rounds_completed'], 2)
                self.assertTrue((root/'run/checkpoint_judge.json').exists())
                self.assertIn('revision-2', (root/'run/active_task.json').read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_birth_three_way_split_preserves_families(self):
        from benchmark_authoring import split_training_groups
        rows = [{'id': str(i), 'metadata': {'benchmark_group': str(i//2)}} for i in range(12)]
        splits = split_training_groups(rows, 0.25, 0.25, 7)
        self.assertTrue(all(splits))
        families = [{r['metadata']['benchmark_group'] for r in split} for split in splits]
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
        self.assertEqual(sum(map(len, splits)), 12)
        with self.assertRaises(ValueError):
            split_training_groups(rows[:4])

    def test_birth_cli_exports_training_splits(self):
        from benchmark_authoring import import_experts, candidate_hash
        from dataclasses import asdict
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        task = self.task()
        experts = [{**record, 'expert_id': 'fixture-expert', 'difficulty': 'boundary',
                    'boundary_reason': 'Fixture boundary', 'group_id': record['id']}
                   for record in self.records('birth')]
        candidates = import_experts(experts, task)
        ratings = [{'id': candidate['id'], 'candidate_hash': candidate_hash(candidate), 'rater_id': rater,
                    'passed': expert['human_decision'], 'rationale': expert['human_rationale']}
                   for candidate, expert in zip(candidates, experts) for rater in ('a', 'b')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'task.json').write_text(json.dumps(asdict(task)))
            for name, rows in [('candidates', candidates), ('ratings', ratings)]:
                (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            run = subprocess.run([sys.executable, str(Path(__file__).with_name('create_benchmark.py')), 'build',
                                  '--task', str(root/'task.json'), '--candidates', str(root/'candidates.jsonl'),
                                  '--ratings', str(root/'ratings.jsonl'), '--output-dir', str(root/'built'),
                                  '--validation-fraction', '0.25', '--holdout-fraction', '0.25'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for split in ('calibration', 'development', 'holdout'):
                self.assertTrue((root/f'built/{split}.jsonl').read_text().strip())
