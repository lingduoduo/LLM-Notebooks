import json
from pathlib import Path
import tempfile
import unittest


class QualityTests(unittest.TestCase):
    def task(self):
        from quality import TaskSpec
        return TaskSpec('summary', 'Check source support.',
                        {'accuracy': '0=wrong, 1=partial, 2=supported'}, {'accuracy': 2})

    def record(self, i=0, human=None):
        record = {'id': str(i), 'context': {'source': 'A fact.'}, 'output': 'A fact.',
                  'metadata': {'catalog_version': '2026-09', 'generator_model': 'generator'}}
        if human is not None:
            record['human'] = human
        return record

    def test_arbitrary_task_and_no_label_leakage(self):
        from quality import TaskJudge
        class Client:
            model, base_url = 'judge', 'http://localhost/v1'
            def complete(self, system, payload):
                self.payload = payload
                return {'scores': {'accuracy': 2}, 'rationale': 'Supported'}
        client = Client()
        result = TaskJudge(client, self.task()).evaluate(self.record(human={'accuracy': 0}))
        self.assertTrue(result['passed'])
        self.assertEqual(set(client.payload), {'context', 'output'})

    def test_invalid_model_scores_fail(self):
        from quality import TaskJudge
        class Client:
            def complete(self, system, payload):
                return {'scores': {'accuracy': True}, 'rationale': 'Bad'}
        with self.assertRaises(ValueError):
            TaskJudge(Client(), self.task()).evaluate(self.record())

    def test_streaming_failures_and_bounded_review_queue(self):
        from quality import TaskJudge, evaluate_stream
        from llm_backend import BackendError
        class Client:
            model, base_url = 'judge', 'http://localhost/v1'
            def complete(self, system, payload):
                if payload['output'] == 'error':
                    raise BackendError('timeout')
                return {'scores': {'accuracy': 2}, 'rationale': 'Supported'}
        def records():
            for i in range(20):
                record = self.record(i, {'accuracy': 0} if i == 1 else None)
                if i == 0:
                    record['output'] = 'error'
                yield record
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_stream(records(), TaskJudge(Client(), self.task()), Path(tmp), workers=2, review_size=3)
            self.assertEqual(report['processed'], 20)
            self.assertEqual(report['errors'], 1)
            self.assertEqual(report['human_alignment']['false_accept_rate'], 1)
            self.assertEqual(len((Path(tmp)/'review_queue.jsonl').read_text().splitlines()), 3)
            rows = [json.loads(line) for line in (Path(tmp)/'evaluations.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['status'], 'error')
            self.assertFalse(rows[0]['passed'])
            self.assertEqual(rows[1]['record']['metadata']['catalog_version'], '2026-09')
            for line in (Path(tmp)/'review_queue.jsonl').read_text().splitlines():
                self.assertNotIn('result', json.loads(line))

    def test_no_labels_does_not_claim_alignment(self):
        from quality import Alignment
        stats = Alignment(self.task())
        self.assertIsNone(stats.report()['macro_agreement'])
        self.assertEqual(stats.report()['labeled_count'], 0)

    def test_invalid_task_threshold(self):
        from quality import TaskSpec
        with self.assertRaises(ValueError):
            TaskSpec('bad', 'judge', {'accuracy': 'rubric'}, {'accuracy': 3})

    def test_calibration_excludes_validation_from_reflection(self):
        from quality import TaskJudge, align
        class Client:
            model, base_url = 'judge', 'local'
            def complete(self, system, payload):
                if 'rubric' in payload:
                    self.feedback = payload['disagreements']
                    return {'accuracy': 'FIXED 0=wrong, 1=partial, 2=supported'}
                return {'scores': {'accuracy': 2 if 'FIXED' in system else 0}, 'rationale': 'Evidence'}
        client = Client()
        calibration = [self.record('calibration', {'accuracy': 2})]
        validation = [self.record('validation', {'accuracy': 2})]
        validation[0]['output'] = 'A separate fact.'
        judge, report = align(TaskJudge(client, self.task()), calibration, validation)
        self.assertTrue(report['promoted'])
        self.assertEqual([r['id'] for r in client.feedback], ['calibration'])
        with self.assertRaises(ValueError):
            align(judge, calibration, calibration)

    def test_batch_cli_over_http(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        import subprocess
        import sys
        from dataclasses import asdict
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                content = json.dumps({'scores': {'accuracy': 2}, 'rationale': 'Supported'})
                response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}).encode()
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
                (root/'input.jsonl').write_text(json.dumps(self.record()) + '\n')
                command = [sys.executable, str(Path(__file__).with_name('batch.py')), '--task', str(root/'task.json'),
                           '--input', str(root/'input.jsonl'), '--output-dir', str(root/'run'),
                           '--judge-model', 'evaluator', '--base-url', f'http://127.0.0.1:{server.server_port}/v1']
                run = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((root/'run/report.json').read_text())
                self.assertEqual(report['processed'], 1)
                self.assertEqual(report['alignment_status'], 'unavailable_no_scored_human_labels')
                second = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertEqual(second.returncode, 1)
                self.assertIn('new or empty', second.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_stream_does_not_eagerly_read_all_records(self):
        from quality import evaluate_stream, TaskJudge
        class Client:
            calls = 0
            def complete(self, system, payload):
                self.calls += 1
                return {'scores': {'accuracy': 2}, 'rationale': 'Supported'}
        client = Client()
        def records():
            for i in range(50):
                if i >= 4:
                    self.assertGreater(client.calls, 0)
                yield self.record(i)
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_stream(records(), TaskJudge(client, self.task()), Path(tmp), workers=2)
            self.assertEqual(report['processed'], 50)
