import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from benchmark import load_jsonl

SCRIPT = Path(__file__).with_name('demo.py').resolve()


class CLITests(unittest.TestCase):
    def test_run_from_any_directory_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(SCRIPT), '--output-dir', tmp],
                                 cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            path = Path(tmp) / 'report.json'
            self.assertTrue(path.exists(), 'CLI must persist an evaluation report')
            report = json.loads(path.read_text())
            self.assertTrue(report['monitoring']['drifted'])
            self.assertTrue(report['generation']['passed'])
            self.assertFalse(set(report['calibration_ids']) & set(report['validation_ids']))
            self.assertIn('candidate_promoted', report)

    def test_invalid_score_is_rejected_with_line_number(self):
        source = json.loads(SCRIPT.with_name('data').joinpath('benchmark.jsonl').read_text().splitlines()[0])
        source['human']['clarity'] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.jsonl'
            path.write_text(json.dumps(source) + '\n')
            with self.assertRaisesRegex(ValueError, ':1:'):
                load_jsonl(path)

    def test_drift_exit_code_and_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(SCRIPT), '--output-dir', tmp, '--fail-on-drift'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            run = subprocess.run([sys.executable, str(SCRIPT), '--output-dir', tmp,
                                  '--validation', str(SCRIPT.parent / 'data/benchmark.jsonl')],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn('distinct IDs', run.stderr)

    def test_real_backend_workflow_over_http(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from dataclasses import asdict
        from judge import DemoJudge, DEFAULT_RUBRIC
        from models import BenchmarkExample, UserContext, Item, HumanLabel
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                system = body['messages'][0]['content']
                payload = json.loads(body['messages'][1]['content'])
                requests.append((system, payload))
                if system.startswith('Evaluate'):
                    ex = BenchmarkExample('test', UserContext(**payload['user']), Item(**payload['item']),
                                          payload['explanation'], HumanLabel(0, 0, 0, 0, 'unused'))
                    result = DemoJudge(decouple_relevance='INDEPENDENT' in system).evaluate(ex)
                    answer = asdict(result)
                    answer.pop('passed')
                elif system.startswith('Improve'):
                    answer = dict(DEFAULT_RUBRIC)
                    answer['relevance'] += ' INDEPENDENT'
                else:
                    answer = {'text': ('You follow this creator and frequently watch Lakers games.'
                                       if payload['attempt'] == 0 else 'Your basketball interests match this NBA stream.')}
                response = json.dumps({'choices': [{'finish_reason': 'stop',
                                                     'message': {'content': json.dumps(answer)}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run = subprocess.run([sys.executable, str(SCRIPT), '--backend', 'llm', '--model', 'test-model',
                                      '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                                      '--output-dir', tmp], capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads((Path(tmp) / 'report.json').read_text())
                self.assertTrue(report['candidate_promoted'])
                self.assertTrue(report['generation']['passed'])
                self.assertEqual(len(report['generation']['traces']), 2)
                self.assertTrue(report['monitoring']['drifted'])
                self.assertTrue(any(s.startswith('Improve') for s, _ in requests))
                for system, payload in requests:
                    if system.startswith('Evaluate'):
                        self.assertNotIn('human', payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
