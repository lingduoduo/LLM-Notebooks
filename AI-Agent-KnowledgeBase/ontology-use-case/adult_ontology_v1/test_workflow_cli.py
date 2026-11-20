"""Run actual CLI subprocesses and inspect report/output contracts."""
import csv
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import utils

ROOT = Path(__file__).resolve().parent


class WorkflowCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)

    def cli(self, *args, good=True):
        result = subprocess.run([sys.executable, str(ROOT / 'main.py'), 'workflow', *map(str, args)],
                                cwd=self.out, capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, good, result.stdout + result.stderr)
        if not good:
            self.assertNotIn('Traceback', result.stderr)
        return result

    def demo(self):
        self.cli('demo', '--out', self.out)
        return json.loads((self.out / 'graph.json').read_text())

    def test_demo_runs_all_stages_and_protects_inputs(self):
        paths = [ROOT / 'taxonomy_v1.json', ROOT / 'keywords.csv', ROOT / 'workflow_examples' / 'rooms.csv']
        before = [p.read_bytes() for p in paths]
        graph = self.demo()
        self.assertEqual(len([n for n in graph['nodes'] if n['type'] == 'Room']), 4)
        result = json.loads((self.out / 'sandbox.json').read_text())
        self.assertEqual(len(result['changes']['nodes']['added']), 1)
        self.assertEqual(result['changes']['nodes']['added'][0]['type'], 'ReviewTask')
        self.assertEqual(before, [p.read_bytes() for p in paths])
        report = (self.out / 'report.html').read_text()
        for stage in ('Build ontology', 'Map data', 'Query graph', 'Infer knowledge', 'Define behavior', 'Sandbox'):
            self.assertIn(stage, report)
        self.assertIn('supporting_paths', report)
        self.assertIn('matched_term', report)

    def test_map_custom_columns_and_query(self):
        source = self.out / 'input.csv'
        with source.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(['id', 'text', 'labels'])
            writer.writerow(['mine', 'こんにちは, chat', '["chat"]'])
        mapping = self.out / 'mapping.json'
        mapping.write_text(json.dumps({'room_id':'id', 'subject':'text', 'tags':'labels'}))
        graph_path = self.out / 'mapped.json'
        self.cli('map', '--csv', source, '--mapping', mapping, '--output', graph_path)
        searched = self.cli('query', '--graph', graph_path, '--operation', 'search', '--text', 'こんにちは')
        self.assertEqual(json.loads(searched.stdout)[0]['id'], 'room:mine')
        self.cli('query', '--graph', graph_path, '--operation', 'neighbors', '--node', 'unknown', good=False)

    def test_queries_paths_and_filters(self):
        self.demo()
        graph = self.out / 'graph.json'
        path = self.cli('query', '--graph', graph, '--operation', 'path',
                        '--node', 'concept:latex', '--target', 'concept:adult_content', '--relation', 'IS_A')
        self.assertEqual(json.loads(path.stdout)[0]['path'],
                         ['concept:latex', 'concept:material_interest', 'concept:fetish_kink', 'concept:adult_content'])
        self.cli('query', '--graph', graph, '--operation', 'bfs', '--node', 'room:clear', '--depth', '-1', good=False)

    def test_sandbox_event_and_query_result_envelope(self):
        self.demo()
        destination = self.out / 'event.json'
        self.cli('sandbox', '--graph', self.out / 'graph.json', '--kind', 'event', '--name', 'ambiguous',
                 '--params', '{"room_id":"ambiguous"}', '--output', destination)
        value = json.loads(destination.read_text())
        self.assertTrue(value['trace'][0]['matched'])
        searched = self.cli('query', '--graph', destination, '--operation', 'search', '--text', 'ReviewTask')
        self.assertEqual(len(json.loads(searched.stdout)), 1)
        self.cli('sandbox', '--graph', destination, '--kind', 'function', '--name', 'pending', '--params', '{}',
                 '--output', self.out / 'pending.json')
        self.assertEqual(len(json.loads((self.out / 'pending.json').read_text())['result']), 1)

    def test_invalid_json_csv_and_behavior_exit_nonzero(self):
        self.demo()
        bad = self.out / 'bad.json'; bad.write_text('{')
        self.cli('query', '--graph', bad, '--operation', 'search', good=False)
        bad.write_text('[]')
        self.cli('query', '--graph', bad, '--operation', 'search', good=False)
        source = self.out / 'bad.csv'; source.write_text('room_id,subject,tags\na,chat,no-json\n')
        mapping = self.out / 'mapping.json'; mapping.write_text('{"room_id":"room_id","subject":"subject","tags":"tags"}')
        self.cli('map', '--csv', source, '--mapping', mapping, '--output', self.out / 'badout.json', good=False)
        self.assertFalse((self.out / 'badout.json').exists())
        self.cli('sandbox', '--graph', self.out / 'graph.json', '--behaviors', bad, '--kind', 'action',
                 '--name', 'classify', '--params', '{"room_id":"clear"}', '--output', self.out / 'badout.json', good=False)

    def test_refuses_to_overwrite_inputs(self):
        self.demo()
        graph = self.out / 'graph.json'; before = graph.read_bytes()
        self.cli('sandbox', '--graph', graph, '--kind', 'action', '--name', 'classify',
                 '--params', '{"room_id":"clear"}', '--output', graph, good=False)
        self.assertEqual(graph.read_bytes(), before)

    def test_report_escapes_untrusted_text(self):
        from workflow_report import render_report
        render_report([{'title': '<script>alert(1)</script>', 'data': {'text': '<img src=x onerror=alert(2)>'}}],
                       self.out / 'escaped.html')
        content = (self.out / 'escaped.html').read_text()
        self.assertNotIn('<script>', content)
        self.assertNotIn('<img src=x', content)
        self.assertIn('&lt;script&gt;', content)
        self.assertIn('&lt;img', content)


    def test_results_are_visible_without_expanding_raw_json(self):
        self.demo()

        class VisibleResults(HTMLParser):
            def __init__(self):
                super().__init__()
                self.hidden = 0
                self.section = None
                self.results = {}

            def handle_starttag(self, tag, attrs):
                if tag == 'section':
                    self.section = dict(attrs)['id']
                    self.results[self.section] = []
                if tag in ('details', 'style'):
                    self.hidden += 1

            def handle_endtag(self, tag):
                if tag in ('details', 'style'):
                    self.hidden -= 1
                if tag == 'section':
                    self.section = None

            def handle_data(self, data):
                if self.section and not self.hidden:
                    self.results[self.section].append(data)

        parser = VisibleResults()
        parser.feed((self.out / 'report.html').read_text())
        expected = {
            'stage-1': ['Concept', 'ASSIGNED_TO', 'Material Interest'],
            'stage-2': ['Private show with cosplay and latex', 'Coffee, music and conversation'],
            'stage-3': ['latex → material_interest → fetish_kink → adult_content'],
            'stage-4': ['costume_roleplay', 'requires context review', 'sexy', 'orientation'],
            'stage-5': ['classify_room', 'create_review_task', 'requires_context_review'],
            'stage-6': ['review:ambiguous:context', 'pending', 'HAS_REVIEW'],
        }
        for section, values in expected.items():
            for value in values:
                with self.subTest(section=section, value=value):
                    self.assertIn(value, ' '.join(parser.results[section]))
