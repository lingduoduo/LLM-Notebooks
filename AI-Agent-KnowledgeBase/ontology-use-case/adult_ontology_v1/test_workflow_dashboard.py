"""Dashboard data and API contracts use actual workflow implementations."""
import copy
import unittest

import workflow_dashboard as dashboard


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = dashboard.DashboardSession()

    def test_payload_has_actual_rooms_mapping_vocabulary_and_evidence(self):
        state = self.session.state()
        self.assertEqual(sum(n['type'] == 'Room' for n in state['graph']['nodes']), 4)
        self.assertEqual(len(state['taxonomy']['nodes']), 83)
        self.assertEqual(state['mapping']['subject'], 'title')
        self.assertTrue(any(r['keyword'] == 'latex' for r in state['keywords']))
        self.assertEqual(state['classifications']['ambiguous']['subject'], 'sexy latex')
        self.assertTrue(state['classifications']['ambiguous']['requires_context_review'])
        state['graph']['nodes'].clear()
        self.assertEqual(len(self.session.state()['graph']['nodes']), 87)

    def test_query_returns_real_paths(self):
        result = self.session.query({'operation': 'path', 'node': 'concept:latex', 'target': 'concept:adult_content', 'relation': 'IS_A'})
        self.assertEqual(result[0]['path'], ['concept:latex', 'concept:material_interest', 'concept:fetish_kink', 'concept:adult_content'])

    def test_behavior_is_stateful_idempotent_and_resettable(self):
        initial = self.session.state()['graph']
        first = self.session.execute('event', 'ambiguous', {'room_id': 'ambiguous'})
        self.assertEqual(len(first['changes']['nodes']['added']), 1)
        again = self.session.execute('event', 'ambiguous', {'room_id': 'ambiguous'})
        self.assertEqual(again['changes']['nodes']['added'], [])
        function = self.session.execute('function', 'pending', {})
        self.assertEqual(len(function['result']), 1)
        self.assertEqual(function['graph'], again['graph'])
        self.assertEqual(self.session.reset()['graph'], initial)

    def test_failure_preserves_state(self):
        before = copy.deepcopy(self.session.state())
        with self.assertRaises(ValueError):
            self.session.execute('action', 'classify', {'room_id': 'missing'})
        self.assertEqual(self.session.state(), before)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import threading
        cls.session = dashboard.DashboardSession()
        cls.server = dashboard.make_server(0, cls.session)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.session.reset()

    def request(self, path, body=None, headers=None):
        import json
        import urllib.error
        import urllib.request
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        request = urllib.request.Request(self.base + path, data=data,
                                         headers=headers or {'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def test_state_query_behavior_and_reset_routes(self):
        import json
        status, content = self.request('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(content)['graph']['nodes']), 87)
        status, content = self.request('/api/query', {'operation': 'search', 'text': 'latex'})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(content))
        status, content = self.request('/api/behavior', {'kind': 'event', 'name': 'ambiguous', 'params': {'room_id': 'ambiguous'}})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(content)['changes']['nodes']['added']), 1)
        status, content = self.request('/api/reset', {})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(content)['graph']['nodes']), 87)

    def test_invalid_requests_leave_state_intact(self):
        before = self.session.state()
        for path, body, headers, expected in (
            ('/api/query', b'{', None, 400),
            ('/api/query', [], None, 400),
            ('/api/query', {'operation': []}, None, 400),
            ('/api/behavior', {'kind': [], 'name': 'a', 'params': {}}, None, 400),
            ('/api/query', {}, {'Content-Type': 'text/plain'}, 415),
            ('/api/reset', {}, {'Content-Type': 'application/json', 'Origin': 'https://example.com'}, 403),
            ('/api/query', b' ' * 70000, None, 413),
            ('/not-a-route', {}, None, 404),
            ('/../utils.py', None, None, 404),
        ):
            with self.subTest(path=path, expected=expected):
                status, _ = self.request(path, body, headers)
                self.assertEqual(status, expected)
        self.assertEqual(self.session.state(), before)


class RenderingTests(unittest.TestCase):
    def test_dashboard_embeds_actual_payload_and_safe_text(self):
        import json
        from html.parser import HTMLParser
        session = dashboard.DashboardSession()
        session.taxonomy['nodes'][0]['label'] = '</script><img src=x onerror=alert(1)>'
        page = dashboard.render_dashboard(session)

        class Payload(HTMLParser):
            def __init__(self):
                super().__init__(); self.inside = False; self.data = ''; self.tabs = 0
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if attrs.get('id') == 'dashboard-data': self.inside = True
                if attrs.get('role') == 'tab': self.tabs += 1
            def handle_data(self, data):
                if self.inside: self.data += data
            def handle_endtag(self, tag):
                if tag == 'script': self.inside = False
        parser = Payload(); parser.feed(page)
        self.assertEqual(parser.tabs, 6)
        self.assertEqual(json.loads(parser.data)['classifications']['ambiguous']['subject'], 'sexy latex')
        self.assertEqual(json.loads(parser.data)['taxonomy']['nodes'][0]['label'], '</script><img src=x onerror=alert(1)>')
        self.assertNotIn('<img src=x', page)
        self.assertIn('<svg', page)

    def test_cli_dashboard_help(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, str(dashboard.utils.ROOT / 'main.py'), 'workflow', 'dashboard', '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--port', result.stdout)
