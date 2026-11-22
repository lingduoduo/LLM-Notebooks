"""Local ontology workspace backed by the existing Python workflow."""
import json
from threading import RLock

import utils
import workflow
from workflow import clone

EXAMPLES = utils.ROOT / 'workflow_examples'


class DashboardSession:
    """A private, transactional graph copy for one local dashboard process.

    Stored values are replaced, never mutated in place, so they can be shared
    internally; only values handed to callers are cloned.
    """

    def __init__(self):
        self.lock = RLock()
        self.taxonomy = utils.load_taxonomy()
        self.keywords = utils.load()[1]
        self.classifier = utils.RoomClassifier(self.taxonomy, self.keywords)
        self.mapping = json.loads((EXAMPLES / 'mapping.json').read_text(encoding='utf-8'))
        self.behaviors = json.loads((EXAMPLES / 'behaviors.json').read_text(encoding='utf-8'))
        workflow.validate_behaviors(self.behaviors)
        self.rules = json.loads((EXAMPLES / 'rules.json').read_text(encoding='utf-8'))
        workflow.validate_rules(self.rules)
        self.sample_csv = (EXAMPLES / 'rooms.csv').read_text(encoding='utf-8-sig')
        self.sample_mapping = clone(self.mapping)
        self.csv_text = self.sample_csv
        self.mapped = workflow.map_rooms(workflow.build_graph(self.taxonomy), EXAMPLES / 'rooms.csv', self.mapping)
        self.initial, trace = workflow.infer_graph(self.mapped, self.classifier)
        self.classifications = {entry['result']['room_id']: entry['result'] for entry in trace}
        self.graph = self.initial
        self.last_execution = None

    def state(self) -> dict:
        with self.lock:
            return clone(self._state_view())

    def state_json(self) -> str:
        """Serialize directly under the lock; no intermediate copy."""
        with self.lock:
            return json.dumps(self._state_view(), ensure_ascii=False)

    def _state_view(self) -> dict:
        """References to current state; call with the lock held."""
        return {'taxonomy': self.taxonomy, 'keywords': self.keywords,
                'mapping': self.mapping, 'mapped': self.mapped,
                'graph': self.graph, 'initial_graph': self.initial,
                'classifications': self.classifications,
                'behaviors': self.behaviors, 'last_execution': self.last_execution,
                'rules': [{'name': name, 'label': rule['label'], 'text': workflow.describe_rule(rule)}
                          for name, rule in self.rules['rules'].items()],
                'contradiction_pairs': self.rules['contradictions'],
                'csv_text': self.csv_text, 'sample_csv': self.sample_csv,
                'sample_mapping': self.sample_mapping,
                'schema': {'types': {kind: sorted(spec['required']) for kind, spec in workflow.SCHEMA.items()},
                           'relations': {name: list(ends) for name, ends in workflow.RELATIONS.items()}}}

    def infer(self) -> dict:
        """Evaluate the rule registry against the current sandbox graph (read-only)."""
        with self.lock:
            return workflow.run_rules(self.graph, self.rules)

    def map_preview(self, csv_text: str, mapping: dict, apply: bool = False) -> dict:
        """Map edited CSV text into a fresh graph; optionally make it the workspace graph."""
        if not isinstance(csv_text, str) or len(csv_text) > 50000:
            raise ValueError('CSV must be text of at most 50000 characters')
        if not isinstance(apply, bool):
            raise ValueError('apply must be true or false')
        mapped = workflow.map_rooms_text(workflow.build_graph(self.taxonomy), csv_text, mapping)
        inferred, trace = workflow.infer_graph(mapped, self.classifier)
        classifications = {entry['result']['room_id']: entry['result'] for entry in trace}
        alerts = []
        for room_id, result in classifications.items():
            if not result['subject'].strip():
                alerts.append({'room_id': room_id, 'message': 'Empty subject; only tags were classified.'})
            if not result['assignments']:
                alerts.append({'room_id': room_id, 'message': 'No concept matched this room.'})
        types = {node['type'] for node in inferred['nodes']}
        preview = {'graph': inferred, 'classifications': classifications, 'alerts': alerts,
                   'headers': workflow.csv_headers(csv_text),
                   'stats': {'nodes': len(inferred['nodes']), 'edges': len(inferred['edges']),
                             'types': len(types), 'rooms': len(classifications), 'alerts': len(alerts)}}
        if apply:
            with self.lock:
                self.mapping, self.csv_text, self.mapped = clone(mapping), csv_text, mapped
                self.initial, self.classifications = inferred, classifications
                self.graph, self.last_execution = inferred, None
        return clone(preview)

    def query(self, params: dict) -> list:
        if not isinstance(params, dict) or 'operation' not in params:
            raise ValueError('query requires an operation')
        operation = params['operation']
        if not isinstance(operation, str):
            raise ValueError('operation must be a string')
        allowed = {'operation', 'node', 'target', 'text', 'depth', 'relation'}
        if params.keys() - allowed:
            raise ValueError('unsupported query parameter')
        with self.lock:
            return workflow.query_graph(self.graph, operation, **{k: v for k, v in params.items() if k != 'operation'})

    def execute(self, kind: str, name: str, params: dict) -> dict:
        if not isinstance(kind, str):
            raise ValueError('kind must be a string')
        with self.lock:
            result = workflow.run_sandbox(self.graph, self.behaviors, kind, name, params, self.classifier)
            self.graph, self.last_execution = result['graph'], result
            return clone(result)

    def reset(self) -> dict:
        with self.lock:
            self.graph, self.last_execution = self.initial, None
            return self.state()


def make_server(port: int, session: DashboardSession):
    """Serve only the workspace and JSON APIs on IPv4 loopback."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, status, value, content_type='application/json; charset=utf-8'):
            data = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            if (self.headers.get('Host') not in hosts or
                    (origin is not None and origin not in {'http://' + host for host in hosts}) or
                    self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                self.respond(403, {'error': 'Only same-origin local requests are allowed'})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path in ('/', '/dashboard.html'):
                self.respond(200, render_dashboard(session), 'text/html; charset=utf-8')
            elif path == '/api/state':
                self.respond(200, session.state_json())
            else:
                self.respond(404, {'error': 'Unknown route'})

        def do_POST(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path not in ('/api/query', '/api/behavior', '/api/reset', '/api/infer', '/api/mapping'):
                self.respond(404, {'error': 'Unknown route'})
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                self.respond(415, {'error': 'Use application/json'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size > 65536:
                    self.respond(413, {'error': 'Request body is too large'})
                    return
                if size < 0:
                    raise ValueError('Invalid Content-Length')
                body = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(body, dict):
                    raise ValueError('Request must be a JSON object')
                if path == '/api/query':
                    result = session.query(body)
                elif path == '/api/behavior':
                    if set(body) != {'kind', 'name', 'params'}:
                        raise ValueError('Behavior needs kind, name and params')
                    result = session.execute(body['kind'], body['name'], body['params'])
                elif path == '/api/infer':
                    if body:
                        raise ValueError('Inference takes an empty object')
                    result = session.infer()
                elif path == '/api/mapping':
                    if not {'csv', 'mapping'} <= set(body) or set(body) - {'csv', 'mapping', 'apply'}:
                        raise ValueError('Mapping needs csv, mapping and optional apply')
                    result = session.map_preview(body['csv'], body['mapping'], body.get('apply', False))
                else:
                    if body:
                        raise ValueError('Reset takes an empty object')
                    result = session.reset()
                self.respond(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {'error': str(error)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def render_dashboard(session: DashboardSession) -> str:
    visuals = utils.ROOT / 'visuals'
    template = (visuals / 'workflow-dashboard.template.html').read_text(encoding='utf-8')
    payload = session.state_json().replace('<', '\\u003c')
    # Substitute data last: imported text cannot be mistaken for an asset placeholder.
    return (template.replace('__DASHBOARD_CSS__', (visuals / 'workflow-dashboard.css').read_text(encoding='utf-8'))
            .replace('__DASHBOARD_JS__', (visuals / 'workflow-dashboard.js').read_text(encoding='utf-8'))
            .replace('__DASHBOARD_DATA__', payload))


def serve(port=8765):
    server = make_server(port, DashboardSession())
    print(f'Ontology dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    print('Local sandbox only. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
