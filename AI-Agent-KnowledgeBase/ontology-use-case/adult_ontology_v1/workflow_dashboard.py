"""Local ontology workspace backed by the existing Python workflow."""
from copy import deepcopy
import json
from threading import RLock

import utils
import workflow

EXAMPLES = utils.ROOT / 'workflow_examples'


class DashboardSession:
    """A private, transactional graph copy for one local dashboard process."""

    def __init__(self):
        self.lock = RLock()
        self.taxonomy = utils.load_taxonomy()
        self.keywords = utils.load()[1]
        self.classifier = utils.RoomClassifier(self.taxonomy, self.keywords)
        self.mapping = json.loads((EXAMPLES / 'mapping.json').read_text(encoding='utf-8'))
        self.behaviors = json.loads((EXAMPLES / 'behaviors.json').read_text(encoding='utf-8'))
        workflow.validate_behaviors(self.behaviors)
        self.mapped = workflow.map_rooms(workflow.build_graph(self.taxonomy), EXAMPLES / 'rooms.csv', self.mapping)
        self.initial, trace = workflow.infer_graph(self.mapped, self.classifier)
        self.classifications = {entry['result']['room_id']: entry['result'] for entry in trace}
        self.graph = deepcopy(self.initial)
        self.last_execution = None

    def state(self) -> dict:
        with self.lock:
            return deepcopy({'taxonomy': self.taxonomy, 'keywords': self.keywords,
                             'mapping': self.mapping, 'mapped': self.mapped,
                             'graph': self.graph, 'initial_graph': self.initial,
                             'classifications': self.classifications,
                             'behaviors': self.behaviors, 'last_execution': self.last_execution})

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
            self.graph = deepcopy(result['graph'])
            self.last_execution = deepcopy(result)
            return result

    def reset(self) -> dict:
        with self.lock:
            self.graph = deepcopy(self.initial)
            self.last_execution = None
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
                self.respond(200, session.state())
            else:
                self.respond(404, {'error': 'Unknown route'})

        def do_POST(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path not in ('/api/query', '/api/behavior', '/api/reset'):
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
    payload = json.dumps(session.state(), ensure_ascii=False).replace('<', '\\u003c')
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
