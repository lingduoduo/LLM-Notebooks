"""CLI adapter for the offline ontology workflow."""
import csv
import json
from pathlib import Path
import sys

import utils
import workflow
from workflow_report import render_report

EXAMPLES = utils.ROOT / 'workflow_examples'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_graph(path):
    value = read_json(path)
    if isinstance(value, dict) and 'graph' in value:
        value = value['graph']
    workflow.validate_graph(value)
    return value


def protect_outputs(outputs, inputs):
    """Reject source aliases (including symlinks and existing hard links)."""
    protected = [Path(p) for p in inputs] + [utils.ROOT / name for name in
                 ('taxonomy_v1.json', 'taxonomy_schema.json', 'keywords.csv')]
    for output in map(Path, outputs):
        for source in protected:
            if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
                raise ValueError(f'output would overwrite an input: {output}')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def classifier():
    return utils.RoomClassifier(utils.load_taxonomy(), utils.load()[1])


def demo(out):
    taxonomy = utils.load_taxonomy()
    graph = workflow.build_graph(taxonomy)
    schema = {kind: {'prefix': spec['prefix'], 'required_properties': {
        key: (' or '.join(t.__name__ for t in expected) if isinstance(expected, tuple) else expected.__name__)
        for key, expected in spec['required'].items()}} for kind, spec in workflow.SCHEMA.items()}
    stages = [{'title': 'Build ontology',
               'description': 'Validate the versioned taxonomy and define entity properties and relationship endpoints.',
               'stats': {'concepts': len(graph['nodes']), 'taxonomy version': taxonomy['version']},
               'data': {'entities': schema, 'relationships': workflow.RELATIONS},
               'tables': [
                   {'caption': 'Entity structure', 'headers': ['Entity', 'ID prefix', 'Required properties'],
                    'rows': [[kind, spec['prefix'], ', '.join(spec['required_properties'])] for kind, spec in schema.items()]},
                   {'caption': 'Relationship structure', 'headers': ['Relationship', 'From', 'To'],
                    'rows': [[name, *endpoints] for name, endpoints in workflow.RELATIONS.items()]},
                   {'caption': 'Category roots and example branch', 'headers': ['Concept', 'Label', 'Parent', 'Axis'],
                    'rows': [[n['id'], n['label'], n['parent_id'] or 'Root', n['axis']] for n in taxonomy['nodes'] if n['level'] <= 1 or n['id'] in ('material_interest', 'latex')]},
               ]}]
    mapped = workflow.map_rooms(graph, EXAMPLES / 'rooms.csv', read_json(EXAMPLES / 'mapping.json'))
    rooms = [n for n in mapped['nodes'] if n['type'] == 'Room']
    stages.append({'title': 'Map data', 'description': 'Map explicit CSV columns into validated room records.',
                   'stats': {'rooms': len(rooms)},
                   'data': {'mapping': read_json(EXAMPLES / 'mapping.json'), 'rooms': rooms},
                   'tables': [{'caption': 'Mapped room records', 'headers': ['Room', 'Subject', 'Tags', 'Self-declared concepts'],
                               'rows': [[n['properties']['room_id'], n['properties']['subject'],
                                         ', '.join(n['properties']['tags']) or 'None',
                                         ', '.join(n['properties']['self_declared_concepts']) or 'None'] for n in rooms]}]})
    queries = {
        'search_latex': workflow.query_graph(mapped, 'search', text='latex'),
        'latex_neighbors': workflow.query_graph(mapped, 'neighbors', node='concept:latex'),
        'latex_two_hops': workflow.query_graph(mapped, 'bfs', node='concept:latex', depth=2),
        'latex_to_content': workflow.query_graph(mapped, 'path', node='concept:latex', target='concept:adult_content'),
    }
    stages.append({'title': 'Query graph', 'description': 'Search properties, inspect neighbors, traverse two hops and find a shortest path.',
                   'stats': {'queries': len(queries)}, 'data': queries,
                   'tables': [{'caption': 'Query results', 'headers': ['Query', 'Result'], 'rows': [
                       ['Search: latex', ', '.join(n['id'] for n in queries['search_latex']) or 'No matches'],
                       ['Neighbors: concept:latex', ', '.join(n['id'] for n in queries['latex_neighbors']) or 'No neighbors'],
                       ['Two-hop traversal: concept:latex', '\n'.join(' → '.join(n['path']) for n in queries['latex_two_hops'])],
                       ['Shortest path: latex to adult_content', '\n'.join(' → '.join(i.removeprefix('concept:') for i in n['path'])
                                                                                     for n in queries['latex_to_content']) or 'No path'],
                   ]}]})
    model = classifier()
    inferred, trace = workflow.infer_graph(mapped, model)
    derived = [e for e in inferred['edges'] if e['type'] in ('ASSIGNED_TO', 'INFERRED_AS')]
    stages.append({'title': 'Infer knowledge',
                   'description': 'Reuse classifier evidence and taxonomy ancestry. Ambiguous terms retain a review flag.',
                   'stats': {'direct assignments': sum(e['type'] == 'ASSIGNED_TO' for e in derived),
                             'ancestor assignments': sum(e['type'] == 'INFERRED_AS' for e in derived)},
                   'data': {'trace': trace, 'derived_relationships': derived},
                   'tables': [
                       {'caption': 'Classification results', 'headers': ['Room', 'Direct concepts', 'Inferred ancestors', 'Review'],
                        'rows': [[t['result']['room_id'], ', '.join(a['concept_id'] for a in t['result']['assignments']) or 'No matches',
                                  ', '.join(t['result']['inferred_ancestors']) or 'None',
                                  'requires context review' if t['result']['requires_context_review'] else 'No review flag'] for t in trace]},
                       {'caption': 'Direct evidence', 'headers': ['Room', 'Concept', 'Field', 'Matched term', 'Original text'],
                        'rows': [[t['result']['room_id'], a['concept_id'], e['field'], e['matched_term'] or 'Explicit declaration', e['text']]
                                 for t in trace for a in t['result']['assignments'] for e in a['evidence']]},
                       {'caption': 'Ancestor inference paths', 'headers': ['Room', 'Inferred concept', 'Supporting path'],
                        'rows': [[e['source'], e['target'], ' → '.join(path)] for e in derived if e['type'] == 'INFERRED_AS'
                                 for path in e['properties']['supporting_paths']]},
                   ]})
    definitions = read_json(EXAMPLES / 'behaviors.json')
    workflow.validate_behaviors(definitions)
    stages.append({'title': 'Define behavior',
                   'description': 'Register allowlisted actions, read-only functions and an ambiguity event.',
                   'stats': {k: len(definitions[k]) for k in ('actions', 'functions', 'events')}, 'data': definitions,
                   'tables': [{'caption': 'Registered behaviors', 'headers': ['Kind', 'Name', 'Parameters', 'Operations or condition'],
                               'rows': [
                                   *[['Action', name, ', '.join(entry['parameters']) or 'None',
                                      ' → '.join(step['operation'] for step in entry['steps'])] for name, entry in definitions['actions'].items()],
                                   *[['Function', name, ', '.join(entry['parameters']) or 'None', entry['operation']]
                                     for name, entry in definitions['functions'].items()],
                                   *[['Event', name, ', '.join(entry['parameters']) or 'None', entry['condition'] + ' → ' + entry['action']]
                                     for name, entry in definitions['events'].items()],
                               ]}]})
    action = workflow.run_sandbox(inferred, definitions, 'action', 'classify', {'room_id': 'clear'}, model)
    sandbox = workflow.run_sandbox(inferred, definitions, 'event', 'ambiguous', {'room_id': 'ambiguous'}, model)
    pending = workflow.run_sandbox(sandbox['graph'], definitions, 'function', 'pending', {}, model)
    stages.append({'title': 'Sandbox',
                   'description': 'Classify on a copy, trigger the ambiguity event and inspect its pending review task. Inputs stay unchanged.',
                   'stats': {'review tasks created': len(sandbox['changes']['nodes']['added']),
                             'pending reviews': len(pending['result'])},
                   'data': {'action_trace': action['trace'], 'event_trace': sandbox['trace'],
                            'function_trace': pending['trace'], 'changes': sandbox['changes']},
                   'tables': [
                       {'caption': 'Execution outcomes', 'headers': ['Behavior', 'Outcome'], 'rows': [
                           ['Action: classify(clear)', 'Reclassified ' + action['result']['room_id'] + ' on an isolated graph copy'],
                           ['Event: ambiguous(ambiguous)', 'requires_context_review = ' + str(sandbox['trace'][0]['matched']).lower()
                            + ('; dispatched review' if sandbox['trace'][0]['matched'] else '; no action dispatched')],
                           ['Function: pending()', str(len(pending['result'])) + ' pending review task'],
                       ]},
                       {'caption': 'Sandbox graph changes', 'headers': ['Change', 'Entity or relationship', 'Result'], 'rows': [
                           *[['Node added', n['id'], n['properties']['status']] for n in sandbox['changes']['nodes']['added']],
                           *[['Relationship added', e['type'], e['source'] + ' → ' + e['target']] for e in sandbox['changes']['edges']['added']],
                       ]},
                   ]})
    out = Path(out)
    outputs = [out / name for name in ('graph.json', 'sandbox.json', 'report.html')]
    protect_outputs(outputs, [EXAMPLES / name for name in ('rooms.csv', 'mapping.json', 'behaviors.json')])
    write_json(outputs[0], inferred)
    write_json(outputs[1], sandbox)
    render_report(stages, outputs[2])
    return json.dumps({'stages': len(stages), 'outputs': [str(p.resolve()) for p in outputs]}, indent=2)


def run(args):
    try:
        if args.workflow_command == 'dashboard':
            from workflow_dashboard import serve
            return serve(args.port)
        if args.workflow_command == 'demo':
            return demo(args.out)
        if args.workflow_command == 'map':
            protect_outputs([args.output], [args.csv, args.mapping])
            graph = workflow.map_rooms(workflow.build_graph(utils.load_taxonomy()), args.csv, read_json(args.mapping))
            write_json(args.output, graph)
            return str(args.output)
        graph = read_graph(args.graph)
        if args.workflow_command == 'query':
            result = workflow.query_graph(graph, args.operation, text=args.text, node=args.node,
                                          target=args.target, depth=args.depth, relation=args.relation)
            return json.dumps(result, ensure_ascii=False, indent=2)
        protect_outputs([args.output], [args.graph, args.behaviors])
        result = workflow.run_sandbox(graph, read_json(args.behaviors), args.kind, args.name,
                                      json.loads(args.params), classifier())
        write_json(args.output, result)
        return str(args.output)
    except (ValueError, OSError, csv.Error) as error:
        print(f'workflow error: {error}', file=sys.stderr)
        return 1


def add_parser(subparsers):
    parser = subparsers.add_parser('workflow', help='run the six-stage offline ontology workflow')
    commands = parser.add_subparsers(dest='workflow_command', required=True)
    dashboard = commands.add_parser('dashboard', help='open a local interactive graph workspace')
    dashboard.add_argument('--port', type=int, default=8765)
    d = commands.add_parser('demo', help='run all six stages and write an HTML report')
    d.add_argument('--out', type=Path, default=utils.ROOT / 'out' / 'workflow')
    m = commands.add_parser('map', help='map CSV records to an ontology graph')
    m.add_argument('--csv', type=Path, required=True)
    m.add_argument('--mapping', type=Path, default=EXAMPLES / 'mapping.json')
    m.add_argument('--output', type=Path, required=True)
    q = commands.add_parser('query', help='search and traverse a saved graph')
    q.add_argument('--graph', type=Path, required=True)
    q.add_argument('--operation', choices=('search', 'neighbors', 'bfs', 'path'), required=True)
    q.add_argument('--text', default='')
    q.add_argument('--node')
    q.add_argument('--target')
    q.add_argument('--depth', type=int, default=2)
    q.add_argument('--relation', choices=tuple(workflow.RELATIONS))
    s = commands.add_parser('sandbox', help='execute behavior on a copy of a saved graph')
    s.add_argument('--graph', type=Path, required=True)
    s.add_argument('--behaviors', type=Path, default=EXAMPLES / 'behaviors.json')
    s.add_argument('--kind', choices=('action', 'function', 'event'), required=True)
    s.add_argument('--name', required=True)
    s.add_argument('--params', default='{}', help='JSON object of behavior arguments')
    s.add_argument('--output', type=Path, required=True)
    for command in (dashboard, d, m, q, s):
        command.set_defaults(func=run)
