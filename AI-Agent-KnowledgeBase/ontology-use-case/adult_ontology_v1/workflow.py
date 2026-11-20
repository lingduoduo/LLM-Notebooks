"""Offline ontology graph, mapping, inference and allowlisted behavior execution."""
from collections import deque
from copy import deepcopy
import csv
import json
from pathlib import Path

import utils

SCHEMA = {
    'Concept': {'prefix': 'concept:', 'required': {'label': str, 'parent_id': (str, type(None))}},
    'Room': {'prefix': 'room:', 'required': {'room_id': str, 'subject': str, 'tags': list,
                                          'self_declared_concepts': list}},
    'ReviewTask': {'prefix': 'review:', 'required': {'room_id': str, 'status': str}},
}
RELATIONS = {
    'IS_A': ('Concept', 'Concept'),
    'ASSIGNED_TO': ('Room', 'Concept'),
    'INFERRED_AS': ('Room', 'Concept'),
    'HAS_REVIEW': ('Room', 'ReviewTask'),
}


def string_array(value, label):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError(f'{label} must be an array of strings')
    return value


def validate_graph(graph: dict) -> None:
    if not isinstance(graph, dict) or graph.get('schema_version') != 1:
        raise ValueError('unsupported graph schema_version')
    if not isinstance(graph.get('taxonomy_version'), str) or not graph['taxonomy_version']:
        raise ValueError('graph needs taxonomy_version')
    if not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list):
        raise ValueError('graph nodes and edges must be arrays')
    nodes = {}
    for node in graph['nodes']:
        if not isinstance(node, dict):
            raise ValueError('node must be an object')
        kind, identifier, props = node.get('type'), node.get('id'), node.get('properties')
        if not isinstance(kind, str) or kind not in SCHEMA or not isinstance(identifier, str):
            raise ValueError('invalid node type or ID')
        schema = SCHEMA[kind]
        if not identifier.startswith(schema['prefix']) or not identifier[len(schema['prefix']):].strip():
            raise ValueError(f'invalid {kind} ID: {identifier}')
        if identifier in nodes or not isinstance(props, dict):
            raise ValueError(f'duplicate node ID or invalid properties: {identifier}')
        for field, expected in schema['required'].items():
            if field not in props or not isinstance(props[field], expected):
                raise ValueError(f'{identifier}: invalid required property {field}')
        if kind == 'Room':
            if not props['room_id'].strip() or identifier != 'room:' + props['room_id']:
                raise ValueError('room ID does not match properties')
            string_array(props['tags'], 'tags')
            string_array(props['self_declared_concepts'], 'self_declared_concepts')
            if len(props['tags']) > 5:
                raise ValueError('room supports at most five tags')
        nodes[identifier] = node
    seen = set()
    for edge in graph['edges']:
        if not isinstance(edge, dict):
            raise ValueError('edge must be an object')
        source, target, kind = edge.get('source'), edge.get('target'), edge.get('type')
        if not all(isinstance(v, str) for v in (source, target, kind)):
            raise ValueError('edge IDs and type must be strings')
        if source not in nodes or target not in nodes or kind not in RELATIONS:
            raise ValueError('dangling edge or unsupported relationship')
        if (nodes[source]['type'], nodes[target]['type']) != RELATIONS[kind]:
            raise ValueError(f'invalid endpoint types for {kind}')
        if not isinstance(edge.get('properties'), dict):
            raise ValueError('edge properties must be an object')
        key = (source, kind, target)
        if key in seen:
            raise ValueError('duplicate edge')
        seen.add(key)
    for node in nodes.values():
        if node['type'] == 'Room':
            for concept in node['properties']['self_declared_concepts']:
                declared = nodes.get('concept:' + concept)
                if not declared or declared['properties'].get('category_id') != 'IDENTITY':
                    raise ValueError(f'invalid explicit identity declaration: {concept}')


def build_graph(taxonomy: dict) -> dict:
    utils.validate_taxonomy(taxonomy)
    graph = {'schema_version': 1, 'taxonomy_version': taxonomy['version'], 'nodes': [], 'edges': []}
    for node in taxonomy['nodes']:
        graph['nodes'].append({'id': 'concept:' + node['id'], 'type': 'Concept',
                               'properties': deepcopy(node)})
        if node['parent_id'] is not None:
            graph['edges'].append({'source': 'concept:' + node['id'],
                                   'target': 'concept:' + node['parent_id'],
                                   'type': 'IS_A', 'properties': {}})
    validate_graph(graph)
    return graph


def map_rooms(graph: dict, csv_path: Path, mapping: dict) -> dict:
    validate_graph(graph)
    required = {'room_id', 'subject', 'tags'}
    allowed = required | {'self_declared_concepts'}
    if (not isinstance(mapping, dict) or not required <= mapping.keys()
            or mapping.keys() - allowed or any(not isinstance(v, str) or not v for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('mapping needs distinct source columns for room_id, subject and tags')
    result = deepcopy(graph)
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(mapping.values()) <= set(reader.fieldnames):
            raise ValueError('CSV is missing mapped headers')
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('CSV has duplicate headers')
        for number, row in enumerate(reader, 2):
            if None in row or any(row[column] is None for column in mapping.values()):
                raise ValueError(f'CSV row {number} has an unexpected number of fields')
            props = {field: row[column] for field, column in mapping.items()}
            for field in ('tags', 'self_declared_concepts'):
                raw = props.get(field, '')
                try:
                    props[field] = string_array(json.loads(raw or '[]'), field)
                except (ValueError, TypeError) as error:
                    raise ValueError(f'CSV row {number}: invalid {field}: {error}') from error
            result['nodes'].append({'id': 'room:' + props['room_id'], 'type': 'Room', 'properties': props})
    validate_graph(result)
    return result


def query_graph(graph: dict, operation: str, **params) -> list:
    """Traverse both directions; relationship filters apply to every traversed edge."""
    validate_graph(graph)
    nodes = {node['id']: node for node in graph['nodes']}
    relation = params.get('relation')
    if relation is not None and relation not in RELATIONS:
        raise ValueError('unknown relationship filter')
    if operation == 'search':
        text = params.get('text', '')
        if not isinstance(text, str):
            raise ValueError('search text must be a string')
        return deepcopy([node for node in nodes.values()
                         if text.casefold() in json.dumps(node, ensure_ascii=False).casefold()])
    if operation not in ('neighbors', 'bfs', 'path'):
        raise ValueError('unknown query operation')
    start, target = params.get('node'), params.get('target')
    if not isinstance(start, str) or start not in nodes:
        raise ValueError(f'unknown node: {start}')
    if operation == 'path' and (not isinstance(target, str) or target not in nodes):
        raise ValueError(f'unknown target: {target}')
    depth = params.get('depth', 2)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    adjacency = {identifier: set() for identifier in nodes}
    for edge in graph['edges']:
        if relation is None or edge['type'] == relation:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
    if operation == 'neighbors':
        return deepcopy([nodes[n] for n in sorted(adjacency[start])])
    queue, seen, result = deque([(start, [start])]), {start}, []
    while queue:
        identifier, path = queue.popleft()
        item = {'node': nodes[identifier], 'path': path}
        if operation == 'path' and identifier == target:
            return deepcopy([item])
        if operation == 'bfs':
            result.append(item)
            if len(path) - 1 >= depth:
                continue
        for neighbor in sorted(adjacency[identifier]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return deepcopy(result)


def _classify_node(graph, room, classifier):
    if graph['taxonomy_version'] != classifier.version:
        raise ValueError('classifier taxonomy version does not match graph')
    # A matching version string is insufficient if the graph was edited independently.
    concepts = {n['id'][8:]: n['properties'] for n in graph['nodes'] if n['type'] == 'Concept'}
    if concepts != classifier.nodes:
        raise ValueError('graph concepts differ from classifier taxonomy')
    hierarchy = {(e['source'], e['target']) for e in graph['edges'] if e['type'] == 'IS_A'}
    expected = {('concept:' + identifier, 'concept:' + node['parent_id'])
                for identifier, node in classifier.nodes.items() if node['parent_id'] is not None}
    if hierarchy != expected:
        raise ValueError('graph hierarchy edges differ from classifier taxonomy')
    props = room['properties']
    result = classifier.classify_room(props['room_id'], props['subject'], props['tags'],
                                     self_declared_concepts=props['self_declared_concepts'])
    graph['edges'] = [e for e in graph['edges'] if not
                      (e['source'] == room['id'] and e['type'] in ('ASSIGNED_TO', 'INFERRED_AS'))]
    props['requires_context_review'] = result['requires_context_review']
    support = {}
    for assignment in result['assignments']:
        graph['edges'].append({'source': room['id'], 'target': 'concept:' + assignment['concept_id'],
                               'type': 'ASSIGNED_TO',
                               'properties': {**assignment, 'taxonomy_version': classifier.version}})
        path = [assignment['concept_id']]
        parent = classifier.nodes[path[-1]]['parent_id']
        while parent is not None:
            path = path + [parent]
            support.setdefault(parent, []).append(path)
            parent = classifier.nodes[parent]['parent_id']
    for ancestor in result['inferred_ancestors']:
        paths = support[ancestor]
        graph['edges'].append({'source': room['id'], 'target': 'concept:' + ancestor,
                               'type': 'INFERRED_AS',
                               'properties': {'taxonomy_version': classifier.version,
                                              'supporting_concepts': sorted({p[0] for p in paths}),
                                              'supporting_paths': paths}})
    return result


def infer_graph(graph: dict, classifier: utils.RoomClassifier) -> tuple[dict, list]:
    validate_graph(graph)
    if graph['taxonomy_version'] != classifier.version:
        raise ValueError('classifier taxonomy version does not match graph')
    result, trace = deepcopy(graph), []
    for node in result['nodes']:
        if node['type'] == 'Room':
            classified = _classify_node(result, node, classifier)
            trace.append({'operation': 'classify_room', 'room': node['id'], 'result': classified})
    validate_graph(result)
    return result, trace


OPERATIONS = {
    'classify_room': ('action', {'room_id'}),
    'create_review_task': ('action', {'room_id'}),
    'room_assignments': ('function', {'room_id'}),
    'pending_reviews': ('function', set()),
}


def _validate_step(step, parameters, kind):
    if not isinstance(step, dict) or set(step) != {'operation', 'params'}:
        raise ValueError('behavior step requires operation and params')
    operation, arguments = step['operation'], step['params']
    if not isinstance(operation, str) or operation not in OPERATIONS or OPERATIONS[operation][0] != kind:
        raise ValueError(f'unsupported {kind} operation: {operation}')
    if not isinstance(arguments, dict) or set(arguments) != OPERATIONS[operation][1]:
        raise ValueError(f'invalid parameters for {operation}')
    for value in arguments.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('step parameters must be nonempty strings')
        if value.startswith('$') and value[1:] not in parameters:
            raise ValueError(f'unknown parameter reference: {value}')


def validate_behaviors(definitions: dict) -> None:
    """Validate a closed registry: events call actions; actions cannot call events."""
    if (not isinstance(definitions, dict) or definitions.get('version') != 1
            or set(definitions) != {'version', 'actions', 'functions', 'events'}):
        raise ValueError('behavior registry needs version 1, actions, functions and events')
    for section in ('actions', 'functions', 'events'):
        entries = definitions[section]
        if not isinstance(entries, dict):
            raise ValueError(f'{section} must be an object')
        for name, entry in entries.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
                raise ValueError('behavior names and definitions must be nonempty objects')
            parameters = string_array(entry.get('parameters'), 'parameters')
            if (len(set(parameters)) != len(parameters)
                    or any(p not in ('room_id',) for p in parameters)):
                raise ValueError('supported behavior parameter is room_id')
            if section == 'actions':
                if set(entry) != {'parameters', 'steps'} or not isinstance(entry['steps'], list) or not entry['steps']:
                    raise ValueError('actions need a nonempty steps array')
                for step in entry['steps']:
                    _validate_step(step, parameters, 'action')
            elif section == 'functions':
                if set(entry) != {'parameters', 'operation', 'params'}:
                    raise ValueError('function requires parameters, operation and params')
                _validate_step({k: entry[k] for k in ('operation', 'params')}, parameters, 'function')
            else:
                if (set(entry) != {'parameters', 'condition', 'action'}
                        or entry['condition'] != 'requires_context_review' or parameters != ['room_id']
                        or not isinstance(entry['action'], str)):
                    raise ValueError('event requires room_id, requires_context_review and an action reference')
    for event in definitions['events'].values():
        action = definitions['actions'].get(event['action'])
        if action is None or not set(action['parameters']) <= set(event['parameters']):
            raise ValueError('event action reference or parameters are invalid')


def _room(graph, room_id):
    if not isinstance(room_id, str) or not room_id.strip():
        raise ValueError('room_id must be a nonempty string')
    for node in graph['nodes']:
        if node['type'] == 'Room' and node['id'] == 'room:' + room_id:
            return node
    raise ValueError(f'unknown room: {room_id}')


def _execute_operation(graph, operation, params, classifier):
    if operation == 'pending_reviews':
        return [n for n in graph['nodes'] if n['type'] == 'ReviewTask' and n['properties']['status'] == 'pending']
    room = _room(graph, params['room_id'])
    if operation == 'classify_room':
        return _classify_node(graph, room, classifier)
    if operation == 'room_assignments':
        return [e for e in graph['edges'] if e['source'] == room['id']
                and e['type'] in ('ASSIGNED_TO', 'INFERRED_AS')]
    pending = [n for n in graph['nodes'] if n['type'] == 'ReviewTask'
               and n['properties']['room_id'] == params['room_id'] and n['properties']['status'] == 'pending']
    if pending:
        return pending[0]
    identifier = 'review:' + params['room_id'] + ':context'
    used = {n['id'] for n in graph['nodes']}
    suffix = 1
    base = identifier
    while identifier in used:
        identifier = f'{base}:{suffix}'
        suffix += 1
    task = {'id': identifier, 'type': 'ReviewTask',
            'properties': {'room_id': params['room_id'], 'status': 'pending'}}
    graph['nodes'].append(task)
    graph['edges'].append({'source': room['id'], 'target': identifier, 'type': 'HAS_REVIEW', 'properties': {}})
    return task


def _diff(before, after):
    changes = {}
    for section in ('nodes', 'edges'):
        def key(item):
            return item['id'] if section == 'nodes' else (item['source'], item['type'], item['target'])
        old, new = ({key(item): item for item in graph[section]} for graph in (before, after))
        changes[section] = {
            'added': [new[k] for k in new if k not in old],
            'removed': [old[k] for k in old if k not in new],
            'updated': [{'before': old[k], 'after': new[k]} for k in new if k in old and new[k] != old[k]],
        }
    return changes


def run_sandbox(graph: dict, definitions: dict, kind: str, name: str, params: dict,
                classifier: utils.RoomClassifier) -> dict:
    """Execute transactionally on a private copy; failed runs return no graph."""
    validate_graph(graph)
    validate_behaviors(definitions)
    section = {'action': 'actions', 'function': 'functions', 'event': 'events'}.get(kind)
    if not section or not isinstance(name, str) or name not in definitions[section]:
        raise ValueError('unknown behavior kind or name')
    entry = definitions[section][name]
    if (not isinstance(params, dict) or set(params) != set(entry['parameters'])
            or any(not isinstance(v, str) or not v.strip() for v in params.values())):
        raise ValueError(f'{name} requires parameters {entry["parameters"]}')
    work, trace, result = deepcopy(graph), [], None
    if kind == 'event':
        room = _room(work, params['room_id'])
        if 'requires_context_review' not in room['properties']:
            raise ValueError('classify the room before evaluating its ambiguity event')
        matched = room['properties']['requires_context_review'] is True
        trace.append({'event': name, 'condition': entry['condition'], 'matched': matched})
        if not matched:
            return {'graph': work, 'changes': _diff(graph, work), 'trace': trace, 'result': None}
        entry = definitions['actions'][entry['action']]
    steps = entry['steps'] if kind != 'function' else [{k: entry[k] for k in ('operation', 'params')}]
    for number, step in enumerate(steps, 1):
        arguments = {k: params[v[1:]] if v.startswith('$') else v for k, v in step['params'].items()}
        result = _execute_operation(work, step['operation'], arguments, classifier)
        trace.append({'step': number, 'operation': step['operation'], 'params': arguments, 'result': deepcopy(result)})
    validate_graph(work)
    return deepcopy({'graph': work, 'changes': _diff(graph, work), 'trace': trace, 'result': result})
