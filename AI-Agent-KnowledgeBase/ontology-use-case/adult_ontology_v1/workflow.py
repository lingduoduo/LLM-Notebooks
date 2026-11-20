"""Offline ontology graph, mapping, inference and allowlisted behavior execution."""
from collections import deque
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import re

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
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as stream:
        return _map_stream(graph, stream, mapping)


def map_rooms_text(graph: dict, text: str, mapping: dict) -> dict:
    """Map CSV supplied as text (an edited preview); never touches files."""
    if not isinstance(text, str):
        raise ValueError('CSV must be text')
    return _map_stream(graph, io.StringIO(text.lstrip('\ufeff'), newline=''), mapping)


def csv_headers(text: str) -> list:
    if not isinstance(text, str):
        raise ValueError('CSV must be text')
    return next(csv.reader(io.StringIO(text.lstrip('\ufeff'), newline='')), [])


def _map_stream(graph: dict, stream, mapping: dict) -> dict:
    validate_graph(graph)
    required = {'room_id', 'subject', 'tags'}
    allowed = required | {'self_declared_concepts'}
    if (not isinstance(mapping, dict) or not required <= mapping.keys()
            or mapping.keys() - allowed or any(not isinstance(v, str) or not v for v in mapping.values())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError('mapping needs distinct source columns for room_id, subject and tags')
    result = deepcopy(graph)
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
    if operation == 'pattern':
        return deepcopy(_pattern_query(graph, nodes, params.get('text')))
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


_NODE = re.compile(r'\(\s*([A-Za-z_]\w*)\s*(?::\s*([A-Za-z]\w*))?\s*\)')
_REL = re.compile(r'\s*-\[\s*(?::\s*([A-Z_]+))?\s*\]->\s*|\s*-->\s*')
_COND = re.compile(r'([A-Za-z_]\w*)\.(\w+)\s*(=|CONTAINS)\s*(\'[^\']*\'|"[^"]*"|true|false|-?\d+)', re.I)
PATTERN_LIMIT = 200


def parse_pattern(text: str) -> dict:
    """Parse a small Cypher-like subset: MATCH (a:Type)-[:REL]->(b) [WHERE a.x = 'v' AND ...] RETURN a, b."""
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError('pattern must be text of at most 1000 characters')
    match = re.fullmatch(r'\s*MATCH\s+(.+?)(?:\s+WHERE\s+(.+?))?\s+RETURN\s+(.+?)\s*;?\s*', text, re.I | re.S)
    if not match:
        raise ValueError('use MATCH (a:Type)-[:REL]->(b) [WHERE a.prop = "value"] RETURN a, b')
    chain, where, returns = match.groups()
    position, nodes, relations = 0, [], []
    while True:
        node = _NODE.match(chain, position)
        if not node:
            raise ValueError(f'expected a node pattern like (a:Room) at: {chain[position:position + 20]!r}')
        nodes.append({'var': node.group(1), 'type': node.group(2)})
        position = node.end()
        if position == len(chain.rstrip()):
            break
        relation = _REL.match(chain, position)
        if not relation:
            raise ValueError(f'expected -[:RELATION]-> at: {chain[position:position + 20]!r}')
        if relation.group(1) is not None and relation.group(1) not in RELATIONS:
            raise ValueError(f'unknown relationship: {relation.group(1)}')
        relations.append(relation.group(1))
        position = relation.end()
    if len(nodes) > 4:
        raise ValueError('patterns support at most three relationship hops')
    variables = [n['var'] for n in nodes]
    if len(set(variables)) != len(variables):
        raise ValueError('each node variable may appear once')
    for n in nodes:
        if n['type'] is not None and n['type'] not in SCHEMA:
            raise ValueError(f'unknown node type: {n["type"]}')
    conditions = []
    for part in re.split(r'\s+AND\s+', where or '', flags=re.I) if where else []:
        condition = _COND.fullmatch(part.strip())
        if not condition or condition.group(1) not in variables:
            raise ValueError(f'unsupported condition: {part.strip()}')
        var, prop, op, raw = condition.groups()
        value = raw[1:-1] if raw[0] in '\'"' else {'true': True, 'false': False}.get(raw.lower(), raw)
        if isinstance(value, str) and raw[0] not in '\'"':
            value = int(value)
        conditions.append({'var': var, 'property': prop, 'op': op.upper(), 'value': value})
    returned = [v.strip() for v in returns.split(',')]
    if not returned or any(v not in variables for v in returned):
        raise ValueError('RETURN must list variables from the MATCH pattern')
    return {'nodes': nodes, 'relations': relations, 'conditions': conditions, 'returns': returned}


def _pattern_query(graph, nodes, text):
    plan = parse_pattern(text)
    outgoing = {}
    for edge in graph['edges']:
        outgoing.setdefault(edge['source'], []).append(edge)

    def accepts(spec, node):
        if spec['type'] is not None and node['type'] != spec['type']:
            return False
        for c in plan['conditions']:
            if c['var'] == spec['var']:
                actual = node['properties'].get(c['property'], node.get(c['property']))
                if c['op'] == '=' and actual != c['value']:
                    return False
                if c['op'] == 'CONTAINS' and str(c['value']).casefold() not in json.dumps(actual, ensure_ascii=False).casefold():
                    return False
        return True

    results = []
    def extend(path):
        if len(results) >= PATTERN_LIMIT:
            return
        if len(path) == len(plan['nodes']):
            binding = dict(zip([n['var'] for n in plan['nodes']], path))
            first = plan['returns'][0]
            results.append({'node': nodes[binding[first]], 'path': path,
                            'bindings': {v: binding[v] for v in plan['returns']}})
            return
        relation, spec = plan['relations'][len(path) - 1], plan['nodes'][len(path)]
        for edge in outgoing.get(path[-1], []):
            if (relation is None or edge['type'] == relation) and edge['target'] not in path \
                    and accepts(spec, nodes[edge['target']]):
                extend(path + [edge['target']])

    for node in nodes.values():
        if accepts(plan['nodes'][0], node):
            extend([node['id']])
    return results


RULE_RELATIONS = set(RELATIONS) | {'HAS_CONCEPT'}


def _premise_vars(premise):
    if 'not' in premise:
        return set()
    terms = [premise.get('var'), premise.get('source'), premise.get('target'), premise.get('of')]
    return {t for t in terms if isinstance(t, str) and t.startswith('?')}


def _validate_premise(premise, bound, negated=False):
    if not isinstance(premise, dict):
        raise ValueError('rule premise must be an object')
    if set(premise) == {'not'}:
        if negated:
            raise ValueError('nested negation is not supported')
        _validate_premise(premise['not'], bound, True)
        return
    def term(value):
        if not isinstance(value, str) or not value or (value.startswith('?') and len(value) < 2):
            raise ValueError('rule terms must be variables (?x) or node IDs')
    if set(premise) == {'type', 'var'}:
        term(premise['var'])
        if premise['type'] not in SCHEMA:
            raise ValueError(f'unknown rule node type: {premise["type"]}')
    elif set(premise) == {'relation', 'source', 'target'}:
        term(premise['source'])
        term(premise['target'])
        if premise['relation'] not in RULE_RELATIONS:
            raise ValueError(f'unknown rule relationship: {premise["relation"]}')
    elif set(premise) == {'property', 'of', 'equals'}:
        term(premise['of'])
        if not isinstance(premise['property'], str) or premise['of'] not in bound:
            raise ValueError('property premises need a variable bound by an earlier premise')
        if not isinstance(premise['equals'], (str, bool, int)):
            raise ValueError('property values must be strings, numbers or booleans')
    else:
        raise ValueError('unsupported rule premise')


def validate_rules(definitions: dict) -> None:
    if (not isinstance(definitions, dict) or definitions.get('version') != 1
            or set(definitions) != {'version', 'rules', 'contradictions'}):
        raise ValueError('rule registry needs version 1, rules and contradictions')
    facts = set()
    if not isinstance(definitions['rules'], dict) or not definitions['rules']:
        raise ValueError('rules must be a nonempty object')
    for name, rule in definitions['rules'].items():
        if (not isinstance(rule, dict) or set(rule) != {'label', 'premises', 'conclusion'}
                or not isinstance(rule['label'], str) or not isinstance(rule['premises'], list) or not rule['premises']):
            raise ValueError(f'rule {name} needs label, premises and conclusion')
        bound = set()
        for premise in rule['premises']:
            _validate_premise(premise, bound)
            bound |= _premise_vars(premise)
        conclusion = rule['conclusion']
        if (not isinstance(conclusion, dict) or set(conclusion) != {'fact', 'subject'}
                or not isinstance(conclusion['fact'], str) or not re.fullmatch(r'[A-Z]\w*', conclusion['fact'])
                or conclusion['subject'] not in bound):
            raise ValueError(f'rule {name} conclusion needs a Fact name and a bound subject')
        facts.add(conclusion['fact'])
    pairs = definitions['contradictions']
    if not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 or not set(p) <= facts for p in pairs):
        raise ValueError('contradictions must be pairs of concluded facts')


def describe_premise(premise):
    if 'not' in premise:
        return '¬' + describe_premise(premise['not'])
    if 'type' in premise:
        return f'{premise["type"]}({premise["var"]})'
    if 'relation' in premise:
        return f'{premise["relation"]}({premise["source"]}, {premise["target"]})'
    return f'{premise["property"]}({premise["of"]}) = {json.dumps(premise["equals"], ensure_ascii=False)}'


def describe_rule(rule):
    conclusion = rule['conclusion']
    return ' ∧ '.join(describe_premise(p) for p in rule['premises']) + f' → {conclusion["fact"]}({conclusion["subject"]})'


def run_rules(graph: dict, definitions: dict) -> dict:
    """Forward-chain one pass of conjunctive rules over the graph; never mutates it."""
    validate_graph(graph)
    validate_rules(definitions)
    nodes = {n['id']: n for n in graph['nodes']}
    edges = [(e['source'], e['type'], e['target']) for e in graph['edges']]
    edges += [(s, 'HAS_CONCEPT', t) for s, kind, t in edges if kind in ('ASSIGNED_TO', 'INFERRED_AS')]

    def resolve(term, binding):
        return binding.get(term) if term.startswith('?') else term

    def match(premise, binding):
        if 'not' in premise:
            if not any(True for _ in match(premise['not'], binding)):
                yield binding, 'no match'
            return
        if 'type' in premise:
            value = resolve(premise['var'], binding)
            for node in ([nodes[value]] if value in nodes else [] if value else nodes.values()):
                if node['type'] == premise['type']:
                    yield {**binding, premise['var']: node['id']}, f'{premise["type"]}({node["id"]})'
            return
        if 'relation' in premise:
            source, target = resolve(premise['source'], binding), resolve(premise['target'], binding)
            for s, kind, t in edges:
                if kind == premise['relation'] and source in (None, s) and target in (None, t):
                    extra = {k: v for k, v in ((premise['source'], s), (premise['target'], t)) if k.startswith('?')}
                    yield {**binding, **extra}, f'{kind}({s}, {t})'
            return
        node = nodes.get(resolve(premise['of'], binding))
        if node is not None and node['properties'].get(premise['property']) == premise['equals']:
            yield binding, f'{premise["property"]}({node["id"]}) = {json.dumps(premise["equals"], ensure_ascii=False)}'

    def solve(premises, binding, steps):
        if not premises:
            yield binding, steps
            return
        for extended, fact in match(premises[0], binding):
            yield from solve(premises[1:], extended,
                             steps + [{'premise': describe_premise(premises[0]), 'matched': fact}])

    inferences, seen = [], set()
    for name, rule in definitions['rules'].items():
        for binding, steps in solve(rule['premises'], {}, []):
            subject = binding[rule['conclusion']['subject']]
            if (name, subject) in seen:
                continue
            seen.add((name, subject))
            inferences.append({'rule': name, 'label': rule['label'], 'fact': rule['conclusion']['fact'],
                               'subject': subject, 'bindings': binding, 'steps': steps})
    by_subject = {}
    for item in inferences:
        by_subject.setdefault(item['subject'], set()).add(item['fact'])
    contradictions = [{'subject': subject, 'facts': list(pair)}
                      for subject, facts in sorted(by_subject.items())
                      for pair in definitions['contradictions'] if set(pair) <= facts]
    rules = [{'name': name, 'label': rule['label'], 'text': describe_rule(rule)}
             for name, rule in definitions['rules'].items()]
    return deepcopy({'rules': rules, 'inferences': inferences, 'contradictions': contradictions})


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
