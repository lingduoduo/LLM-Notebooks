"""Workflow contracts: mapping, graph queries, evidence and sandbox isolation."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import utils
import workflow


class RoomFixtures:

    def setUp(self):
        self.graph = workflow.build_graph(utils.load_taxonomy())

    def rooms(self, records, mapping=None, graph=None):
        mapping = mapping or {key: key for key in ('room_id', 'subject', 'tags', 'self_declared_concepts')}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rooms.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(mapping.values()))
                writer.writeheader()
                writer.writerows(records)
            return workflow.map_rooms(self.graph if graph is None else graph, path, mapping)



class GraphTests(RoomFixtures, unittest.TestCase):
    def test_mapping_preserves_text_and_separates_entity_namespaces(self):
        before = copy.deepcopy(self.graph)
        graph = self.rooms([{'room_id': 'latex', 'subject': '你好, "chat"', 'tags': '["latex"]'},
                            {'room_id': 'empty', 'subject': '', 'tags': '[]'}])
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertIn('concept:latex', nodes)
        self.assertEqual(nodes['room:latex']['properties']['subject'], '你好, "chat"')
        self.assertEqual(nodes['room:empty']['properties']['subject'], '')
        self.assertEqual(self.graph, before)

    def test_mapping_rejects_invalid_records(self):
        for record in ({'room_id': '', 'tags': '[]'}, {'room_id': 'a', 'tags': 'bad'},
                       {'room_id': 'a', 'tags': '[1]'}, {'room_id': 'a', 'tags': json.dumps(['x'] * 6)},
                       {'room_id': 'a', 'tags': '[]', 'self_declared_concepts': '["latex"]'}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.rooms([record])
        with self.assertRaises(ValueError):
            self.rooms([{'room_id': 'a', 'tags': '[]'}] * 2)
        with self.assertRaises(ValueError):
            self.rooms([], mapping={'room_id': 'id'})

    def test_mapping_requires_source_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rooms.csv'
            path.write_text('id,title\na,chat\n')
            with self.assertRaises(ValueError):
                workflow.map_rooms(self.graph, path, {'room_id': 'id', 'subject': 'title', 'tags': 'tags'})

    def test_graph_rejects_dangling_edges_and_wrong_endpoint_types(self):
        for target in ('missing', 'concept:latex'):
            graph = copy.deepcopy(self.graph)
            graph['edges'].append({'source': 'concept:latex', 'target': target,
                                   'type': 'ASSIGNED_TO', 'properties': {}})
            with self.subTest(target=target), self.assertRaises(ValueError):
                workflow.validate_graph(graph)

    def test_queries_search_and_traverse_in_both_directions(self):
        found = workflow.query_graph(self.graph, 'search', text='latex')
        self.assertIn('concept:latex', [n['id'] for n in found])
        neighbors = workflow.query_graph(self.graph, 'neighbors', node='concept:latex')
        self.assertIn('concept:material_interest', [n['id'] for n in neighbors])
        path = workflow.query_graph(self.graph, 'path', node='concept:latex', target='concept:adult_content')
        self.assertEqual(path[0]['path'][0], 'concept:latex')
        self.assertEqual(path[0]['path'][-1], 'concept:adult_content')
        self.assertEqual(workflow.query_graph(self.graph, 'path', node='concept:latex', target='concept:orientation'), [])
        self.assertEqual(workflow.query_graph(self.graph, 'bfs', node='concept:latex', depth=0),
                         [{'node': next(n for n in self.graph['nodes'] if n['id'] == 'concept:latex'),
                           'path': ['concept:latex']}])
        with self.assertRaises(ValueError):
            workflow.query_graph(self.graph, 'bfs', node='concept:latex', depth=-1)
        with self.assertRaises(ValueError):
            workflow.query_graph(self.graph, 'neighbors', node='unknown')

    def test_shortest_path_terminates_on_cycles(self):
        graph = copy.deepcopy(self.graph)
        graph['edges'].append({'source': 'concept:adult_content', 'target': 'concept:latex',
                               'type': 'IS_A', 'properties': {}})
        result = workflow.query_graph(graph, 'path', node='concept:latex', target='concept:adult_content')
        self.assertEqual(result[0]['path'], ['concept:latex', 'concept:adult_content'])


class InferenceTests(RoomFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.classifier = utils.RoomClassifier(utils.load_taxonomy(), utils.load()[1])

    def test_inference_keeps_axes_identity_and_evidence(self):
        graph = self.rooms([
            {'room_id': 'format', 'subject': 'Private show', 'tags': '[]'},
            {'room_id': 'identity', 'subject': 'orientation', 'tags': '[]'},
            {'room_id': 'declared', 'subject': '', 'tags': '[]', 'self_declared_concepts': '["orientation"]'},
            {'room_id': 'ambiguous', 'subject': 'sexy latex', 'tags': '[]'},
        ])
        inferred, trace = workflow.infer_graph(graph, self.classifier)
        assignments = {(e['source'], e['target']) for e in inferred['edges'] if e['type'] != 'IS_A'}
        self.assertNotIn(('room:format', 'concept:adult_content'), assignments)
        self.assertNotIn(('room:identity', 'concept:orientation'), assignments)
        self.assertIn(('room:declared', 'concept:orientation'), assignments)
        direct = next(e for e in inferred['edges'] if e['source'] == 'room:ambiguous'
                      and e['target'] == 'concept:latex' and e['type'] == 'ASSIGNED_TO')
        self.assertEqual(direct['properties']['evidence'][0]['text'], 'sexy latex')
        ancestor = next(e for e in inferred['edges'] if e['source'] == 'room:ambiguous'
                        and e['target'] == 'concept:adult_content')
        self.assertIn(['latex', 'material_interest', 'fetish_kink', 'adult_content'],
                      ancestor['properties']['supporting_paths'])
        room = next(n for n in inferred['nodes'] if n['id'] == 'room:ambiguous')
        self.assertTrue(room['properties']['requires_context_review'])
        self.assertEqual(len(trace), 4)
        self.assertEqual(direct['properties']['taxonomy_version'], '1.3.0')
        self.assertFalse(any(e['source'].startswith('room:') for e in graph['edges']))

    def test_reclassification_replaces_stale_edges(self):
        graph = self.rooms([{'room_id': 'a', 'subject': 'latex', 'tags': '[]'}])
        graph, _ = workflow.infer_graph(graph, self.classifier)
        next(n for n in graph['nodes'] if n['id'] == 'room:a')['properties']['subject'] = 'hello'
        updated, _ = workflow.infer_graph(graph, self.classifier)
        self.assertFalse(any(e['source'] == 'room:a' for e in updated['edges']))
        self.assertEqual(next(n for n in updated['nodes'] if n['id'] == 'room:a')['properties']['subject'], 'hello')

    def test_inference_rejects_hierarchy_edges_that_disagree_with_taxonomy(self):
        graph = self.rooms([{'room_id': 'a', 'subject': 'latex', 'tags': '[]'}])
        graph['edges'] = [e for e in graph['edges'] if e['source'] != 'concept:latex']
        self.assertEqual(workflow.query_graph(graph, 'path', node='concept:latex',
                                             target='concept:adult_content'), [])
        before = copy.deepcopy(graph)
        with self.assertRaisesRegex(ValueError, 'hierarchy'):
            workflow.infer_graph(graph, self.classifier)
        self.assertEqual(graph, before)
        with self.assertRaisesRegex(ValueError, 'hierarchy'):
            workflow.run_sandbox(graph, behavior_fixture(), 'action', 'classify',
                                 {'room_id': 'a'}, self.classifier)

    def test_inference_rejects_mismatched_taxonomy_version(self):
        graph = self.rooms([])
        graph['taxonomy_version'] = '0.0.0'
        with self.assertRaises(ValueError):
            workflow.infer_graph(graph, self.classifier)


def behavior_fixture():
    return {'version': 1,
            'actions': {
                'classify': {'parameters': ['room_id'], 'steps': [
                    {'operation': 'classify_room', 'params': {'room_id': '$room_id'}}]},
                'review': {'parameters': ['room_id'], 'steps': [
                    {'operation': 'create_review_task', 'params': {'room_id': '$room_id'}}]}},
            'functions': {
                'assignments': {'parameters': ['room_id'], 'operation': 'room_assignments',
                                'params': {'room_id': '$room_id'}},
                'pending': {'parameters': [], 'operation': 'pending_reviews', 'params': {}}},
            'events': {'ambiguous': {'parameters': ['room_id'],
                                     'condition': 'requires_context_review', 'action': 'review'}}}


class SandboxTests(RoomFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.classifier = utils.RoomClassifier(utils.load_taxonomy(), utils.load()[1])
        self.graph = self.rooms([{'room_id': 'a', 'subject': 'sexy latex', 'tags': '[]'},
                                 {'room_id': 'b', 'subject': 'hello', 'tags': '[]'}])
        self.definitions = behavior_fixture()

    def run_behavior(self, kind, name, graph=None, params=None):
        return workflow.run_sandbox(self.graph if graph is None else graph, self.definitions,
                                    kind, name, {'room_id': 'a'} if params is None else params,
                                    self.classifier)

    def test_action_preserves_input_and_functions_do_not_mutate(self):
        before = json.dumps(self.graph, sort_keys=True)
        result = self.run_behavior('action', 'classify')
        self.assertEqual(json.dumps(self.graph, sort_keys=True), before)
        self.assertTrue(result['changes']['edges']['added'])
        queried = self.run_behavior('function', 'assignments', graph=result['graph'])
        self.assertIn('concept:latex', [e['target'] for e in queried['result']])
        self.assertEqual(queried['graph'], result['graph'])
        self.assertFalse(any(v for group in queried['changes'].values() for v in group.values()))

    def test_event_condition_and_idempotency(self):
        graph, _ = workflow.infer_graph(self.graph, self.classifier)
        first = self.run_behavior('event', 'ambiguous', graph=graph)
        second = self.run_behavior('event', 'ambiguous', graph=first['graph'])
        reviews = [n for n in second['graph']['nodes'] if n['type'] == 'ReviewTask']
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]['properties']['room_id'], 'a')
        self.assertEqual(second['changes']['nodes']['added'], [])
        skipped = self.run_behavior('event', 'ambiguous', graph=graph, params={'room_id': 'b'})
        self.assertEqual(skipped['graph'], graph)
        self.assertFalse(skipped['trace'][0]['matched'])
        pending = self.run_behavior('function', 'pending', graph=first['graph'], params={})
        self.assertEqual(len(pending['result']), 1)

    def test_action_failure_after_mutation_rolls_back(self):
        self.definitions['actions']['broken'] = {'parameters': ['room_id'], 'steps': [
            {'operation': 'create_review_task', 'params': {'room_id': '$room_id'}},
            {'operation': 'classify_room', 'params': {'room_id': 'missing'}}]}
        before = copy.deepcopy(self.graph)
        with self.assertRaises(ValueError):
            self.run_behavior('action', 'broken')
        self.assertEqual(self.graph, before)

    def test_behavior_validation_rejects_unknown_operations_references_and_cycles(self):
        invalid = []
        bad = behavior_fixture(); bad['actions']['review']['steps'][0]['operation'] = 'eval'; invalid.append(bad)
        bad = behavior_fixture(); bad['events']['ambiguous']['action'] = 'missing'; invalid.append(bad)
        bad = behavior_fixture(); bad['events']['ambiguous']['action'] = 'ambiguous'; invalid.append(bad)
        bad = behavior_fixture(); bad['actions']['review']['steps'][0]['params'] = {}; invalid.append(bad)
        bad = behavior_fixture(); bad['actions']['review']['steps'][0]['params']['room_id'] = '$missing'; invalid.append(bad)
        bad = behavior_fixture(); bad['functions']['pending']['operation'] = 'create_review_task'; invalid.append(bad)
        for definitions in invalid:
            with self.subTest(definitions=definitions), self.assertRaises(ValueError):
                workflow.validate_behaviors(definitions)
        for params in ({}, {'room_id': 'a', 'extra': 1}, {'room_id': 3}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.run_behavior('action', 'classify', params=params)
