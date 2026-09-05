import json
import unittest
from unittest.mock import patch
from benchmark import seed_examples


class BackendTests(unittest.TestCase):
    def test_backend_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('llm_backend'))

    def test_labels_never_sent_and_pass_computed_locally(self):
        from llm_backend import LLMJudge
        class Client:
            def complete(self, system, payload):
                self.payload = payload
                return dict(groundedness=0, relevance=2, privacy_safety=2,
                            clarity=2, unsupported_claims=['claim'], rationale='Unsupported')
        client = Client()
        result = LLMJudge(client).evaluate(seed_examples()[2])
        self.assertFalse(result.passed)
        self.assertNotIn('human', client.payload)
        self.assertNotIn('id', client.payload)

    def test_malformed_scores_rejected(self):
        from llm_backend import parse_result
        for score in [True, 3, -1, '2', 1.5]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                parse_result(dict(groundedness=score, relevance=2, privacy_safety=2,
                                  clarity=2, unsupported_claims=[], rationale='Reason'))

    def test_transport_request_and_response(self):
        from llm_backend import ChatClient
        from io import BytesIO
        response = {'choices': [{'message': {'content': '{"text":"hello"}'}, 'finish_reason': 'stop'}]}
        with patch('llm_backend.urlopen', return_value=BytesIO(json.dumps(response).encode())) as call:
            result = ChatClient('test-model', 'http://localhost:1234/v1').complete('JSON please', {'x': 1})
        self.assertEqual(result, {'text': 'hello'})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:1234/v1/chat/completions')
        self.assertEqual(json.loads(request.data)['model'], 'test-model')

    def test_api_failure_fails_closed(self):
        from llm_backend import BackendError
        from generation import generate_with_guardrail, DemoGenerator
        class BrokenJudge:
            def evaluate(self, ex):
                raise BackendError('unavailable')
        result = generate_with_guardrail(seed_examples()[0], DemoGenerator(), BrokenJudge(), 1)
        self.assertTrue(result.used_fallback)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.traces), 2)
