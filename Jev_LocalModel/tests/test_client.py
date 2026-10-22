"""Exercise the public wrapper without loading a checkpoint."""
import unittest
from Jev_LocalModel.client import LayaClient


class AgentStub:
    def system_one(self, state, questions):
        return {"model": "rl-agent", "answers": {
            q: {"type": definition["type"], "rl_agent": {"act_probability": 0.5}}
            for q, definition in questions.items()
        }, "usage": {"input_tokens": 3, "output_tokens": 0}}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = LayaClient.__new__(LayaClient)
        self.client.model_name = "laya-multilingual"
        self.client._agent = AgentStub()

    def test_invalid_questions_rejected_before_inference(self):
        invalid = [None, [], {}, {"": {"type": "noul", "instructions": "x"}},
                   {"q": None}, {"q": {}},
                   {"q": {"type": [], "instructions": "x"}},
                   {"q": {"type": "other", "instructions": "x"}},
                   {"q": {"type": "noul", "instructions": None}},
                   {"q": {"type": "choice", "instructions": "x", "criteria": []}},
                   {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},
                   {"q": {"type": "choice", "instructions": "x", "criteria": [1, 2]}},
                   {"q": {"type": "score", "instructions": "x", "criteria": {"0": "low"}}},
                   {"q": {"type": "score", "instructions": "x", "criteria": []}},
                   {"q": {"type": "noul", "instructions": "x", "criteria": ["no", "yes"]}},
                   {"q": {"type": "noul", "instructions": "x", "criteria": {"true": 1}}}]
        for questions in invalid:
            with self.subTest(questions=questions), self.assertRaises((TypeError, ValueError)):
                self.client.system_one("state", questions)

    def test_invalid_model_names_rejected(self):
        for model in ("", [], {}, 1, "different"):
            with self.subTest(model=model), self.assertRaises((TypeError, ValueError)):
                self.client.system_one("state", {"q": {"type": "noul", "instructions": "x"}}, model=model)

    def test_supported_questions_and_response_cleanup(self):
        questions = {
            "choice": {"type": "choice", "instructions": "choose", "criteria": ["a", "b"]},
            "mapping": {"type": "choice", "instructions": {"task": "choose"}, "criteria": {"a": None, "b": "B"}},
            "score": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
            "noul": {"type": "noul", "instructions": "true?"},
        }
        result = self.client.system_one({"text": "中文"}, questions, model="laya-latest")
        self.assertEqual(result["model"], "laya-multilingual")
        self.assertEqual(result["usage"], {"input_tokens": 3, "output_tokens": 0})
        self.assertEqual(set(result["answers"]), set(questions))
        self.assertTrue(all("rl_agent" not in answer for answer in result["answers"].values()))
