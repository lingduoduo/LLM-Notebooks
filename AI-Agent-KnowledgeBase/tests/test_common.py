"""Shared data contracts, independent of models, servers and API credentials."""

import json
import tempfile
import unittest
from pathlib import Path

from knowledge_common.datasets import InputError, load_corpus, load_labels
from knowledge_common.evaluation import metrics_for_query


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_json_envelope_and_jsonl(self):
        documents = [{"doc_id": "one", "text": "你好 world"}]
        for data in (documents, {"documents": documents}):
            self.assertEqual(
                load_corpus(self.write("corpus.json", data), []), documents
            )
        path = self.root / "corpus.jsonl"
        path.write_text("\n" + json.dumps(documents[0]) + "\n", encoding="utf-8")
        self.assertEqual(load_corpus(path, []), documents)

    def test_defaults_are_supplied_by_caller(self):
        documents = [{"text": "example"}]
        labels = {"query": ["one"]}
        self.assertEqual(load_corpus(None, documents), documents)
        self.assertEqual(load_labels(None, labels), labels)

    def test_invalid_corpus_has_actionable_errors(self):
        for data in ([], {}, 3, "text", [{"text": None}], [{"text": 4}], ["text"]):
            with self.subTest(data=data), self.assertRaises(InputError):
                load_corpus(self.write("bad.json", data), [])
        with self.assertRaisesRegex(InputError, "not found"):
            load_corpus(self.root / "missing.json", [])
        path = self.root / "invalid.jsonl"
        path.write_text('{"text":"okay"}\nnot json\n', encoding="utf-8")
        with self.assertRaisesRegex(InputError, "line 2"):
            load_corpus(path, [])

    def test_labels_validate_relevant_id_lists(self):
        labels = {"query": ["one", "two"], "no hits": []}
        self.assertEqual(load_labels(self.write("labels.json", labels), {}), labels)
        for data in ({}, [], {"query": "one"}, {"query": [1]}, {"": ["one"]}):
            with self.subTest(data=data), self.assertRaises(InputError):
                load_labels(self.write("labels.json", data), {})

    def test_unreadable_text_is_reported_as_input_error(self):
        path = self.root / "invalid.json"
        path.write_bytes(b"\xff")
        for loader in (load_corpus, load_labels):
            with self.subTest(loader=loader), self.assertRaises(InputError):
                loader(path, {})


class MetricTests(unittest.TestCase):
    def test_metrics_preserve_rr_over_full_ranking_and_precision_over_returned_hits(
        self,
    ):
        self.assertEqual(
            metrics_for_query(["wrong", "one"], {"one", "two"}, 1),
            {
                "recall": 0.0,
                "precision": 0.0,
                "rr": 0.5,
                "hits": [],
                "retrieved": ["wrong"],
            },
        )
        result = metrics_for_query(["one"], {"one", "two"}, 5)
        self.assertEqual(
            (result["recall"], result["precision"], result["rr"]), (0.5, 1.0, 1.0)
        )

    def test_empty_rankings_and_labels(self):
        self.assertEqual(
            metrics_for_query([], set(), 5),
            {
                "recall": 0.0,
                "precision": 0.0,
                "rr": 0.0,
                "hits": [],
                "retrieved": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
