"""CLI loading and evaluation work without native ANN backends or model weights."""

import sys
from unittest.mock import patch


def test_loading_corpus_does_not_import_optional_ann_backends():
    with patch.dict(sys.modules, {"annoy": None, "hnswlib": None}):
        import cli

        assert len(cli.load_corpus(None)) == 12
        assert cli.DEFAULT_CORPUS[-1]["doc_id"] == "doc_12"
        assert cli.load_labels(None)["a cat playing"] == ["doc_7", "doc_8"]
        assert cli._metrics_for_query(["doc_7"], {"doc_7"}, 5)["recall"] == 1
