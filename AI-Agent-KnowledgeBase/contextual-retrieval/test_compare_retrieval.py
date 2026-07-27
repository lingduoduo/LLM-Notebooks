import json
from pathlib import Path

from compare_retrieval import build_arg_parser, load_corpus


ROOT = Path(__file__).parent
CORPUS_PATH = ROOT / "evaluation" / "contextual_retrieval_corpus.json"
EVAL_PATH = ROOT / "evaluation" / "retrieval_eval.json"


def test_bundled_corpus_covers_every_evaluation_gold_chunk():
    queries = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    chunks = load_corpus(str(CORPUS_PATH))

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    gold_ids = {query["gold_chunk_id"] for query in queries}

    assert gold_ids <= chunk_ids
    assert all(chunk["plain"] for chunk in chunks)
    assert all(chunk["contextual"] for chunk in chunks)


def test_cli_defaults_to_bundled_offline_corpus():
    args = build_arg_parser().parse_args([])

    assert args.corpus == "evaluation/contextual_retrieval_corpus.json"
