"""Regression tests for bugs found in the BM25 engine.

Each test pins a specific defect so it cannot silently come back:

1. document_frequency was never incremented (the membership check ran after
   the posting was added, so the condition was always false).
2. Re-indexing an existing external doc_id left the old copy orphaned in the
   index, so the same doc_id came back twice in search results.
3. Query lookup only ever tried lowercase, so a lowercase query could not
   reach case-preserved index terms such as HTTP or XK9-2B4-7Q1.
4. index_document() rebuilt BM25 with default k1/b, silently discarding any
   tuned parameters.

Run directly (python test_regressions.py) or under pytest.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.disable(logging.CRITICAL)  # the engine logs at DEBUG by design

from bm25_engine import BM25, SparseSearchEngine  # noqa: E402


def _engine(docs):
    engine = SparseSearchEngine()
    engine.index_batch(docs)
    return engine


def test_document_frequency_is_populated():
    """df must equal the posting list size for every term."""
    engine = _engine([
        {"text": "machine learning is fun", "doc_id": "a"},
        {"text": "machine vision systems", "doc_id": "b"},
        {"text": "deep learning models", "doc_id": "c"},
    ])
    df = engine.index.document_frequency
    assert df["machine"] == 2
    assert df["learning"] == 2
    assert df["vision"] == 1
    for term, postings in engine.index.index.items():
        assert df[term] == len(postings), f"df out of sync for {term!r}"


def test_reindexing_same_doc_id_replaces_it():
    """Indexing an existing doc_id must update, not duplicate."""
    engine = SparseSearchEngine()
    engine.index_document("machine learning tutorial", external_doc_id="dup")
    engine.index_document("machine learning tutorial UPDATED", external_doc_id="dup")

    results = engine.search("machine learning")
    assert engine.index.total_documents == 1
    assert len(results) == 1, "the same doc_id was returned more than once"
    assert results[0]["text"].endswith("UPDATED"), "stale copy won"


def test_lookup_is_case_insensitive_for_codes_and_acronyms():
    """A lowercase query must still reach case-preserved index terms."""
    engine = SparseSearchEngine()
    engine.index_document("Error code XK9-2B4-7Q1 indicates a hardware fault",
                          external_doc_id="hw")
    engine.index_document("The HTTP 404 error code means not found",
                          external_doc_id="http")

    for query, expected in [("XK9-2B4-7Q1", "hw"), ("xk9-2b4-7q1", "hw"),
                            ("Xk9-2B4-7q1", "hw"), ("HTTP", "http"),
                            ("http", "http"), ("HtTp", "http")]:
        hits = [r["doc_id"] for r in engine.search(query)]
        assert hits, f"query {query!r} found nothing"
        assert hits[0] == expected, f"query {query!r} ranked {hits} first"


def test_tuned_bm25_parameters_survive_later_indexing():
    """index_document() must not reset a caller's k1/b."""
    engine = SparseSearchEngine()
    engine.index_document("alpha beta", external_doc_id="x")
    engine.bm25 = BM25(engine.index, k1=2.0, b=0.3)

    engine.index_document("gamma delta", external_doc_id="y")

    assert engine.bm25.k1 == 2.0
    assert engine.bm25.b == 0.3


def test_delete_document_removes_it_everywhere():
    """A deleted document must not resurface, and df must stay consistent."""
    engine = _engine([
        {"text": "alpha beta", "doc_id": "p"},
        {"text": "beta gamma", "doc_id": "q"},
    ])
    assert engine.delete_document("p") is True
    assert engine.delete_document("p") is False, "second delete should report missing"

    assert engine.index.total_documents == 1
    assert [r["doc_id"] for r in engine.search("alpha")] == []
    assert [r["doc_id"] for r in engine.search("beta")] == ["q"]
    for term, postings in engine.index.index.items():
        assert engine.index.document_frequency[term] == len(postings)
    assert "alpha" not in engine.index.index, "term with no postings was left behind"


def test_negative_top_k_is_rejected():
    """A negative top_k sliced results[:top_k], silently returning the WORST matches."""
    engine = _engine([{"text": f"learning topic {i}", "doc_id": f"d{i}"} for i in range(5)])

    assert len(engine.search("learning", top_k=5)) == 5
    assert engine.search("learning", top_k=0) == []
    for bad in (-1, -3):
        try:
            engine.search("learning", top_k=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"top_k={bad} should raise, it silently dropped results")


def test_empty_documents_are_rejected():
    """Empty docs add nothing but still count toward N and drag avgdl down."""
    engine = SparseSearchEngine()
    for blank in ("", "   ", "\n\t"):
        try:
            engine.index_document(blank)
        except ValueError:
            pass
        else:
            raise AssertionError(f"blank text {blank!r} should have been rejected")
    assert engine.index.total_documents == 0


def test_scoring_survives_zero_avgdl():
    """A BM25 built against an empty index must not divide by zero later."""
    from bm25_engine import InvertedIndex

    stale = BM25(InvertedIndex())      # avgdl == 0
    assert stale.avgdl == 0
    assert stale.calculate_term_score("absent", 0) == 0

    engine = SparseSearchEngine()
    engine.index_document("alpha beta gamma", external_doc_id="d")
    stale.index = engine.index          # stale avgdl, populated index
    stale.calculate_term_score("alpha", 0)  # must not raise ZeroDivisionError


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
