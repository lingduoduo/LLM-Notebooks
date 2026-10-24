"""Offline fusion contracts used by both the service and evaluation CLI."""

import pytest

from fusion import fuse, min_max_normalize


def test_rrf_combines_rank_evidence_independently_of_score_scale():
    ranks = {"dense": [("a", 0.9), ("b", 0.8)], "sparse": [("b", 900), ("c", 800)]}
    result = fuse(ranks)
    assert [doc for doc, _ in result] == ["b", "a", "c"]
    assert dict(result)["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_fusion_normalizes_each_source_and_keeps_single_source_hits():
    ranks = {"dense": [("a", 10), ("b", 5)], "sparse": [("b", 100), ("c", 0)]}
    assert fuse(ranks, method="weighted", weights={"dense": 0.25, "sparse": 0.75}) == [
        ("b", 0.75),
        ("a", 0.25),
        ("c", 0.0),
    ]
    assert min_max_normalize({"a": 3, "b": 3}) == {"a": 1.0, "b": 1.0}
    assert min_max_normalize({}) == {}


def test_empty_fusion_and_invalid_method():
    assert fuse({}) == []
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fuse({}, method="invalid")
