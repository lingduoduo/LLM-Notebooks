"""Regression: RAPTOR clustering must not hand UMAP invalid parameters.

cluster_nodes() reduced embeddings with a fixed n_components=50 whenever the
embedding dimension exceeded 50. UMAP requires n_components < n_samples and
n_neighbors >= 2, so every tree level holding fewer than ~51 nodes raised:

    ValueError: n_components=50 must be < n_samples=5

RAPTOR builds levels that shrink as it summarizes upward, so upper levels
always hit this. The fix skips the reduction when the level is too small and
clamps both parameters otherwise.

Run directly (python test_raptor_clustering.py) or under pytest.
"""
import numpy as np

# conftest installs the dependency stubs. Importing it explicitly means running
# this file directly (python test_raptor_clustering.py) works too -- otherwise
# `import umap` below would load the real, possibly broken, library.
import conftest  # noqa: F401
import umap  # stubbed by conftest

from raptor_indexer import RaptorIndexer  # noqa: E402


def _indexer():
    return RaptorIndexer.__new__(RaptorIndexer)


def _embeddings(n_samples, dim=384):
    return np.random.default_rng(0).normal(size=(n_samples, dim)).astype("float32")


def test_small_levels_do_not_crash():
    """Upper tree levels hold few nodes; every size must cluster without raising."""
    indexer = _indexer()
    for n_samples in (2, 3, 5, 10, 20, 49, 51, 200):
        labels = indexer.cluster_nodes(_embeddings(n_samples))
        assert len(labels) == n_samples, f"wrong label count for {n_samples} samples"


def test_umap_parameters_stay_valid():
    """n_components must stay < n_samples and n_neighbors >= 2 whenever UMAP runs."""
    indexer = _indexer()
    for n_samples in (5, 20, 60, 200):
        umap.CALLS.clear()
        indexer.cluster_nodes(_embeddings(n_samples))
        for call in umap.CALLS:
            assert call["n_components"] < n_samples, (
                f"n_components={call['n_components']} not < n_samples={n_samples}")
            assert call["n_neighbors"] >= 2, f"n_neighbors={call['n_neighbors']} < 2"


def test_reduction_skipped_when_pointless():
    """Reducing fewer samples than target dimensions is meaningless -- skip it."""
    indexer = _indexer()
    for n_samples in (2, 3):
        umap.CALLS.clear()
        indexer.cluster_nodes(_embeddings(n_samples))
        assert not umap.CALLS, f"UMAP should be skipped for {n_samples} samples"


def test_single_sample_returns_early():
    indexer = _indexer()
    assert len(indexer.cluster_nodes(_embeddings(1))) == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
