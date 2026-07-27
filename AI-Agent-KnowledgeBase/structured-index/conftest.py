"""Shared test stubs for the heavy indexing dependencies.

The RAPTOR/GraphRAG modules import umap, sentence-transformers, openai and
friends at module level. The unit tests only exercise pure logic (chunking,
clustering parameters), so this installs one complete set of stubs before any
test module imports them.

It lives in conftest.py because each test file previously installed its own
partial stubs via sys.modules.setdefault(). Whichever file imported first won,
so the files passed individually but failed when pytest collected them together
-- the second file silently got the first file's incomplete stubs.
"""
import sys
import types

import numpy as np


def _install_stubs() -> None:
    for name in ["tiktoken", "tqdm", "openai", "sentence_transformers", "loguru",
                 "sklearn", "sklearn.mixture", "sklearn.metrics",
                 "sklearn.metrics.pairwise", "umap"]:
        sys.modules.setdefault(name, types.ModuleType(name))

    class GaussianMixture:
        def __init__(self, n_components=2, random_state=None):
            self.n_components = n_components

        def fit_predict(self, X):
            if self.n_components > len(X):
                raise ValueError(
                    f"n_components={self.n_components} > n_samples={len(X)}")
            return np.zeros(len(X), dtype=int)

    umap_stub = sys.modules["umap"]
    # Every UMAP construction is recorded here so tests can assert on the
    # parameters the indexer chose.
    umap_stub.CALLS = []

    class UMAP:
        def __init__(self, n_components=2, n_neighbors=15, **kwargs):
            umap_stub.CALLS.append(
                {"n_components": n_components, "n_neighbors": n_neighbors})
            # Constraints enforced by the real umap-learn:
            if n_neighbors < 2:
                raise ValueError(f"n_neighbors must be >= 2, got {n_neighbors}")
            self.n_components = n_components

        def fit_transform(self, X):
            if self.n_components >= len(X):
                raise ValueError(
                    f"n_components={self.n_components} must be < n_samples={len(X)}")
            return X[:, :self.n_components]

    umap_stub.UMAP = UMAP

    sys.modules["sklearn.mixture"].GaussianMixture = GaussianMixture
    sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: None
    sys.modules["openai"].OpenAI = object
    sys.modules["sentence_transformers"].SentenceTransformer = object
    sys.modules["loguru"].logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    sys.modules["tqdm"].tqdm = lambda x, **k: x
    # NOTE: `config` is deliberately NOT stubbed. Stubbing it shadowed the real
    # module and broke test_indexing.py's `from config import get_raptor_config`.


_install_stubs()
