"""Regression: equal chunk_size/overlap must not crash range() with step 0."""
import sys
import types
from dataclasses import dataclass


def _stub_raptor_deps() -> None:
    mods = [
        "tiktoken",
        "tqdm",
        "umap",
        "openai",
        "sentence_transformers",
        "loguru",
        "sklearn",
        "sklearn.mixture",
        "sklearn.metrics",
        "sklearn.metrics.pairwise",
    ]
    for name in mods:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sklearn.mixture"].GaussianMixture = object
    sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: None
    sys.modules["openai"].OpenAI = object
    sys.modules["sentence_transformers"].SentenceTransformer = object
    sys.modules["loguru"].logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    sys.modules["tqdm"].tqdm = lambda x, **k: x


_stub_raptor_deps()

from raptor_indexer import RaptorIndexer  # noqa: E402


@dataclass
class _Cfg:
    chunk_size: int = 1000
    chunk_overlap: int = 1000


def test_chunk_text_equal_size_and_overlap():
    indexer = RaptorIndexer.__new__(RaptorIndexer)
    indexer.config = _Cfg()
    words = ("alpha beta gamma " * 200).strip()
    chunks = indexer.chunk_text(words)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_chunk_text_normal_overlap_still_advances():
    indexer = RaptorIndexer.__new__(RaptorIndexer)
    indexer.config = _Cfg(chunk_size=10, chunk_overlap=2)
    chunks = indexer.chunk_text(" ".join(f"w{i}" for i in range(30)))
    assert len(chunks) > 1


if __name__ == "__main__":
    # These are pytest-style tests; running this file as a plain script would
    # define them and exit 0 without executing any, which reads as "passed".
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
