"""Core active tool discovery based on embedding-vector similarity.

Given a natural-language capability need, retrieve the 3-5 most relevant
candidates from 126 tools.

- Tool vectors embed ``name: description`` and are cached in
  .cache/tool_embeddings_<embedder>.json to avoid repeated computation.
- discover_tools(need) embeds the need, compares cosine similarity against the
  tool vectors, and returns the top-k results.

Embedding backends are pluggable (see the ``Embedder`` protocol):
- OpenAIEmbedder calls the OpenAI embeddings API (online default; best quality).
- Offline mode uses offline_backend.LocalEmbedder, a local hash bag-of-words
  implementation for validating the pipeline and measuring tokens/latency.
"""

import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

from tools_library import ALL_TOOLS

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _tool_text(tool: Dict) -> str:
    f = tool["function"]
    return f"{f['name']}: {f['description']}"


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


class OpenAIEmbedder:
    """Embedding backend based on the OpenAI embeddings API."""

    def __init__(self, client, model: str = None):
        self.client = client
        self.model = model or EMBED_MODEL
        self.name = self.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class ToolIndex:
    """Tool vector index and similarity retrieval.

    embedder: An object with ``.embed(texts) -> List[vec]`` and ``.name``.
              For backward compatibility, an OpenAI client may be passed
              directly and will be wrapped in OpenAIEmbedder.
    tools:    Tool subset to index; defaults to ALL_TOOLS (used with
              --tool-set-size).
    """

    def __init__(self, embedder, tools: List[Dict] = None):
        self.embedder = embedder if hasattr(embedder, "embed") else OpenAIEmbedder(embedder)
        tools = tools if tools is not None else ALL_TOOLS
        self.names = [t["function"]["name"] for t in tools]
        self.texts = [_tool_text(t) for t in tools]
        self.vectors = self._load_or_build()

    def _cache_file(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.embedder.name)
        return os.path.join(_CACHE_DIR, f"tool_embeddings_{safe}.json")

    def _signature(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedder.name.encode())
        for t in self.texts:
            h.update(t.encode())
        return h.hexdigest()[:16]

    def _load_or_build(self) -> Dict[str, List[float]]:
        sig = self._signature()
        cache_file = self._cache_file()
        if os.path.exists(cache_file):
            try:
                cached = json.load(open(cache_file, encoding="utf-8"))
                if cached.get("signature") == sig:
                    return cached["vectors"]
            except Exception:
                pass
        # Cache missing or stale: generate embeddings in one backend batch.
        print(f"[discovery] Generating embeddings for {len(self.texts)} tools with {self.embedder.name} ...")
        embeddings = self.embedder.embed(self.texts)
        vectors = {name: vec for name, vec in zip(self.names, embeddings)}
        os.makedirs(_CACHE_DIR, exist_ok=True)
        json.dump({"signature": sig, "vectors": vectors},
                  open(cache_file, "w", encoding="utf-8"))
        return vectors

    def search(self, need: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """Return the top_k (tool name, similarity) pairs most relevant to need."""
        q = self.embedder.embed([need])[0]
        scored = [(name, _cosine(q, self.vectors[name])) for name in self.names]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
