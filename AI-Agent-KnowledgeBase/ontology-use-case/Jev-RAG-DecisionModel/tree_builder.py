from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from jev import RETRY_DELAYS

ROOT_ID = "__root__"
ROOT_DESCRIPTION = "All keywords"
MAX_DESCRIBED = 60
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "intfloat/e5-base-v2"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
USER_AGENT = "jev-decisionrag-prototype/0.1"
DESCRIBE_PROMPT = (
    "Write one sentence (at most 30 words) describing what the items in this "
    "group are about. Name the main kinds of subjects it covers, including the "
    "less typical ones, so a reader can tell whether a given item belongs to "
    "this group.")


class DescriberUnavailable(RuntimeError):
    """The describer's credentials are rejected; no further calls are made."""


class E5Embedder:
    def __init__(self, model: str = DEFAULT_EMBED_MODEL):
        self.model = model
        self._st = None

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model)
        vecs = self._st.encode(["passage: " + t for t in texts],
                               normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


def sklearn_kmeans(vectors, k: int, seed: int) -> list[int]:
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
    return [int(x) for x in labels]


class OpenAIDescriber:
    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL,
                 timeout: float = 60):
        self.model = model
        self.timeout = timeout
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.disabled = False
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "User-Agent": USER_AGENT}

    def __repr__(self) -> str:  # never expose the key
        return f"OpenAIDescriber(model={self.model!r})"

    def __call__(self, keywords: list[str]) -> str:
        if self.disabled:
            raise DescriberUnavailable("OpenAI credentials were rejected")
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": DESCRIBE_PROMPT},
                             {"role": "user", "content": ", ".join(keywords)}]}
        req = urllib.request.Request(
            OPENAI_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST", headers=self._headers)
        payload = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            last = attempt == len(RETRY_DELAYS)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    self.disabled = True
                    raise DescriberUnavailable(f"OpenAI returned HTTP {e.code}") from e
                if last or not (e.code == 429 or e.code >= 500):
                    raise RuntimeError(f"OpenAI HTTP {e.code}") from e
            except OSError as e:  # URLError, timeouts, resets
                if last:
                    raise RuntimeError(f"OpenAI request failed: {type(e).__name__}") from e
            else:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except ValueError as e:
                    raise RuntimeError("invalid JSON in OpenAI response") from e
                break
            time.sleep(RETRY_DELAYS[attempt])
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("unexpected OpenAI response shape") from e
        usage = payload.get("usage") or {}
        self.prompt_tokens += (usage.get("prompt_tokens") or 0)
        self.completion_tokens += (usage.get("completion_tokens") or 0)
        return (text or "").strip()


def load_keyword_rows(rows: Iterable[dict]) -> list[dict]:
    return [{"keyword": r["keyword"], "concept_id": r["concept_id"]}
            for r in rows if r.get("review_status") != "excluded"]


def fallback_description(keywords: list[str]) -> str:
    return "Terms such as: " + ", ".join(keywords[:8])


def _by_centrality(idx: list[int], vectors: list[list[float]]) -> list[int]:
    dim = len(vectors[idx[0]])
    mean = [sum(vectors[i][d] for i in idx) / len(idx) for d in range(dim)]

    def dist(i: int) -> float:
        return sum((vectors[i][d] - mean[d]) ** 2 for d in range(dim))

    return sorted(idx, key=dist)  # stable: ties keep file order


def _groups(idx: list[int], vectors, kmeans, k: int, seed: int) -> list[list[int]]:
    """Cluster idx into <= k groups, numbered by first member's file position."""
    k = min(k, len(idx))
    labels = kmeans([vectors[i] for i in idx], k, seed)
    by_label: dict[int, list[int]] = {}
    for i, lab in zip(idx, labels):
        by_label.setdefault(lab, []).append(i)
    return sorted(by_label.values(), key=lambda g: g[0])


def build_keyword_tree(rows: list[dict], embed: Callable, kmeans: Callable,
                       describe: Callable, *, clusters: int = 20,
                       leaf_size: int = 5, split_over: int = 8,
                       seed: int = 0) -> dict:
    keywords = [r["keyword"] for r in rows]
    vectors = embed(keywords) if keywords else []
    counts = {"keywords": len(rows), "clusters": 0, "subclusters": 0,
              "fallback_descriptions": 0}

    def leaf(i: int) -> dict:
        return {"id": f"kw:{keywords[i]}", "keyword": keywords[i],
                "concept_id": rows[i]["concept_id"]}

    def cluster_node(node_id: str, idx: list[int], split: bool) -> dict:
        ordered = _by_centrality(idx, vectors)
        kws = [keywords[i] for i in ordered][:MAX_DESCRIBED]
        try:
            desc, source = describe(kws), "llm"
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError("empty description")
        except Exception:
            desc, source = fallback_description(kws), "fallback"
            counts["fallback_descriptions"] += 1
        if split and len(idx) > split_over:
            subs = _groups(idx, vectors, kmeans, math.ceil(len(idx) / leaf_size), seed)
            children = [cluster_node(f"{node_id}.{j}", g, False)
                        for j, g in enumerate(subs)]
            counts["subclusters"] += len(children)
        else:
            children = [leaf(i) for i in idx]
        return {"id": node_id, "description": desc,
                "description_source": source, "children": children}

    top = []
    if rows:
        groups = _groups(list(range(len(rows))), vectors, kmeans, clusters, seed)
        counts["clusters"] = len(groups)
        top = [cluster_node(f"c{i}", g, True) for i, g in enumerate(groups)]
    return {
        "meta": {"counts": counts, "clusters": clusters, "leaf_size": leaf_size,
                 "split_over": split_over, "seed": seed},
        "tree": {"id": ROOT_ID, "description": ROOT_DESCRIPTION,
                 "description_source": "fallback", "children": top},
    }


def write_tree(tree: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(json.dumps(tree, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_tree(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
