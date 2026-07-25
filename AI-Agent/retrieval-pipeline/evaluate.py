"""Offline evaluation CLI for the hybrid retrieval pipeline.

This script runs the whole retrieval pipeline -- chunk -> embed -> retrieve ->
fuse -> rerank -- in a **single process that can work offline**, and compares
retrieval quality stage by stage on a small labelled evaluation set. It does
not depend on the dense/sparse microservices (ports 4240/4241/4242), so you can
reproduce the core result -- how the metrics improve as each stage is added --
with local models alone, no services running.

Local components used per stage:
  - sparse retrieval : BM25 (pure Python, rank_bm25, no model download needed)
  - dense retrieval  : a local sentence-embedding model (default
                       Qwen3-Embedding-0.6B, multilingual, loaded through
                       transformers; BGE-M3 and others work too)
  - fusion           : see fusion.py, either RRF or weighted normalization
  - reranking        : a cross-encoder (default
                       cross-encoder/ms-marco-MiniLM-L-6-v2)

Default behavior (no arguments): evaluate five configurations on the built-in
evaluation set -- BM25 / Dense / Hybrid-RRF / Hybrid-Weighted /
Hybrid-RRF+Rerank -- and print a Recall@k, MRR and nDCG@k comparison table.

Examples:
  python evaluate.py                         # built-in eval set, full table
  python evaluate.py --top-k 10 --rerank-top-k 5
  python evaluate.py --no-rerank             # skip the reranking stage
  python evaluate.py --embed-model BAAI/bge-m3 --pooling cls
  python evaluate.py --query "how to improve retrieval accuracy"  # single query, stage-by-stage trace
  python evaluate.py --corpus my_corpus.json --queries my_queries.json --output result.json
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fusion import fuse

# ---------------------------------------------------------------------------
# Built-in evaluation set: reuses the teaching test cases from test_client.py
# (four kinds -- semantic similarity / exact names / multilingual / technical
# codes). Their `expected` field is the hand-labelled set of relevant documents
# and serves as the gold standard. Two longer documents are added to exercise
# the chunking stage.
# ---------------------------------------------------------------------------
DEFAULT_CORPUS: List[Dict[str, Any]] = [
    # --- Near-duplicate code cluster (sparse wins, dense falls over) ---
    # The texts are nearly identical and differ only in the model code. Dense
    # vectors can barely tell members of one cluster apart, while sparse
    # retrieval nails it via exact term matching. The bigger the cluster, the
    # more likely dense picks the wrong one.
    {"doc_id": "xr_7001", "text": "Product model XR-7001 is a smartphone available now."},
    {"doc_id": "xr_7002", "text": "Product model XR-7002 is a smartphone available now."},
    {"doc_id": "xr_7003", "text": "Product model XR-7003 is a smartphone available now."},
    {"doc_id": "xr_7004", "text": "Product model XR-7004 is a smartphone available now."},
    {"doc_id": "xr_7005", "text": "Product model XR-7005 is a smartphone available now."},
    {"doc_id": "xr_7006", "text": "Product model XR-7006 is a smartphone available now."},
    # Near-duplicate HTTP status-code cluster (sparse wins, dense falls over)
    {"doc_id": "http_400", "text": "The HTTP-400 response is a client error status code."},
    {"doc_id": "http_401", "text": "The HTTP-401 response is a client error status code."},
    {"doc_id": "http_403", "text": "The HTTP-403 response is a client error status code."},
    {"doc_id": "http_404", "text": "The HTTP-404 response is a client error status code."},
    {"doc_id": "http_500", "text": "The HTTP-500 response is a server error status code."},
    # --- Paraphrase cluster (dense wins, sparse falls over) ---
    # Query and document share almost no words, so BM25 has nothing to match on
    # while dense retrieval hits it semantically.
    {"doc_id": "sem_readable", "text": "The language emphasizes clean, readable code that newcomers can pick up quickly."},
    {"doc_id": "sem_gc", "text": "Automatic memory management frees developers from manually releasing objects."},
    {"doc_id": "sem_photo", "text": "Green plants convert sunlight into chemical energy stored as sugars."},
    {"doc_id": "sem_crypto", "text": "Encryption scrambles a message so that only the intended recipient can read it."},
    # Longer documents on unrelated topics, used to exercise the chunking stage
    # (they get split into several chunks before retrieval)
    {"doc_id": "doc_watercycle", "text": (
        "The water cycle describes how water moves continuously between the ocean, the atmosphere and the land. "
        "Heat from the sun evaporates water from the sea surface into vapor that rises high into the sky. "
        "As the vapor cools it condenses into tiny droplets that gather to form clouds. "
        "When the droplets grow heavy enough they fall back to the ground as rain or snow, "
        "and rivers eventually carry that water back to the ocean, closing the loop."
    )},
    {"doc_id": "doc_volcano", "text": (
        "A volcano forms where molten rock called magma rises from deep inside the planet toward the surface. "
        "Magma collects in a chamber beneath the crust, and mounting pressure forces it upward through cracks. "
        "During an eruption the magma bursts out as lava, ash and gas, which pile up around the vent. "
        "Layer after layer of cooled lava slowly builds the cone-shaped mountain we recognize as a volcano."
    )},
]

DEFAULT_QUERIES: List[Dict[str, Any]] = [
    # Exact-code queries: sparse nails them, dense struggles to tell near-identical
    # model numbers apart (`expected` is the single correct answer)
    {"query": "XR-7003", "expected": ["xr_7003"]},
    {"query": "XR-7005", "expected": ["xr_7005"]},
    {"query": "HTTP-403", "expected": ["http_403"]},
    {"query": "HTTP-400", "expected": ["http_400"]},
    # Paraphrase queries: almost no words in common with the document, so dense
    # hits them semantically and sparse has nothing to match
    {"query": "a beginner friendly language with tidy syntax", "expected": ["sem_readable"]},
    {"query": "reclaiming unused heap space without programmer effort", "expected": ["sem_gc"]},
    {"query": "how vegetation turns light into food", "expected": ["sem_photo"]},
    {"query": "hiding a note so eavesdroppers cannot understand it", "expected": ["sem_crypto"]},
    # Long-document semantic queries: the matching long document is chunked
    # first, then recalled and reranked through one of its chunks
    {"query": "how does water move between the ocean and the sky", "expected": ["doc_watercycle"]},
    {"query": "how are volcanoes formed from molten rock", "expected": ["doc_volcano"]},
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a document into overlapping chunks using a character window.

    Short documents (length <= chunk_size) are returned unchanged as a single
    chunk. In a real system the chunk is the smallest unit of retrieval; a
    character-level sliding window keeps this implementation simple and
    language-agnostic.

    Args:
        text: Raw document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared by adjacent chunks.

    Returns:
        A list of chunk texts (at least one).
    """
    text = text.strip()
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks or [text]


# ---------------------------------------------------------------------------
# Tokenization (for BM25): keep words/numbers/hyphenated or underscored codes
# intact, and run CJK spans through jieba plus per-character fallback
# ---------------------------------------------------------------------------
# Second alternative matches a run of CJK characters (U+4E00-U+9FFF)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """Split text into BM25 terms.

    - Words, bare numbers and technical codes such as ``http-403`` /
      ``max_buffer_size`` / ``xr-7000`` are kept whole (hyphens and underscores
      are not split), which is what makes exact matching work.
    - A run of CJK characters yields both its jieba segmentation and its
      individual characters, which makes Chinese recall more robust.
    """
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        span = match.group()
        if "\u4e00" <= span[0] <= "\u9fff":  # CJK run
            try:
                import jieba
                tokens.extend(w for w in jieba.cut(span) if w.strip())
            except Exception:
                pass
            tokens.extend(list(span))
        else:
            tokens.append(span)
    return tokens


# ---------------------------------------------------------------------------
# Sparse retrieval: BM25
# ---------------------------------------------------------------------------
class BM25Retriever:
    """BM25 retriever built on rank_bm25, operating at chunk level."""

    def __init__(self, chunk_ids: List[str], chunk_texts: List[str]):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = chunk_ids
        self.tokenized = [tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Return (chunk_id, score) sorted by descending score, positives only."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda kv: kv[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Dense retrieval: a local sentence-embedding model (transformers)
# ---------------------------------------------------------------------------
class DenseEncoder:
    """Load a local sentence-embedding model via transformers for dense retrieval."""

    def __init__(self, model_name: str, pooling: str, device: str,
                 query_instruct: str = "", max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.pooling = self._resolve_pooling(pooling, model_name)
        # Instruction-style retrieval models (e.g. Qwen3-Embedding, last-token
        # pooling) expect a task instruction on the query side; models using
        # mean/cls pooling (MiniLM / BGE-M3) do not, so it is disabled for them.
        self.query_instruct = query_instruct if (query_instruct and self.pooling == "last") else ""
        # last-token pooling needs left padding so the final position lines up
        # with the real last token
        padding_side = "left" if self.pooling == "last" else "right"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _resolve_pooling(pooling: str, model_name: str) -> str:
        if pooling != "auto":
            return pooling
        name = model_name.lower()
        if "qwen" in name:
            return "last"
        if "bge-m3" in name or "bge-large" in name or "bge-base" in name:
            return "cls"
        return "mean"

    def _pool(self, last_hidden, attention_mask):
        torch = self.torch
        if self.pooling == "cls":
            return last_hidden[:, 0]
        if self.pooling == "last":
            return last_hidden[:, -1]
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 16):
        torch = self.torch
        if is_query and self.query_instruct:
            texts = [f"Instruct: {self.query_instruct}\nQuery:{t}" for t in texts]
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            pooled = self._forward(batch)
            # Some models produce NaN on the mps forward pass (transformers 5.x
            # with certain weights). On detection, fall back to CPU permanently
            # and recompute, so vectors stay finite and results reproducible.
            if self.device != "cpu" and torch.isnan(pooled).any():
                self.device = "cpu"
                self.model = self.model.to("cpu")
                pooled = self._forward(batch)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu())
        return torch.cat(vectors, dim=0)

    def _forward(self, batch: List[str]):
        torch = self.torch
        enc = self.tokenizer(
            batch, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        return self._pool(out.last_hidden_state, enc["attention_mask"])


class DenseRetriever:
    """Chunk-level retriever using cosine similarity over dense vectors."""

    def __init__(self, encoder: DenseEncoder, chunk_ids: List[str], chunk_texts: List[str]):
        self.encoder = encoder
        self.chunk_ids = chunk_ids
        self.matrix = encoder.encode(chunk_texts)  # [N, D], already normalized

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = self.encoder.encode([query], is_query=True)[0]
        sims = (self.matrix @ q).tolist()
        ranked = sorted(zip(self.chunk_ids, sims), key=lambda kv: kv[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Reranking: cross-encoder
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """Rerank candidates precisely with a cross-encoder.

    On transformers 5.x with some BERT weights the fp32 forward pass can produce
    NaN. When that is detected this class falls back to CPU + float64 and
    recomputes, so the output stays finite and reproducible.
    """

    def __init__(self, model_name: str, device: str, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        torch = self.torch
        if not docs:
            return []
        enc = self.tokenizer(
            [query] * len(docs), list(docs),
            padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits.squeeze(-1).float()
        if torch.isnan(logits).any():
            # Fallback: recompute on CPU in float64
            enc_cpu = {k: v.to("cpu") for k, v in enc.items()}
            model64 = self.model.to("cpu").double()
            with torch.no_grad():
                logits = model64(**enc_cpu).logits.squeeze(-1)
            self.model = self.model.to(self.device).float()
        return [float(x) for x in logits.reshape(-1).tolist()]

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int) -> List[Tuple[str, float]]:
        """candidates: [(doc_id, text)] -> [(doc_id, rerank_score)] descending, top_k."""
        scores = self.score(query, [text for _, text in candidates])
        ranked = sorted(
            ((doc_id, s) for (doc_id, _), s in zip(candidates, scores)),
            key=lambda kv: kv[1], reverse=True,
        )
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Chunk-level results -> document-level results (keep each document's best chunk)
# ---------------------------------------------------------------------------
def chunks_to_docs(ranked_chunks: List[Tuple[str, float]], chunk_to_doc: Dict[str, str]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for chunk_id, score in ranked_chunks:
        doc_id = chunk_to_doc[chunk_id]
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked_ids: List[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & set(gold)) / len(gold)


def reciprocal_rank(ranked_ids: List[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_ids: List[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics(per_query_ranked: List[Tuple[List[str], Sequence[str]]], k: int) -> Dict[str, float]:
    n = len(per_query_ranked)
    if n == 0:
        return {"recall@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}
    recall = sum(recall_at_k(r, g, k) for r, g in per_query_ranked) / n
    mrr = sum(reciprocal_rank(r, g) for r, g in per_query_ranked) / n
    ndcg = sum(ndcg_at_k(r, g, k) for r, g in per_query_ranked) / n
    return {"recall@k": recall, "mrr": mrr, "ndcg@k": ndcg}


# ---------------------------------------------------------------------------
# Pipeline: produce a document-level ranking per method for one query
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, corpus, args):
        self.args = args
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_to_doc: Dict[str, str] = {}
        self.doc_text: Dict[str, str] = {}

        # Chunking
        for doc in corpus:
            self.doc_text[doc["doc_id"]] = doc["text"]
            chunks = chunk_text(doc["text"], args.chunk_size, args.chunk_overlap)
            for i, chunk in enumerate(chunks):
                cid = f"{doc['doc_id']}::c{i}" if len(chunks) > 1 else doc["doc_id"]
                self.chunk_ids.append(cid)
                self.chunk_texts.append(chunk)
                self.chunk_to_doc[cid] = doc["doc_id"]

        self.n_docs = len(corpus)
        self.n_chunks = len(self.chunk_ids)

        # Sparse index
        self.bm25 = BM25Retriever(self.chunk_ids, self.chunk_texts)

        # Dense index (optional)
        self.dense: Optional[DenseRetriever] = None
        if args.use_dense:
            encoder = DenseEncoder(args.embed_model, args.pooling, args.device,
                                   query_instruct=args.query_instruct)
            self.dense = DenseRetriever(encoder, self.chunk_ids, self.chunk_texts)

        # Reranker (optional)
        self.reranker: Optional[CrossEncoderReranker] = None
        if args.use_rerank:
            self.reranker = CrossEncoderReranker(args.reranker_model, args.device)

    def run_query(self, query: str) -> Dict[str, List[Tuple[str, float]]]:
        """Return the doc-level ranking per method: {method: [(doc_id, score)]}."""
        top_k = self.args.top_k
        sparse_chunks = self.bm25.search(query, top_k)
        sparse_docs = chunks_to_docs(sparse_chunks, self.chunk_to_doc)

        out: Dict[str, List[Tuple[str, float]]] = {"sparse": sparse_docs}

        if self.dense is not None:
            dense_chunks = self.dense.search(query, top_k)
            dense_docs = chunks_to_docs(dense_chunks, self.chunk_to_doc)
            out["dense"] = dense_docs

            ranked_lists = {"dense": dense_docs, "sparse": sparse_docs}
            weights = {"dense": self.args.dense_weight, "sparse": self.args.sparse_weight}
            rrf = fuse(ranked_lists, method="rrf", k=self.args.k_rrf, weights=weights)
            weighted = fuse(ranked_lists, method="weighted", weights=weights)
            out["rrf"] = rrf
            out["weighted"] = weighted

            if self.reranker is not None:
                # Precisely rerank the top-N pool from the RRF fusion
                pool = [doc_id for doc_id, _ in rrf[: self.args.rerank_pool]]
                candidates = [(doc_id, self.doc_text[doc_id]) for doc_id in pool]
                reranked = self.reranker.rerank(query, candidates, self.args.rerank_top_k)
                out["rerank"] = reranked

        return out


# ---------------------------------------------------------------------------
# Output: comparison table / single-query trace
# ---------------------------------------------------------------------------
METHOD_LABELS = [
    ("sparse", "BM25 (sparse)"),
    ("dense", "Dense"),
    ("rrf", "Hybrid-RRF"),
    ("weighted", "Hybrid-Weighted"),
    ("rerank", "Hybrid-RRF+Rerank"),
]


def run_evaluation(pipeline: Pipeline, queries, args) -> Dict[str, Any]:
    k = args.eval_k
    per_method: Dict[str, List[Tuple[List[str], Sequence[str]]]] = {m: [] for m, _ in METHOD_LABELS}
    per_query_records = []

    t0 = time.time()
    for spec in queries:
        query = spec["query"]
        gold = spec.get("expected", [])
        results = pipeline.run_query(query)
        record = {"query": query, "expected": gold, "methods": {}}
        for method, _ in METHOD_LABELS:
            if method not in results:
                continue
            ranked_ids = [doc_id for doc_id, _ in results[method]]
            per_method[method].append((ranked_ids, gold))
            record["methods"][method] = {
                "top": [{"doc_id": d, "score": round(s, 4)} for d, s in results[method][:5]],
                "recall@k": round(recall_at_k(ranked_ids, gold, k), 4),
                "mrr": round(reciprocal_rank(ranked_ids, gold), 4),
                "ndcg@k": round(ndcg_at_k(ranked_ids, gold, k), 4),
            }
        per_query_records.append(record)
    elapsed = time.time() - t0

    summary = {}
    for method, _ in METHOD_LABELS:
        if per_method[method]:
            summary[method] = aggregate_metrics(per_method[method], k)

    return {
        "summary": summary,
        "per_query": per_query_records,
        "elapsed_sec": round(elapsed, 2),
        "eval_k": k,
    }


def print_table(report: Dict[str, Any], pipeline: Pipeline, args) -> None:
    k = report["eval_k"]
    print("=" * 78)
    print("Hybrid retrieval pipeline - stage-by-stage evaluation")
    print("=" * 78)
    print(f"Corpus: {pipeline.n_docs} documents -> {pipeline.n_chunks} chunks "
          f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Queries: {len(report['per_query'])}   "
          f"Dense model: {args.embed_model if args.use_dense else '(disabled)'}   "
          f"Reranker: {args.reranker_model if args.use_rerank else '(disabled)'}")
    print(f"Retrieval top_k={args.top_k}  fusion k(RRF)={args.k_rrf}  "
          f"rerank pool={args.rerank_pool}  metric cutoff k={k}  device={args.device}")
    print(f"Elapsed: {report['elapsed_sec']}s")
    print("-" * 78)
    header = f"{'Stage / Method':<22}{'Recall@'+str(k):>12}{'MRR':>12}{'nDCG@'+str(k):>12}"
    print(header)
    print("-" * 78)
    for method, label in METHOD_LABELS:
        if method not in report["summary"]:
            continue
        m = report["summary"][method]
        print(f"{label:<22}{m['recall@k']:>12.4f}{m['mrr']:>12.4f}{m['ndcg@k']:>12.4f}")
    print("-" * 78)
    print("How to read this: going down the rows adds the dense retrieval / fusion / "
          "reranking stages in turn -- watch how the metrics move.")
    print("=" * 78)


def print_per_query(report: Dict[str, Any]) -> None:
    """Print each method's MRR per query, showing where a single route fails and
    fusion picks up the slack."""
    methods = [m for m, _ in METHOD_LABELS]
    short = {"sparse": "BM25", "dense": "Dense", "rrf": "RRF",
             "weighted": "Wgt", "rerank": "Rerank"}
    print("\nPer-query MRR (1.00 = correct document ranked first; shows which route "
          "fails on which kind of query)")
    print("-" * 78)
    header = f"{'Query':<42}" + "".join(f"{short[m]:>7}" for m in methods)
    print(header)
    print("-" * 78)
    for rec in report["per_query"]:
        cells = ""
        for m in methods:
            if m in rec["methods"]:
                cells += f"{rec['methods'][m]['mrr']:>7.2f}"
            else:
                cells += f"{'-':>7}"
        q = rec["query"]
        q = q if len(q) <= 41 else q[:38] + "..."
        print(f"{q:<42}{cells}")
    print("=" * 78)


def print_query_trace(pipeline: Pipeline, query: str, args) -> None:
    results = pipeline.run_query(query)
    print("=" * 78)
    print(f"Single-query stage-by-stage ranking trace   query = {query!r}")
    print(f"Corpus: {pipeline.n_docs} documents -> {pipeline.n_chunks} chunks   "
          f"device={args.device}")
    print("=" * 78)
    for method, label in METHOD_LABELS:
        if method not in results:
            continue
        print(f"\n[{label}]")
        for rank, (doc_id, score) in enumerate(results[method][:5], start=1):
            snippet = pipeline.doc_text.get(doc_id, "")[:60].replace("\n", " ")
            print(f"  {rank}. {doc_id:<14} score={score:8.4f}  {snippet}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline evaluation CLI for the hybrid retrieval pipeline "
                    "(chunk -> embed -> retrieve -> fuse -> rerank, compared stage by stage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py                       # built-in eval set, full table\n"
            "  python evaluate.py --no-rerank           # skip the reranking stage\n"
            "  python evaluate.py --no-dense            # BM25 only (fully offline, no model)\n"
            "  python evaluate.py --query 'how to improve retrieval accuracy'  # single-query trace\n"
            "  python evaluate.py --embed-model BAAI/bge-m3 --pooling cls\n"
            "  python evaluate.py --output result.json  # also write results to JSON\n"
        ),
    )
    data = parser.add_argument_group("Data")
    data.add_argument("--corpus", help="Corpus JSON file, format [{'doc_id','text'}...]; "
                                       "defaults to the built-in corpus")
    data.add_argument("--queries", help="Queries JSON file, format [{'query','expected':[...]}...]; "
                                        "defaults to the built-in queries")
    data.add_argument("--query", help="Single-query mode: trace the stage-by-stage ranking for this "
                                      "query only, skipping evaluation")
    data.add_argument("--limit-queries", type=int, default=0,
                      help="Evaluate only the first N queries (0 = all)")

    stages = parser.add_argument_group("Pipeline stages")
    stages.add_argument("--no-dense", dest="use_dense", action="store_false",
                        help="Disable dense retrieval (also disables fusion and reranking; "
                             "degrades to plain BM25, fully offline with no model needed)")
    stages.add_argument("--no-rerank", dest="use_rerank", action="store_false",
                        help="Disable the neural reranking stage")
    stages.set_defaults(use_dense=True, use_rerank=True)

    chunk = parser.add_argument_group("Chunking")
    chunk.add_argument("--chunk-size", type=int, default=280,
                       help="Maximum characters per chunk (default 280)")
    chunk.add_argument("--chunk-overlap", type=int, default=40,
                       help="Characters shared by adjacent chunks (default 40)")

    retr = parser.add_argument_group("Retrieval and fusion")
    retr.add_argument("--top-k", type=int, default=10,
                      help="Candidates recalled by each retrieval route (default 10)")
    retr.add_argument("--k-rrf", type=int, default=60, help="RRF smoothing constant k (default 60)")
    retr.add_argument("--dense-weight", type=float, default=1.0,
                      help="Weight of the dense route during fusion (default 1.0)")
    retr.add_argument("--sparse-weight", type=float, default=1.0,
                      help="Weight of the sparse route during fusion (default 1.0)")

    rer = parser.add_argument_group("Reranking")
    rer.add_argument("--rerank-pool", type=int, default=10,
                     help="Size of the candidate pool fed to the reranker "
                          "(top-N of the RRF fusion, default 10)")
    rer.add_argument("--rerank-top-k", type=int, default=10,
                     help="Number of results returned after reranking (default 10)")

    model = parser.add_argument_group("Models")
    model.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Dense sentence-embedding model (default "
                            "sentence-transformers/all-MiniLM-L6-v2, about 90MB, mostly English; "
                            "for multilingual corpora use Qwen/Qwen3-Embedding-0.6B or BAAI/bge-m3)")
    model.add_argument("--pooling", default="auto", choices=["auto", "mean", "cls", "last"],
                       help="Sentence-embedding pooling strategy (auto picks by model name: "
                            "qwen -> last, bge-m3 -> cls, everything else -> mean)")
    model.add_argument("--query-instruct",
                       default="Given a search query, retrieve relevant passages that answer the query",
                       help="Query-side task instruction for instruction-style retrieval models "
                            "(only applies to last-token pooling models such as Qwen3-Embedding)")
    model.add_argument("--reranker-model", default="BAAI/bge-reranker-base",
                       help="Cross-encoder reranking model (default BAAI/bge-reranker-base, "
                            "multilingual, about 1.1GB on first run; for production consider the "
                            "stronger BAAI/bge-reranker-v2-m3, or the lighter "
                            "cross-encoder/ms-marco-MiniLM-L-6-v2)")
    model.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Inference device (default auto)")

    out = parser.add_argument_group("Evaluation and output")
    out.add_argument("--eval-k", type=int, default=3,
                     help="Metric cutoff position k (Recall@k / nDCG@k, default 3)")
    out.add_argument("--no-per-query", dest="show_per_query", action="store_false",
                     help="Do not print the per-query MRR matrix")
    out.set_defaults(show_per_query=True)
    out.add_argument("--output", help="Write the full results (including per-query detail) "
                                      "to this JSON file")
    out.add_argument("--offline", action="store_true",
                     help="Set HF_HUB_OFFLINE=1 to force using locally cached models only")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.device = detect_device(args.device)
    # Single-query trace mode does not change use_dense/use_rerank semantics, but
    # reranking depends on the dense fusion pool
    if not args.use_dense:
        args.use_rerank = False

    corpus = load_json(args.corpus) if args.corpus else DEFAULT_CORPUS
    queries = load_json(args.queries) if args.queries else DEFAULT_QUERIES

    try:
        pipeline = Pipeline(corpus, args)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Pipeline initialization failed: {exc}", file=sys.stderr)
        print("Hint: the dense and reranking stages need local sentence-embedding and "
              "cross-encoder models. Use --no-dense to degrade to plain BM25 (fully "
              "offline), or point --embed-model at a model you already have cached.",
              file=sys.stderr)
        return 1

    if args.query:
        print_query_trace(pipeline, args.query, args)
        return 0

    if args.limit_queries > 0:
        queries = queries[: args.limit_queries]

    report = run_evaluation(pipeline, queries, args)
    print_table(report, pipeline, args)
    if args.show_per_query:
        print_per_query(report)

    if args.output:
        payload = {
            "config": {
                "embed_model": args.embed_model if args.use_dense else None,
                "reranker_model": args.reranker_model if args.use_rerank else None,
                "top_k": args.top_k, "k_rrf": args.k_rrf, "eval_k": args.eval_k,
                "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
                "device": args.device,
            },
            **report,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
