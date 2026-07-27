#!/usr/bin/env python3
"""
Dense retrieval command line tool (experiment 3-4)

Runs dense embedding retrieval over a small sample corpus, supporting:
  - custom corpus / query / top-k / output file
  - --eval: compute recall@k / precision@k / MRR on a small labelled evaluation
            set, showing off the core selling point of dense embeddings --
            that they understand paraphrases
  - --compare-ann: reproduce the focus of experiment 3-4 in the book -- compare
            the ANNOY and HNSW backends on recall relative to exact brute-force
            search, index build time and query latency (reusing the server-side
            indexing.py)
  - --embedding-model: switch embedding models; defaults to BAAI/bge-m3, and
            offline you can use a cached sentence-transformers/all-MiniLM-L6-v2

Run with no arguments, it is equivalent to the default demo of experiment 3-4
in the book (query "a cat playing"). --compare-ann uses synthetic vectors and
needs no model at all, so the ANN comparison reproduces fully offline.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np

from indexing import AnnoyIndex, HNSWIndex


# ---------------------------------------------------------------------------
# Built-in sample corpus and labels (English, matching what common sentence
# embedding models handle well, and fully reproducible offline).
# The corpus deliberately includes paraphrase documents (kitten / feline for cat,
# two phrasings of distillation) to show where dense retrieval shines at semantic
# matching -- exactly the cases sparse BM25 (experiment 3-5) misses.
# ---------------------------------------------------------------------------
DEFAULT_CORPUS: List[Dict] = [
    {"doc_id": "doc_1", "title": "Python Language",
     "text": "Python is a high-level programming language known for readability and a simple syntax."},
    {"doc_id": "doc_2", "title": "JavaScript Runtime",
     "text": "JavaScript runs in the browser and on servers via Node.js for full-stack web development."},
    {"doc_id": "doc_3", "title": "Model Distillation",
     "text": "Model distillation compresses a large teacher model into a smaller student model while preserving accuracy."},
    {"doc_id": "doc_4", "title": "Knowledge Distillation",
     "text": "Knowledge distillation transfers knowledge from a big neural network to a compact model for efficient inference."},
    {"doc_id": "doc_5", "title": "BM25 Ranking",
     "text": "BM25 is a probabilistic ranking function using term frequency and inverse document frequency."},
    {"doc_id": "doc_6", "title": "HTTP Errors",
     "text": "The HTTP 404 error code means the requested resource was not found on the web server."},
    {"doc_id": "doc_7", "title": "A Playful Kitten",
     "text": "A cute kitten chased a ball of yarn across the living room floor all afternoon."},
    {"doc_id": "doc_8", "title": "Silent Hunter",
     "text": "The feline predator stalked its prey silently through the tall grass at dusk."},
    {"doc_id": "doc_9", "title": "Hardware Fault",
     "text": "Error code XK9-2B4-7Q1 indicates a hardware fault in the storage controller board."},
    {"doc_id": "doc_10", "title": "Transformers",
     "text": "Transformer models use self-attention to process input sequences in parallel efficiently."},
    {"doc_id": "doc_11", "title": "Deep Learning",
     "text": "Deep learning stacks many layers of neurons to extract hierarchical features from raw data."},
    {"doc_id": "doc_12", "title": "Gradient Descent",
     "text": "Gradient descent minimizes a loss function by iteratively updating the model parameters."},
]

# query -> set of relevant doc_ids (hand-labelled ground truth)
# Most of these queries share no literal keywords with their relevant documents
# and are related only semantically -- which is exactly what tests dense retrieval.
DEFAULT_LABELS: Dict[str, List[str]] = {
    # Neither kitten nor feline contains the literal word "cat": dense retrieval
    # should recall them semantically, while sparse BM25 misses them
    "a cat playing": ["doc_7", "doc_8"],
    # Two phrasings of distillation, semantically the same topic
    "model distillation": ["doc_3", "doc_4"],
    # Semantically related, without the literal phrase "neural network training"
    "training neural networks": ["doc_11", "doc_12"],
    "self attention in sequence models": ["doc_10"],
    "web server resource not found": ["doc_6"],
}

DEFAULT_QUERY = "a cat playing"

DEFAULT_MODEL = "BAAI/bge-m3"
OFFLINE_HINT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Dense embedding encoder: compute sentence vectors directly with transformers'
# AutoModel (mean / cls pooling + L2 normalization). This loads both the book's
# default BAAI/bge-m3 (the bge family uses cls pooling) and a locally cached
# sentence-transformers/all-MiniLM-L6-v2 (mean pooling), with no dependency on
# FlagEmbedding.
# ---------------------------------------------------------------------------
class DenseEncoder:
    def __init__(self, model_name: str, pooling: str = "auto",
                 device: str = "cpu", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        self.max_length = max_length
        if pooling == "auto":
            # bge / bge-m3 take the [CLS] vector; most sentence-transformers
            # models use mean pooling
            pooling = "cls" if "bge" in model_name.lower() else "mean"
        self.pooling = pooling

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**enc)
            if self.pooling == "cls":
                emb = out.last_hidden_state[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = self.torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.cpu().numpy().astype("float32"))
        return np.vstack(vecs)


def load_encoder(model_name: str, pooling: str, device: str) -> Optional["DenseEncoder"]:
    """Load the dense encoder.

    When running offline with no cached model, print a clear hint and return None
    (so argument parsing can still be validated)."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401
    except Exception as e:
        print("\n[dense encoding] Requires transformers and torch, missing here:", e)
        print("        Install with: pip install torch transformers")
        print("        (--compare-ann uses synthetic vectors, needs no model, "
              "and runs fully offline)")
        return None
    try:
        print(f"Loading embedding model {model_name} (pooling={pooling}, device={device})...")
        t0 = time.time()
        encoder = DenseEncoder(model_name, pooling=pooling, device=device)
        print(f"Model loaded in {time.time() - t0:.1f}s, pooling={encoder.pooling}")
        return encoder
    except Exception as e:
        print(f"\n[dense encoding] Could not load model {model_name}: {e}")
        print(f"        Offline environments cannot download the {model_name} weights "
              f"(BGE-M3 is about 2.3GB).")
        print(f"        Use a smaller cached model instead: --embedding-model {OFFLINE_HINT_MODEL}")
        print("        Or pre-cache the target model while online; --compare-ann needs "
              "no model at all.")
        return None


class InputError(Exception):
    """A bad --corpus / --labels file, reported to the user without a traceback."""


CORPUS_FORMAT_HINT = (
    'Expected .json  : [{"doc_id": "doc_1", "title": "...", "text": "..."}, ...]\n'
    '                  (an object with a "documents" key holding that list also works)\n'
    'Expected .jsonl : one such object per line'
)

LABELS_FORMAT_HINT = (
    'Expected .json  : {"a cat playing": ["doc_7", "doc_8"], "model distillation": ["doc_3"]}\n'
    '                  i.e. query -> list of relevant doc_ids'
)


def load_corpus(path: Optional[str]) -> List[Dict]:
    """Load the corpus. Supports .json (array of documents) and .jsonl (one per line)."""
    if not path:
        return DEFAULT_CORPUS
    docs: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            docs.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise InputError(
                                f"Corpus file '{path}' has invalid JSON on line {lineno}: {e}\n"
                                f"{CORPUS_FORMAT_HINT}") from None
            else:
                data = json.load(f)
                docs = data["documents"] if isinstance(data, dict) else data
    except FileNotFoundError:
        raise InputError(
            f"Corpus file not found: '{path}'\n"
            f"Omit --corpus to use the built-in sample corpus "
            f"({len(DEFAULT_CORPUS)} documents).\n"
            f"{CORPUS_FORMAT_HINT}") from None
    except json.JSONDecodeError as e:
        raise InputError(f"Corpus file '{path}' is not valid JSON: {e}\n"
                         f"{CORPUS_FORMAT_HINT}") from None
    except (KeyError, TypeError):
        raise InputError(f"Unexpected structure in corpus file '{path}'.\n"
                         f"{CORPUS_FORMAT_HINT}") from None

    if not docs:
        raise InputError(f"Corpus file is empty: '{path}'\n{CORPUS_FORMAT_HINT}")
    missing = [i for i, d in enumerate(docs) if not isinstance(d, dict) or "text" not in d]
    if missing:
        raise InputError(
            f"Corpus file '{path}': {len(missing)} document(s) have no 'text' field "
            f"(first at index {missing[0]}).\n{CORPUS_FORMAT_HINT}")
    return docs


def load_labels(path: Optional[str]) -> Dict[str, List[str]]:
    """Load evaluation labels: {query: [relevant_doc_id, ...]}."""
    if not path:
        return DEFAULT_LABELS
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except FileNotFoundError:
        raise InputError(
            f"Labels file not found: '{path}'\n"
            f"Omit --labels to use the built-in labels ({len(DEFAULT_LABELS)} queries).\n"
            f"{LABELS_FORMAT_HINT}") from None
    except json.JSONDecodeError as e:
        raise InputError(f"Labels file '{path}' is not valid JSON: {e}\n"
                         f"{LABELS_FORMAT_HINT}") from None
    if not isinstance(labels, dict) or not labels:
        raise InputError(f"Labels file '{path}' must be a non-empty query -> doc_ids object.\n"
                         f"{LABELS_FORMAT_HINT}")
    return labels


# ---------------------------------------------------------------------------
# Dense retrieval (exact brute force, used for single queries and quality evaluation)
# ---------------------------------------------------------------------------
def dense_rank(query_vec: np.ndarray, doc_matrix: np.ndarray) -> List[int]:
    """Vectors are L2-normalized, so cosine similarity is the dot product.

    Returns the document indices ordered by descending similarity."""
    sims = doc_matrix @ query_vec
    return list(np.argsort(-sims)), sims


def run_search(encoder: "DenseEncoder", corpus: List[Dict], doc_matrix: np.ndarray,
               query: str, top_k: int) -> List[Dict]:
    """Run one dense query, print the results, and return them structured for --output."""
    q = encoder.encode([query])[0]
    order, sims = dense_rank(q, doc_matrix)
    print(f"\nQuery: '{query}'  (dense retrieval, top-{top_k})")
    print("-" * 60)
    out = []
    for rank, idx in enumerate(order[:top_k], 1):
        d = corpus[idx]
        title = d.get("title", "")
        print(f"  #{rank}  {d.get('doc_id')}  cos={float(sims[idx]):.4f}  {title}")
        print(f"       preview: {d['text'][:80]}...")
        out.append({
            "rank": rank,
            "doc_id": d.get("doc_id"),
            "score": float(sims[idx]),
            "title": title,
        })
    return out


def _metrics_for_query(retrieved: List[str], relevant: Set[str], k: int) -> Dict:
    """recall@k / precision@k / hit rank (for MRR) of a single query."""
    topk = retrieved[:k]
    hits = [d for d in topk if d in relevant]
    recall = len(set(hits)) / len(relevant) if relevant else 0.0
    precision = len(hits) / len(topk) if topk else 0.0
    rr = 0.0
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            rr = 1.0 / i
            break
    return {"recall": recall, "precision": precision, "rr": rr,
            "hits": hits, "retrieved": topk}


def run_eval(encoder: "DenseEncoder", corpus: List[Dict], doc_matrix: np.ndarray,
             labels: Dict[str, List[str]], k: int) -> Dict:
    """Evaluate dense retrieval on the labelled set, printing per-query metrics
    plus the macro average."""
    doc_ids = [d.get("doc_id") for d in corpus]
    print(f"\n{'=' * 60}")
    print(f"Dense retrieval quality evaluation (recall@{k} / precision@{k} / MRR)")
    print(f"{'=' * 60}")
    per_query = {}
    sum_recall = sum_prec = sum_rr = 0.0
    q_vecs = encoder.encode(list(labels.keys()))
    for (query, rel_list), qv in zip(labels.items(), q_vecs):
        relevant = set(rel_list)
        order, _ = dense_rank(qv, doc_matrix)
        retrieved = [doc_ids[i] for i in order]
        m = _metrics_for_query(retrieved, relevant, k)
        per_query[query] = m
        sum_recall += m["recall"]
        sum_prec += m["precision"]
        sum_rr += m["rr"]
        flag = "" if m["recall"] > 0 else "   <- missed"
        print(f"\nQuery '{query}'  relevant={sorted(relevant)}")
        print(f"  Retrieved order: {retrieved[:k]}")
        print(f"  recall@{k}={m['recall']:.2f}  precision@{k}={m['precision']:.2f}  RR={m['rr']:.2f}{flag}")
    n = len(labels)
    macro = {
        "recall@k": sum_recall / n,
        "precision@k": sum_prec / n,
        "mrr": sum_rr / n,
        "miss_rate@k": 1.0 - sum_recall / n,
    }
    print(f"\n{'-' * 60}")
    print(f"Macro average  recall@{k}={macro['recall@k']:.3f}  "
          f"precision@{k}={macro['precision@k']:.3f}  "
          f"MRR={macro['mrr']:.3f}  miss rate(1-recall@{k})={macro['miss_rate@k']:.3f}")
    return {"k": k, "per_query": {q: {kk: vv for kk, vv in m.items() if kk != "retrieved"}
                                  for q, m in per_query.items()},
            "macro": macro}


# ---------------------------------------------------------------------------
# ANN backend comparison (the focus of experiment 3-4): reuse the ANNOY / HNSW
# implementations from the server-side indexing.py and compare them on a batch of
# synthetic unit vectors -- recall relative to exact brute-force search, index
# build time and query latency.
# Synthetic vectors rather than real text embeddings are used so that (a) it runs
# fully offline with no model download, and (b) the corpus is large enough for the
# "approximate" in ANN to actually diverge from exact search, which is what makes
# the trade-off between the two algorithms visible.
# ---------------------------------------------------------------------------
def _exact_topk(queries: np.ndarray, base: np.ndarray, k: int) -> List[Set[int]]:
    """Exact brute-force nearest neighbors (cosine), the ground truth for ANN recall."""
    sims = queries @ base.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return [set(row.tolist()) for row in idx]


def _sanity_ok(index, base: np.ndarray) -> bool:
    """Self-check: querying with a vector already in the index should return it.

    Used to spot an index backend that is broken in this environment."""
    probe = min(5, len(base))
    for i in range(probe):
        ids, _ = index.search(base[i], min(10, len(base)))
        if f"v{i}" not in set(ids):
            return False
    return True


def compare_ann(base: np.ndarray, queries: np.ndarray, top_k: int, backends: List[str],
                annoy_n_trees: int, hnsw_M: int, hnsw_ef_search: int,
                hnsw_ef_construction: int) -> Dict:
    dim = base.shape[1]
    n = len(base)
    exact_sets = _exact_topk(queries, base, top_k)

    print(f"\n{'=' * 60}")
    print(f"ANN backend comparison: {n} vectors of {dim} dims, {len(queries)} queries, top-{top_k}")
    print(f"Metrics: recall@{top_k} relative to exact brute force / build time / mean query latency")
    print(f"{'=' * 60}")

    report: Dict[str, Dict] = {}
    for backend in backends:
        if backend == "annoy":
            index = AnnoyIndex(dimension=dim, n_trees=annoy_n_trees,
                               metric="angular", logger=None)
        else:
            index = HNSWIndex(dimension=dim, max_elements=n + 16,
                              ef_construction=hnsw_ef_construction, M=hnsw_M,
                              ef_search=hnsw_ef_search, space="cosine", logger=None)

        t0 = time.time()
        for i, v in enumerate(base):
            index.add_item(f"v{i}", v)
        if backend == "annoy":
            index.rebuild_index()
        build_time = time.time() - t0

        healthy = _sanity_ok(index, base)

        recalls: List[float] = []
        qtimes: List[float] = []
        for qi, q in enumerate(queries):
            ts = time.time()
            ids, _ = index.search(q, top_k)
            qtimes.append(time.time() - ts)
            got = {int(d[1:]) for d in ids}
            recalls.append(len(got & exact_sets[qi]) / top_k)

        mean_recall = float(np.mean(recalls))
        mean_qms = float(np.mean(qtimes) * 1000)
        params = (f"n_trees={annoy_n_trees}" if backend == "annoy"
                  else f"M={hnsw_M}, ef_search={hnsw_ef_search}, ef_construction={hnsw_ef_construction}")
        report[backend] = {
            "recall@k": mean_recall,
            "build_time_s": build_time,
            "mean_query_ms": mean_qms,
            "params": params,
            "healthy": healthy,
        }
        warn = ("" if healthy else
                "  [warning] This backend cannot even recall its own vectors, so it "
                "looks broken in this environment; the numbers below are not trustworthy")
        print(f"\n[{backend.upper()}]  {params}{warn}")
        print(f"  recall@{top_k} = {mean_recall:.3f}")
        print(f"  build time        = {build_time * 1000:.1f} ms")
        print(f"  mean query latency = {mean_qms:.3f} ms")

    if "annoy" in report and "hnsw" in report:
        print(f"\n{'-' * 60}")
        print("Summary: the HNSW graph usually gives higher recall and supports incremental")
        print("         inserts, at the cost of more memory and a slower build; the ANNOY")
        print("         tree builds fast and is memory-lean, but deletion requires a rebuild")
        print("         and recall is tuned via n_trees.")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Dense retrieval command line tool (experiment 3-4): run dense embedding "
                    "retrieval over a small corpus, evaluate retrieval quality, and compare "
                    "the ANNOY and HNSW ANN index backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py                                       # default demo (query "a cat playing", needs an embedding model)
  python cli.py -q "model distillation" -k 3          # a single dense query
  python cli.py --eval                                # recall/precision/MRR on the labelled set
  python cli.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --eval   # small offline model
  python cli.py --compare-ann                         # ANNOY vs HNSW recall (synthetic vectors, no model needed)
  python cli.py --compare-ann --ann-base 5000 --annoy-n-trees 5 -k 10 -o ann.json
""",
    )
    parser.add_argument("-q", "--query", default=DEFAULT_QUERY,
                        help=f"Query string (default: '{DEFAULT_QUERY}')")
    parser.add_argument("-c", "--corpus", default=None,
                        help="Path to a corpus file (.json array of documents, or .jsonl "
                             "one per line); defaults to the built-in sample corpus")
    parser.add_argument("-k", "--top-k", type=int, default=5,
                        help="Return the top k results (default: 5)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write results / evaluation metrics to this file as JSON")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL,
                        help=f"Dense embedding model name (default: {DEFAULT_MODEL}); "
                             f"offline you can use a cached {OFFLINE_HINT_MODEL}")
    parser.add_argument("--pooling", choices=["auto", "mean", "cls"], default="auto",
                        help="Sentence-vector pooling: auto (cls for bge*, mean otherwise) / mean / cls")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (cpu / cuda / mps, default: cpu)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate recall@k / precision@k / MRR on the labelled set "
                             "instead of running a single query")
    parser.add_argument("--labels", default=None,
                        help="Evaluation label file {query: [relevant doc_id, ...]}; "
                             "defaults to the built-in labels")

    ann = parser.add_argument_group("ANN backend comparison (--compare-ann)")
    ann.add_argument("--compare-ann", action="store_true",
                     help="Compare ANNOY and HNSW on recall and timing (reuses indexing.py, "
                          "synthetic vectors, no model needed)")
    ann.add_argument("--backend", choices=["annoy", "hnsw", "both"], default="both",
                     help="Which ANN backends to compare (default: both)")
    ann.add_argument("--ann-base", type=int, default=3000,
                     help="Number of synthetic base vectors (default: 3000; the larger it is, "
                          "the more visible the ANN approximation error)")
    ann.add_argument("--ann-queries", type=int, default=100,
                     help="Number of synthetic query vectors (default: 100)")
    ann.add_argument("--ann-dim", type=int, default=128,
                     help="Dimensionality of the synthetic vectors (default: 128)")
    ann.add_argument("--annoy-n-trees", type=int, default=10,
                     help="Number of ANNOY trees (default: 10; more is more accurate but slower)")
    ann.add_argument("--hnsw-M", type=int, default=16,
                     help="HNSW connections per node, M (default: 16; larger means higher "
                          "recall and more memory)")
    ann.add_argument("--hnsw-ef-search", type=int, default=20,
                     help="HNSW dynamic candidate list size at query time, ef_search (default: 20)")
    ann.add_argument("--hnsw-ef-construction", type=int, default=100,
                     help="HNSW dynamic candidate list size at build time, ef_construction "
                          "(default: 100)")
    ann.add_argument("--seed", type=int, default=42,
                     help="Random seed for the synthetic vectors (default: 42)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload: Dict = {"top_k": args.top_k}

    # --- ANN backend comparison: synthetic vectors, no embedding model, fully offline ---
    if args.compare_ann:
        rng = np.random.default_rng(args.seed)
        base = rng.standard_normal((args.ann_base, args.ann_dim)).astype("float32")
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        queries = rng.standard_normal((args.ann_queries, args.ann_dim)).astype("float32")
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        backends = ["annoy", "hnsw"] if args.backend == "both" else [args.backend]
        payload["compare_ann"] = compare_ann(
            base, queries, args.top_k, backends,
            annoy_n_trees=args.annoy_n_trees, hnsw_M=args.hnsw_M,
            hnsw_ef_search=args.hnsw_ef_search, hnsw_ef_construction=args.hnsw_ef_construction)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nResults written to {args.output}")
        return 0

    # --- Dense retrieval / evaluation: needs an embedding model ---
    # Read every input file up front: loading the model can pull gigabytes, and
    # failing on a mistyped path after that wastes the download.
    try:
        corpus = load_corpus(args.corpus)
        labels = load_labels(args.labels) if args.eval else None
    except InputError as e:
        print(f"\n[input error] {e}", file=sys.stderr)
        return 2

    print(f"Loaded corpus: {len(corpus)} documents"
          + (" (built-in sample)" if not args.corpus else f" (from {args.corpus})"))

    encoder = load_encoder(args.embedding_model, args.pooling, args.device)
    if encoder is None:
        return 0  # The missing-model hint was already printed; exit normally

    doc_matrix = encoder.encode([d["text"] for d in corpus])
    payload["embedding_model"] = args.embedding_model
    payload["query"] = args.query

    if args.eval:
        payload["eval"] = run_eval(encoder, corpus, doc_matrix, labels, args.top_k)
    else:
        payload["results"] = run_search(encoder, corpus, doc_matrix, args.query, args.top_k)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
