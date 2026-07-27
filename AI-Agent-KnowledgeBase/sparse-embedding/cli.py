#!/usr/bin/env python3
"""
Sparse retrieval command line tool (experiment 3-5)

Runs BM25 sparse retrieval over a small sample corpus, supporting:
  - custom corpus / query / top-k / output file
  - --explain to reproduce the book's per-term "IDF / TF / BM25 contribution" log
  - --eval to compute recall@k / precision@k / MRR on a small labelled set
  - --method splade for learned sparse retrieval (needs a model download; prints
    a clear hint in offline environments)

Run with no arguments, it is equivalent to the default demo of experiment 3-5 in
the book (query "model distillation").
"""

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional, Set, Tuple

from bm25_engine import SparseSearchEngine


# ---------------------------------------------------------------------------
# Built-in sample corpus and labels (English, matching what the engine's tokenizer
# handles, and fully reproducible offline).
# The corpus deliberately mixes ordinary words, product codes, technical acronyms
# and documents that only paraphrase the query, so it shows both where BM25 excels
# (exact keyword matching) and where it falls short (synonyms).
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
]

# query -> set of relevant doc_ids (hand-labelled ground truth)
DEFAULT_LABELS: Dict[str, List[str]] = {
    "model distillation": ["doc_3", "doc_4"],
    "HTTP 404 error": ["doc_6"],
    "XK9-2B4-7Q1": ["doc_9"],
    "BM25 ranking function": ["doc_5"],
    # The relevant documents say kitten / feline rather than the literal word "cat",
    # which demonstrates the blind spot of sparse retrieval: BM25 cannot read
    # synonyms, so it misses them entirely.
    "cat": ["doc_7", "doc_8"],
}

DEFAULT_QUERY = "model distillation"


def _quiet_logging(verbose: bool) -> None:
    """Keep engine logs quiet by default; --verbose / --explain open them up to
    DEBUG so the calculation is visible."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    logging.getLogger("bm25_engine").setLevel(level)


class InputError(Exception):
    """A bad --corpus / --labels file, reported to the user without a traceback."""


CORPUS_FORMAT_HINT = (
    'Expected .json  : [{"doc_id": "doc_1", "title": "...", "text": "..."}, ...]\n'
    '                  (an object with a "documents" key holding that list also works)\n'
    'Expected .jsonl : one such object per line'
)

LABELS_FORMAT_HINT = (
    'Expected .json  : {"model distillation": ["doc_3", "doc_4"], "HTTP 404 error": ["doc_6"]}\n'
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


def build_engine(corpus: List[Dict], k1: float, b: float) -> SparseSearchEngine:
    """Feed the corpus into the engine and rebuild BM25 with the given k1/b."""
    engine = SparseSearchEngine()
    engine.index_batch([
        {"text": d["text"],
         "doc_id": d.get("doc_id"),
         "metadata": {"title": d.get("title", "")}}
        for d in corpus
    ])
    # index_batch rebuilds BM25 after every document, so pin it once more here
    # with the parameters we actually want
    from bm25_engine import BM25
    engine.bm25 = BM25(engine.index, k1=k1, b=b)
    return engine


def explain_result(engine: SparseSearchEngine, query: str, doc_id: str) -> List[Tuple[str, int, int, float, float]]:
    """Reproduce the book's log: for a matched document, report TF / document
    length / IDF / BM25 contribution for each query term."""
    from bm25_engine import TextProcessor
    internal = engine.external_to_internal[doc_id]
    terms = TextProcessor().tokenize(query)
    rows = []
    for term in terms:
        tf = engine.index.term_frequency[internal].get(term, 0)
        if tf == 0:
            continue
        dl = engine.index.doc_lengths[internal]
        idf = engine.bm25.calculate_idf(term)
        contrib = engine.bm25.calculate_term_score(term, internal)
        rows.append((term, tf, dl, idf, contrib))
    return rows


def run_search(engine: SparseSearchEngine, query: str, top_k: int,
               explain: bool) -> List[Dict]:
    """Run one query, print the results, and return them structured for --output."""
    results = engine.search(query, top_k=top_k)
    print(f"\nQuery: '{query}'  (BM25, top-{top_k})")
    print("-" * 60)
    if not results:
        print("  No documents matched (none of the query terms are in the inverted index).")
        return []
    out = []
    for rank, r in enumerate(results, 1):
        title = r["metadata"].get("title", "")
        print(f"  #{rank}  {r['doc_id']}  score={r['score']:.4f}  {title}")
        print(f"       matched terms: {r['debug']['matched_terms']}")
        print(f"       preview: {r['text'][:80]}...")
        if explain:
            rows = explain_result(engine, query, r["doc_id"])
            for term, tf, dl, idf, contrib in rows:
                print(f"         └ '{term}': TF={tf}, doc length={dl} terms, "
                      f"IDF={idf:.4f}, BM25 contribution={contrib:.4f}")
        out.append({
            "rank": rank,
            "doc_id": r["doc_id"],
            "score": r["score"],
            "title": title,
            "matched_terms": r["debug"]["matched_terms"],
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


def run_eval(engine: SparseSearchEngine, labels: Dict[str, List[str]],
             k: int) -> Dict:
    """Evaluate retrieval on the labelled set, printing per-query metrics plus
    the macro average."""
    print(f"\n{'='*60}")
    print(f"Retrieval quality evaluation (recall@{k} / precision@{k} / MRR)")
    print(f"{'='*60}")
    per_query = {}
    sum_recall = sum_prec = sum_rr = 0.0
    for query, rel_list in labels.items():
        relevant = set(rel_list)
        results = engine.search(query, top_k=max(k, 10))
        retrieved = [r["doc_id"] for r in results]
        m = _metrics_for_query(retrieved, relevant, k)
        per_query[query] = m
        sum_recall += m["recall"]
        sum_prec += m["precision"]
        sum_rr += m["rr"]
        flag = ("" if m["recall"] > 0
                else "   <- missed (synonym blind spot)" if query == "cat"
                else "   <- missed")
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
    print(f"\n{'-'*60}")
    print(f"Macro average  recall@{k}={macro['recall@k']:.3f}  "
          f"precision@{k}={macro['precision@k']:.3f}  "
          f"MRR={macro['mrr']:.3f}  miss rate(1-recall@{k})={macro['miss_rate@k']:.3f}")
    return {"k": k, "per_query": {q: {kk: vv for kk, vv in m.items() if kk != "retrieved"}
                                  for q, m in per_query.items()},
            "macro": macro}


def run_splade(query: str, corpus: List[Dict], top_k: int) -> Optional[List[Dict]]:
    """Learned sparse retrieval (SPLADE). Needs transformers + torch + a pretrained model.

    When an offline environment cannot download the model, print a clear hint and
    return None (so argument parsing can still be validated).
    """
    model_name = "naver/splade-cocondenser-ensembledistil"
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception as e:
        print("\n[SPLADE] Requires transformers and torch, missing here:", e)
        print("        Install with: pip install torch transformers")
        print("        (the BM25 path needs no model and runs fully offline)")
        return None
    try:
        # Load from the local cache only, so an offline environment does not hang
        # on an endless download.
        print(f"\n[SPLADE] Trying to load model {model_name} from the local cache ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
        model.eval()
    except Exception:
        print(f"\n[SPLADE] Model {model_name} is not in the local cache, and the "
              f"weights cannot be downloaded offline.")
        print("        Run this once while online to cache the model locally, then retry:")
        print(f"          huggingface-cli download {model_name}")
        print("        (the BM25 path depends on no model and reproduces experiment 3-5 "
              "fully offline)")
        return None

    import torch

    def encode(text: str) -> Dict[str, float]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq, vocab]
        # SPLADE: log(1+ReLU(logits)) then max-pool over the sequence dimension,
        # giving sparse weights over the vocabulary
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values.squeeze(0)
        nz = torch.nonzero(weights).squeeze(-1)
        return {int(i): float(weights[i]) for i in nz}

    q_vec = encode(query)
    scored = []
    for d in corpus:
        d_vec = encode(d["text"])
        score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
        scored.append((d.get("doc_id"), score, d.get("title", "")))
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"\nQuery: '{query}'  (SPLADE, top-{top_k})")
    print("-" * 60)
    out = []
    for rank, (doc_id, score, title) in enumerate(scored[:top_k], 1):
        print(f"  #{rank}  {doc_id}  score={score:.4f}  {title}")
        out.append({"rank": rank, "doc_id": doc_id, "score": score, "title": title})
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Sparse retrieval command line tool (experiment 3-5): run BM25 / SPLADE "
                    "sparse retrieval over a small corpus and evaluate retrieval quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python cli.py                                  # default demo (query "model distillation")
  python cli.py -q "HTTP 404 error" --explain    # show per-term TF/IDF/BM25 contributions
  python cli.py --eval                           # recall/precision/MRR on the labelled set
  python cli.py -q "cat"                         # watch BM25's synonym blind spot
  python cli.py --corpus my.json -q "..." -o out.json
  python cli.py --method splade -q "model distillation"   # learned sparse retrieval (needs a model)
""",
    )
    parser.add_argument("-q", "--query", default=DEFAULT_QUERY,
                        help=f"Query string (default: '{DEFAULT_QUERY}')")
    parser.add_argument("-c", "--corpus", default=None,
                        help="Path to a corpus file (.json array of documents, or .jsonl "
                             "one per line); defaults to the built-in sample corpus")
    parser.add_argument("-m", "--method", choices=["bm25", "splade"], default="bm25",
                        help="Retrieval method: bm25 (default, offline) or splade "
                             "(learned sparse, requires a model download)")
    parser.add_argument("-k", "--top-k", type=int, default=5,
                        help="Return the top k results (default: 5)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write results / evaluation metrics to this file as JSON")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate recall@k / precision@k / MRR on the labelled set "
                             "instead of running a single query")
    parser.add_argument("--labels", default=None,
                        help="Evaluation label file {query: [relevant doc_id, ...]}; "
                             "defaults to the built-in labels")
    parser.add_argument("--explain", action="store_true",
                        help="Show per-term TF/IDF/BM25 contributions for every matched "
                             "document (reproduces the book's log)")
    parser.add_argument("--k1", type=float, default=1.5,
                        help="BM25 term-frequency saturation parameter k1 (default: 1.5)")
    parser.add_argument("-b", "--b", type=float, default=0.75,
                        help="BM25 document-length normalization parameter b (default: 0.75)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Turn on engine DEBUG logs (shows tokenization, inverted index "
                             "construction and the full scoring process)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _quiet_logging(args.verbose)

    if args.top_k < 1:
        print(f"\n[input error] --top-k must be >= 1, got {args.top_k}", file=sys.stderr)
        return 2

    # Read every input file up front, before building the index or loading a model,
    # so a mistyped path fails immediately with a clear message.
    try:
        corpus = load_corpus(args.corpus)
        labels = load_labels(args.labels) if args.eval else None
    except InputError as e:
        print(f"\n[input error] {e}", file=sys.stderr)
        return 2

    print(f"Loaded corpus: {len(corpus)} documents"
          + (" (built-in sample)" if not args.corpus else f" (from {args.corpus})"))

    payload: Dict = {"method": args.method, "query": args.query, "top_k": args.top_k}

    if args.method == "splade":
        results = run_splade(args.query, corpus, args.top_k)
        if results is None:
            return 0  # The missing-model hint was already printed; exit normally
        payload["results"] = results
    else:
        engine = build_engine(corpus, k1=args.k1, b=args.b)
        print(f"BM25 parameters: k1={args.k1}, b={args.b}, avgdl={engine.bm25.avgdl:.2f}")
        if args.eval:
            payload["eval"] = run_eval(engine, labels, args.top_k)
        else:
            payload["results"] = run_search(engine, args.query, args.top_k, args.explain)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
