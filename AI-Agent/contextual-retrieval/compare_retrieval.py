#!/usr/bin/env python3
"""Contextual retrieval comparison benchmark (experiment 3-11).

This script quantifies how much contextual retrieval improves recall over
traditional chunking, using a controlled comparison: the same chunks are
indexed into BM25 two different ways --

  * plain      : index only the raw chunk (metadata.original_text)
  * contextual : index the LLM-generated prefix plus the raw chunk (the
                 `content` field)

Both are then evaluated on the same query set and compared by recall@k (did the
relevant chunk appear in the top k results?). This is the core claim of
Anthropic's "Contextual Retrieval": prepending a context prefix to each chunk
improves recall for BM25 (sparse) and vector (dense) retrieval alike.

BM25 retrieval is fully offline and needs no API or search service. The
embedding / hybrid methods do call an embedding API (see --method).

Usage examples:
  python compare_retrieval.py                       # comparison table on the default eval set
  python compare_retrieval.py --query "What are the powers of the President?"  # one query, side by side
  python compare_retrieval.py --mode plain          # the no-context baseline only
  python compare_retrieval.py --output result.json  # also save machine-readable results

The example query above is Chinese because the bundled corpus is Chinese law;
pass any query that matches whichever corpus you indexed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import jieba
    if hasattr(jieba, "setLogLevel"):
        jieba.setLogLevel(60)  # silence jieba's loading logs
    _HAS_JIEBA = True
except Exception:  # pragma: no cover - jieba normally ships with requirements
    _HAS_JIEBA = False


# ---------------------------------------------------------------------------
# Tokenization: Chinese has no spaces, so a plain .split() turns a whole passage
# into a single token and BM25 stops working entirely. jieba is used by default;
# --no-jieba falls back to character bigrams, which also runs fully offline.
# ---------------------------------------------------------------------------
def tokenize(text: str, use_jieba: bool = True) -> List[str]:
    """Split text into a list of tokens for BM25."""
    text = (text or "").lower()
    if use_jieba and _HAS_JIEBA:
        return [t for t in jieba.cut(text) if t.strip()]
    # Fallback: Chinese character bigrams plus runs of ASCII words
    tokens: List[str] = []
    buf = ""
    chars = list(text)
    for ch in chars:
        if ch.isascii() and (ch.isalnum()):
            buf += ch
            continue
        if buf:
            tokens.append(buf)
            buf = ""
        if not ch.isspace():
            tokens.append(ch)
    if buf:
        tokens.append(buf)
    # Append Chinese bigrams for finer matching granularity
    cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]  # CJK ideograph range
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus(path: str) -> List[Dict]:
    """Load chunks from document_store.json as [{chunk_id, contextual, plain, context}].

    Each chunk's `content` field is "context prefix + original text", while
    metadata.original_text is the raw text without context -- exactly the pair
    needed to compare the two indexing strategies.
    """
    with open(path, "r", encoding="utf-8") as f:
        store = json.load(f)

    chunks: List[Dict] = []
    for chunk_id, entry in store.items():
        if "_chunk_" not in chunk_id:
            continue  # skip whole-document entries
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata", {}) or {}
        contextual_text = entry.get("content", "") or ""
        plain_text = meta.get("original_text") or contextual_text
        # Context prefix = the contextual text with the trailing original_text removed
        context = contextual_text
        if plain_text and contextual_text.endswith(plain_text):
            context = contextual_text[: len(contextual_text) - len(plain_text)].strip()
        chunks.append({
            "chunk_id": chunk_id,
            "contextual": contextual_text,
            "plain": plain_text,
            "context": context,
        })
    return chunks


def load_eval(path: str) -> List[Dict]:
    """Load the evaluation set as [{id, query, gold_chunk_id, ...}]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", data if isinstance(data, list) else [])


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------
class BM25Retriever:
    """A simple retriever that builds a BM25 index over one text field."""

    def __init__(self, chunks: List[Dict], field: str, use_jieba: bool = True):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.use_jieba = use_jieba
        corpus_tokens = [tokenize(c[field], use_jieba) for c in chunks]
        self.index = BM25Okapi(corpus_tokens)

    def rank(self, query: str) -> List[str]:
        """Return chunk_ids ordered by relevance, most relevant first."""
        scores = self.index.get_scores(tokenize(query, self.use_jieba))
        order = np.argsort(scores)[::-1]
        return [self.chunk_ids[i] for i in order]

    def scored(self, query: str, top_k: int) -> List[Dict]:
        """Return the top_k results with their scores."""
        scores = self.index.get_scores(tokenize(query, self.use_jieba))
        order = np.argsort(scores)[::-1][:top_k]
        return [{"chunk_id": self.chunk_ids[i], "score": float(scores[i])} for i in order]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def recall_at_k(retriever: BM25Retriever, queries: List[Dict], ks: List[int]) -> Dict:
    """Compute recall@k (hit rate) for a set of queries at each k."""
    per_query = []
    hits = {k: 0 for k in ks}
    for q in queries:
        ranking = retriever.rank(q["query"])
        gold = q["gold_chunk_id"]
        rank_pos = ranking.index(gold) + 1 if gold in ranking else None
        row = {"id": q.get("id"), "query": q["query"], "gold": gold, "rank": rank_pos}
        for k in ks:
            hit = rank_pos is not None and rank_pos <= k
            row[f"hit@{k}"] = hit
            if hit:
                hits[k] += 1
        per_query.append(row)
    n = len(queries)
    recall = {k: (hits[k] / n if n else 0.0) for k in ks}
    return {"recall": recall, "per_query": per_query, "n": n}


def print_comparison_table(plain: Optional[Dict], contextual: Optional[Dict], ks: List[int]):
    """Print the recall@k comparison table."""
    print("\n" + "=" * 68)
    print("Retrieval recall: plain chunks  vs.  contextual retrieval (BM25)")
    print("=" * 68)
    header = "  k  | " + " | ".join(f"recall@{k:<3}" for k in ks)
    # Print one row per method
    col_w = 12
    line = f"{'Method':<16}" + "".join(f"recall@{k}".rjust(col_w) for k in ks)
    print(line)
    print("-" * len(line))
    if plain:
        print(f"{'plain':<16}" + "".join(f"{plain['recall'][k]*100:>10.1f}%" for k in ks))
    if contextual:
        print(f"{'contextual':<16}" + "".join(f"{contextual['recall'][k]*100:>10.1f}%" for k in ks))
    if plain and contextual:
        print("-" * len(line))
        deltas = []
        for k in ks:
            d = (contextual["recall"][k] - plain["recall"][k]) * 100
            deltas.append(f"{d:>+9.1f}pp")
        print(f"{'gain (pp)':<16}" + "".join(s.rjust(col_w) for s in deltas))
        # Reduction in retrieval failure rate (the book's "1 - recall@k" measure)
        print("-" * len(line))
        fails = []
        for k in ks:
            p_fail = 1 - plain["recall"][k]
            c_fail = 1 - contextual["recall"][k]
            if p_fail > 0:
                red = (p_fail - c_fail) / p_fail * 100
                fails.append(f"{red:>9.0f}%")
            else:
                fails.append(f"{'-':>10}")
        print(f"{'failure drop':<16}" + "".join(s.rjust(col_w) for s in fails))
    print("=" * 68)


def print_per_query(result: Dict, label: str):
    print(f"\n[{label}] per-query hit rank (rank = position of the gold chunk; - = not recalled)")
    for row in result["per_query"]:
        print(f"  {row['id']}  rank={str(row['rank']):>3}  gold={row['gold']:<28} {row['query'][:32]}")


# ---------------------------------------------------------------------------
# Side-by-side comparison for a single query
# ---------------------------------------------------------------------------
def single_query_compare(chunks: List[Dict], query: str, top_k: int, use_jieba: bool,
                         mode: str):
    id2chunk = {c["chunk_id"]: c for c in chunks}

    def show(field_label, field):
        retr = BM25Retriever(chunks, field, use_jieba)
        print(f"\n[{field_label}] Top-{top_k}")
        print("-" * 60)
        for i, r in enumerate(retr.scored(query, top_k), 1):
            c = id2chunk[r["chunk_id"]]
            snippet = c["plain"].replace("<!-- FORCE BREAK -->", "").replace("\n", " ").strip()[:48]
            ctx = c["context"].replace("\n", " ").strip()[:40]
            print(f"  {i}. score={r['score']:6.2f}  {r['chunk_id']}")
            if field == "contextual" and ctx:
                print(f"       context prefix: {ctx}")
            print(f"       original: {snippet}")

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    if mode in ("plain", "both"):
        show("plain", "plain")
    if mode in ("contextual", "both"):
        show("contextual", "contextual")


# ---------------------------------------------------------------------------
# Optional: embedding / hybrid retrieval (requires an API)
# ---------------------------------------------------------------------------
def embedding_unavailable_notice(method: str):
    print(f"\n[note] --method {method} calls an embedding API (dense vectors) and "
          f"cannot run offline.")
    print("       Set OPENAI_API_KEY / SILICONFLOW_API_KEY etc. in .env and use the")
    print("       embedding/hybrid retrieval in contextual_tools.ContextualKnowledgeBaseTools.")
    print("       The default --method bm25 already reproduces the book's headline result:")
    print("       that context prefixes improve BM25 recall.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Contextual retrieval benchmark: quantify how much a context prefix "
                    "improves retrieval recall@k (experiment 3-11)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python compare_retrieval.py\n"
               "  python compare_retrieval.py --query \"What are the powers of the President?\" --top-k 5\n"
               "  python compare_retrieval.py --mode both --k 1 3 5 --output result.json",
    )
    p.add_argument("--corpus", default="document_store.json",
                   help="Corpus file (chunk store with `content` and metadata.original_text); "
                        "default document_store.json")
    p.add_argument("--eval", dest="eval_path", default="evaluation/retrieval_eval.json",
                   help="Evaluation set (query + gold_chunk_id); default evaluation/retrieval_eval.json")
    p.add_argument("--query", default=None,
                   help="Ad-hoc single query: show plain vs contextual top-K results "
                        "side by side (skips the full evaluation set)")
    p.add_argument("--mode", choices=["plain", "contextual", "both"], default="both",
                   help="Which index to compare: plain, contextual, or both (default)")
    p.add_argument("--method", choices=["bm25", "embedding", "hybrid"], default="bm25",
                   help="Retrieval method: bm25 (offline, default); embedding/hybrid need "
                        "an embedding API")
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5],
                   help="k values to evaluate for recall@k (default: 1 3 5)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Results shown per method in --query mode (default: 5)")
    p.add_argument("--model", default=None,
                   help="Embedding model name (only used with --method embedding/hybrid)")
    p.add_argument("--no-jieba", action="store_true",
                   help="Disable jieba tokenization and use character bigrams instead "
                        "(removes the jieba dependency)")
    p.add_argument("--output", default=None,
                   help="Write machine-readable evaluation results to this JSON file")
    p.add_argument("--per-query", action="store_true",
                   help="Also print the per-query hit-rank detail")
    return p


def main():
    args = build_arg_parser().parse_args()
    use_jieba = not args.no_jieba

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[error] Corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    chunks = load_corpus(str(corpus_path))
    if not chunks:
        print(f"[error] No usable chunks in the corpus (no *_chunk_* entries): {corpus_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(chunks)} chunks | tokenizer: "
          f"{'jieba' if (use_jieba and _HAS_JIEBA) else 'char-bigram'} "
          f"| method: {args.method}")

    if args.method in ("embedding", "hybrid"):
        embedding_unavailable_notice(args.method)
        # Continue with BM25 so there is still a runnable offline result
        print("       Falling back to BM25 for an offline comparison below.\n")

    # Single-query mode
    if args.query:
        single_query_compare(chunks, args.query, args.top_k, use_jieba, args.mode)
        return

    # Evaluation-set mode
    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        print(f"[error] Evaluation set not found: {eval_path}", file=sys.stderr)
        sys.exit(1)
    queries = load_eval(str(eval_path))
    ks = sorted(set(args.k))

    plain_res = contextual_res = None
    if args.mode in ("plain", "both"):
        plain_res = recall_at_k(BM25Retriever(chunks, "plain", use_jieba), queries, ks)
    if args.mode in ("contextual", "both"):
        contextual_res = recall_at_k(BM25Retriever(chunks, "contextual", use_jieba), queries, ks)

    print(f"Evaluation set: {eval_path}  ({len(queries)} queries)")
    print_comparison_table(plain_res, contextual_res, ks)

    if args.per_query:
        if plain_res:
            print_per_query(plain_res, "plain")
        if contextual_res:
            print_per_query(contextual_res, "contextual")

    if args.output:
        out = {
            "corpus": str(corpus_path),
            "eval": str(eval_path),
            "num_chunks": len(chunks),
            "num_queries": len(queries),
            "tokenizer": "jieba" if (use_jieba and _HAS_JIEBA) else "char-bigram",
            "k": ks,
            "plain": plain_res,
            "contextual": contextual_res,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
