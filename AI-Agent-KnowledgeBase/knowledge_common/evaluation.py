"""Evaluation conventions shared by dense and sparse teaching CLIs.

Precision uses the number of returned hits, and RR uses the full ranking.
"""

from typing import Dict, List, Set


def metrics_for_query(retrieved: List[str], relevant: Set[str], k: int) -> Dict:
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
    return {
        "recall": recall,
        "precision": precision,
        "rr": rr,
        "hits": hits,
        "retrieved": topk,
    }
