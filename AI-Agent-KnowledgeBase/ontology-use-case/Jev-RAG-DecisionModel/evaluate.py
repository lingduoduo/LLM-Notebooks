"""Run the 28 labeled test rooms through DecisionRAG and compare with baselines."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from taxonomy_corpus import ONTOLOGY_DIR, load_tree, tag_room

import utils  # noqa: E402  (ontology dir is on sys.path via taxonomy_corpus)
from evaluation.semantic_evaluation import evaluate, load_benchmark  # noqa: E402

MINILM_PREDICTIONS = ONTOLOGY_DIR / "evaluation" / "local_semantic_predictions.jsonl"


def prediction_row(case: dict, result, taxonomy_version: str, cost_usd: float = 0.0) -> dict:
    return {
        "case_id": case["id"],
        "taxonomy_version": taxonomy_version,
        "concept_ids": [r.id for r in result.ranked],
        "scores": {r.id: r.p_bin for r in result.ranked if r.source == "inferred"},
        "ranked": [asdict(r) for r in result.ranked],
        "trace": result.trace,
        "latency_s": result.latency_s,
        "cost_usd": cost_usd,
    }


def rank_metrics(cases: list[dict], rows: list[dict]) -> dict:
    by_id = {r["case_id"]: r for r in rows}
    hits = rr = n = 0
    for case in cases:
        gold = set(case["expected_concepts"])
        ids = by_id.get(case["id"], {}).get("concept_ids") or []
        if not gold or not ids:
            continue
        n += 1
        rank = next((i for i, c in enumerate(ids, 1) if c in gold), None)
        hits += rank == 1
        rr += 1 / rank if rank else 0.0
    return {"hit_at_1": hits / n if n else 0.0, "mrr": rr / n if n else 0.0, "count": n}


def _test_cases(benchmark: dict) -> list[dict]:
    return [c for c in benchmark["cases"] if c["split"] == "test"]


def run_decisionrag(judge, config, mode: str = "navigate") -> dict:
    taxonomy, root = load_tree()
    benchmark = load_benchmark()
    cases = _test_cases(benchmark)
    rows = []
    for case in cases:
        before = getattr(judge, "cost_usd", 0.0)
        res = tag_room(root, judge, config, case["subject"], case["tags"],
                       self_declared=case.get("self_declared_concepts", ()), mode=mode)
        cost = getattr(judge, "cost_usd", 0.0) - before if hasattr(judge, "cost_usd") else 0.0
        rows.append(prediction_row(case, res, taxonomy["version"], cost))
    return {
        "name": f"decisionrag-{mode}",
        "report": evaluate(benchmark, taxonomy, predictions=rows),
        "rank": rank_metrics(cases, rows),
        "rows": rows,
        "cost_usd": getattr(judge, "cost_usd", 0.0),
        "requests": getattr(judge, "requests", 0),
    }


def run_baselines() -> list[dict]:
    taxonomy = utils.load_taxonomy()
    benchmark = load_benchmark()
    cases = _test_cases(benchmark)
    keyword = {
        "name": "keyword",
        "report": evaluate(benchmark, taxonomy,
                           classifier=utils.RoomClassifier(taxonomy, utils.load()[1])),
        "rank": None, "rows": [], "cost_usd": 0.0, "requests": 0,
    }
    rows = []
    with MINILM_PREDICTIONS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                scores = row.get("scores", {})
                row["concept_ids"] = sorted(row["concept_ids"], key=lambda c: -scores.get(c, 0.0))
                rows.append(row)
    minilm = {
        "name": "minilm",
        "report": evaluate(benchmark, taxonomy, predictions=rows),
        "rank": rank_metrics(cases, rows),
        "rows": rows, "cost_usd": 0.0, "requests": 0,
    }
    return [keyword, minilm]


def write_report(runs: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    def mean_latency(rows):
        lat = [row["latency_s"] for row in rows if "latency_s" in row]
        return sum(lat) / len(lat) if lat else None

    lines = ["| name | micro-F1 | macro-F1 | exact match | negative FP rate | Hit@1 | MRR "
             "| latency (s/room) | nominal cost (¢) | requests |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        c, rk = r["report"]["concepts"], r["rank"]
        lines.append("| " + " | ".join([
            r["name"], f(c["micro_f1"]), f(c["macro_f1"]), f(c["exact_match"]),
            f(r["report"]["taxonomy_negative_false_positive_rate"]),
            f(rk and rk["hit_at_1"]), f(rk and rk["mrr"]), f(mean_latency(r.get("rows") or [])),
            f"{r['cost_usd'] * 100:.2f}", str(r["requests"])]) + " |")
    slices = sorted({s for r in runs for s in r["report"]["slices"]})
    lines += ["", "Per-slice concept micro-F1", "",
              "| slice | " + " | ".join(r["name"] for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for s in slices:
        cells = [f(r["report"]["slices"].get(s, {}).get("concepts", {}).get("micro_f1")) for r in runs]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "Nominal cost includes cached responses; real spend is the cache misses."]
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
