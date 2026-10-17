"""Synthesize a partial summary.json from a results.jsonl for cell 14 to read.

Used when a multi-run is still in-flight and we want to surface partial real
data in the cross-model chart immediately. Aggregates results.jsonl with
llm_eval.summarize.public_export (the same code the runner uses) and writes
to <real_root>/<label>/summary.json.

Usage:
  python3 scripts/patch_summary.py <run_id> <label>
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 3:
        print("usage: patch_summary.py <run_id> <label> [runner_dir_name]")
        return 1
    run_id = sys.argv[1]
    label = sys.argv[2]
    # runner_dir_name defaults to label (with dots replaced by underscores for slug).
    # Accept both "real-" and "cell12-" prefixes for run_id.
    if not run_id.startswith(("cell12-", "real-")):
        run_id = f"cell12-{label}"
    runner_dir_name = sys.argv[3] if len(sys.argv) > 3 else label.replace(".", "_")
    real_root = PROJ / "runs" / "notebook-demo" / "real"
    src_results = real_root / "_tmp" / run_id / runner_dir_name / "results.jsonl"
    if not src_results.exists():
        print(f"missing {src_results}")
        return 1
    target = real_root / label
    target.mkdir(parents=True, exist_ok=True)
    tgt_results = target / "results.jsonl"
    src_text = src_results.read_text()
    if not tgt_results.exists() or tgt_results.stat().st_size == 0:
        tgt_results.write_text(src_text)

    # Same aggregation as llm_eval.summarize, so partial and final summaries agree.
    sys.path.insert(0, str(PROJ))
    from llm_eval.adapters import OPENAI_MODELS
    from llm_eval.summarize import load_results, public_export
    from llm_eval.task import load_tasks
    tasks = load_tasks(str(PROJ / "tasks" / "public_all.jsonl"))

    # Map label → (price_in_per_m, price_out_per_m) — same source the runner uses.
    PRICE_BY_LABEL = {
        model: (p["price_in_per_m"], p["price_out_per_m"])
        for model, p in OPENAI_MODELS.items()
    }
    price_in, price_out = PRICE_BY_LABEL.get(label, (None, None))

    results = load_results(str(src_results))
    # Backfill cost for records written before preset prices were applied.
    if price_in is not None and price_out is not None:
        for r in results:
            if r.get("status") in ("ok", "invalid") and not r.get("cost_usd"):
                usage = r.get("usage") or {}
                ti = int(usage.get("prompt_tokens") or 0)
                to = int(usage.get("completion_tokens") or 0)
                r["cost_usd"] = (ti * price_in + to * price_out) / 1e6

    summary = public_export(
        results,
        tasks,
        run_meta={"run_id": run_id, "runner": label, "adapter": label, "model": label},
    )
    summary["partial"] = True
    summary["patched_at"] = time.time()
    counts = {k: sum(1 for r in results if r.get("status") == k)
              for k in ("ok", "invalid", "error", "unattempted")}
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"patched {label}: {counts} acc={summary.get('accuracy')} brier={summary.get('brier')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
