"""Cross-runner comparison.

Loads every per-runner summary.json under runs/<run_id>/ and produces a single
table that aligns accuracy, latency, cost and other metrics side by side.
"""
from __future__ import annotations

import json
from pathlib import Path


# Per-runner headline fields surfaced in the comparison table.
_KEY_FIELDS = (
    "n_scorable",
    "accuracy",
    "majority_class_accuracy",
    "schema_validity",
    "schema_validity_strict",
    "brier",
    "ece",
    "ordinal_mae",
    "p50_s",
    "p95_s",
    "cost_usd_total",
)


def _accuracy_map(breakdown: dict | None) -> dict:
    """{group: {"n", "accuracy"}} -> {group: rounded accuracy}."""
    out = {}
    for k, v in (breakdown or {}).items():
        acc = v.get("accuracy") if isinstance(v, dict) else None
        out[k] = round(acc, 4) if isinstance(acc, float) else acc
    return out


def compare_run(run_dir: str) -> dict:
    """Walk every <run_dir>/<runner>/summary.json and assemble a table."""
    base = Path(run_dir)
    manifest_path = base / "_manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: dict[str, dict] = {}
    # Per-family / per-topic accuracy come from each runner's summary.json,
    # so the breakdown always agrees with the headline accuracy beside it.
    per_family_compare: dict[str, dict] = {}
    per_topic_compare: dict[str, dict] = {}
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = s.get("run_meta", {})
        row = {
            "adapter": meta.get("adapter", ""),
            "model": meta.get("model", ""),
        }
        for k in _KEY_FIELDS:
            v = s.get(k)
            if isinstance(v, float):
                row[k] = round(v, 4)
            else:
                row[k] = v
        rows[child.name] = row
        per_family_compare[child.name] = _accuracy_map(s.get("per_family"))
        per_topic_compare[child.name] = _accuracy_map(s.get("per_topic"))

    return {
        "run_id": manifest.get("run_id", base.name),
        "n_tasks": manifest.get("n_tasks"),
        "dataset_hash": manifest.get("dataset_hash"),
        "manifest": manifest,
        "rows": rows,
        "per_family": per_family_compare,
        "per_topic": per_topic_compare,
    }


def write_markdown_table(compare: dict, path: str) -> None:
    """Optional: render a simple markdown table for the headline row."""
    rows = compare.get("rows", {})
    if not rows:
        return
    cols = ["runner", "model", "accuracy", "n_scorable", "brier", "ece", "p50_s", "p95_s", "cost_usd_total"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, row in rows.items():
        vals = [name]
        for c in cols[1:]:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.4f}" if c not in ("p50_s", "p95_s") else f"{v:.2f}s")
            else:
                vals.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(vals) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")