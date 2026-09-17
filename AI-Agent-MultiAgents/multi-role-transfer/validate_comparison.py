#!/usr/bin/env python3
"""Independently verify the retained Experiment 10-1 comparison package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from provenance import (
    check_artifacts,
    check_runtime_sources,
    load_declared_drift,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?",
                   default=ROOT / "validation" / "comparison" / "runs" / "exp10-1-qwen35flash-20260809-v2")
    return p.parse_args()


def main(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads((run_dir / "acceptance.json").read_text(encoding="utf-8"))
    campaign = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
    judge = json.loads((run_dir / "judge.json").read_text(encoding="utf-8"))
    assert manifest["acceptance"]["evidence_status"] == "pass"
    assert acceptance["evidence_status"] == "pass"
    assert acceptance["passed_gates"] == acceptance["total_gates"]
    assert all(acceptance["gates"].values())
    assert not check_artifacts(manifest, run_dir), "artifact hash mismatch"
    # Runtime sources must still match the manifest unless the change is
    # declared in validation/source_drift.json; see provenance.py.
    run_key = run_dir.relative_to(ROOT).as_posix()
    sources = check_runtime_sources(manifest, ROOT, load_declared_drift(ROOT, run_key))
    assert not sources["missing"], f"recorded source missing: {sources['missing']}"
    assert not sources["undeclared_drift"], (
        f"undeclared runtime source drift: {sources['undeclared_drift']}"
    )
    assert not sources["stale_declarations"], (
        f"source_drift.json claims these changed but they match: "
        f"{sources['stale_declarations']}"
    )
    runs = campaign["runs"]
    assert campaign["paired_samples"] == 30
    assert len(runs) == 60
    assert len(campaign["boundary_runs"]) == 12
    assert all(run.get("provider_receipts") for run in runs)
    assert all(run.get("loaded_skills") for run in runs if run["path"] == "skill")
    assert sum(run["outcome"]["pass"] for run in runs if run["path"] == "skill") == 15
    assert sum(run["outcome"]["pass"] for run in runs if run["path"] == "transfer") == 2
    assert judge["paired_n"] == 30
    assert judge["judge_receipt_count"] == 60
    assert judge["unique_response_ids"] == 60
    assert judge["parse_complete"] is True
    assert all(pair["parse_complete"] for pair in judge["pairs"])
    blob = b"\n".join(path.read_bytes() for path in run_dir.iterdir() if path.is_file())
    assert not re.search(rb'(?i)bearer\s+[a-z0-9._~+/=-]{16,}', blob)
    assert not re.search(rb'(?i)"(?:api[_-]?key|authorization)"\s*:\s*"(?!<redacted>|null|\s*")[^"]+"', blob)
    print(f"validated {manifest['run_id']}: "
          f"{acceptance['passed_gates']}/{acceptance['total_gates']} gates, "
          f"{len(sources['matching'])} pinned sources match, "
          f"{len(sources['declared_drift'])} declared as changed since the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
