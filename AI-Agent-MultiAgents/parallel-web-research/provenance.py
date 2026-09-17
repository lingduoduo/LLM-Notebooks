"""Provenance checks for the retained Experiment 10-4 evidence bundles.

A manifest records the SHA-256 of each runtime source *as it was when the run
executed*.  Once the sources legitimately change the recorded hashes can never
match again, so asserting equality is an assertion that must fail forever.
Deleting the check instead would leave the bundle unpinned.

The middle ground implemented here: a run's own artifacts must still recompute
exactly, and every runtime source must either still match its recorded hash or
be listed in ``validation/source_drift.json``.  Undeclared drift therefore
still fails, which is the case worth catching -- a file everyone believes is
pinned changing quietly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DRIFT_FILE = "validation/source_drift.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_declared_drift(root: Path, run_key: str) -> dict:
    """Return the declared-drift record for ``run_key``, or an empty record."""
    path = root / DRIFT_FILE
    if not path.exists():
        return {"files": [], "reason": ""}
    document = json.loads(path.read_text(encoding="utf-8"))
    record = (document.get("declared_drift") or {}).get(run_key) or {}
    return {"files": list(record.get("files", [])), "reason": record.get("reason", "")}


def check_artifacts(manifest: dict, run_dir: Path) -> list[str]:
    """Return the artifacts whose bytes no longer match the manifest."""
    return [
        name for name, expected in (manifest.get("artifact_sha256") or {}).items()
        if not (run_dir / name).exists() or sha256_file(run_dir / name) != expected
    ]


def check_runtime_sources(manifest: dict, root: Path, declared: dict) -> dict:
    """Classify each recorded runtime source as matching, declared-drift or not.

    ``undeclared_drift`` and ``missing`` are the failure channels; ``drifted``
    is informational so a caller can print what the declaration covers.
    """
    declared_files = set(declared.get("files", []))
    matching: list[str] = []
    drifted: list[str] = []
    undeclared: list[str] = []
    missing: list[str] = []
    for name, expected in (manifest.get("runtime_source_sha256") or {}).items():
        path = root / name
        if not path.exists():
            missing.append(name)
        elif sha256_file(path) == expected:
            matching.append(name)
        elif name in declared_files:
            drifted.append(name)
        else:
            undeclared.append(name)
    return {
        "matching": sorted(matching),
        "declared_drift": sorted(drifted),
        "undeclared_drift": sorted(undeclared),
        "missing": sorted(missing),
        "stale_declarations": sorted(declared_files - set(drifted)),
    }
