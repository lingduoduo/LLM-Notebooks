#!/usr/bin/env python3
"""Regenerate and check the whole room-taxonomy use case in one command.

Stages run in order; each prints OK, SKIPPED (with the reason) or FAILED:

  build        regenerate taxonomy, schema and seed keywords; validate keywords.csv
  fixtures     rebuild the sample outputs and compare them with the committed ones
  review       validate the review workbook (dry run; never rewrites keywords.csv)
  evaluate     score the keyword baseline on the semantic benchmark
  minilm       re-score the recorded MiniLM predictions (no model needed)
  explorer     build the offline taxonomy explorer page
  semicircle   draw the keyword-coverage chart (PNG + SVG)
  minilm-live  re-run the MiniLM comparison (only with --with-minilm)
  tests        run the unit tests (only with --tests)

Generated reports and visuals go to --out (default: ./out, git-ignored).
Only the standard library is required; stages whose optional packages are
missing are skipped. Exit status is 1 if any stage failed.

Run: python run_all.py [--out DIR] [--skip STAGE ...] [--with-minilm] [--tests]
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = (
    "build",
    "fixtures",
    "review",
    "evaluate",
    "minilm",
    "explorer",
    "semicircle",
    "minilm-live",
    "tests",
)


class Skip(Exception):
    """Raised by a stage that cannot run here; the message says why."""


def needs(*modules):
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise Skip(f"needs {', '.join(missing)} (pip install {' '.join(missing)})")


def run(*args):
    """Run a script from this folder; return its last output line or raise."""
    result = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    if result.returncode:
        raise RuntimeError("\n".join(output[-5:]) or f"exit {result.returncode}")
    return output[-1] if output else "done"


def stage_build(out):
    run("build.py", "--init")
    import build

    taxonomy = build.load_taxonomy()
    rows = build.load_keyword_rows()
    keywords = build.load()[1]
    policy = build.load_policy_candidates()
    excluded = len(rows) - len(keywords) - len(policy)
    return (
        f"taxonomy {taxonomy['version']} ({len(taxonomy['nodes'])} concepts); "
        f"keywords.csv: {len(keywords)} keywords, {len(policy)} policy, {excluded} excluded"
    )


def stage_fixtures(out):
    import build

    sample = json.loads((ROOT / "sample_output.json").read_text(encoding="utf-8"))
    room = json.loads((ROOT / "sample_room_output.json").read_text(encoding="utf-8"))
    parents, rows = build.load()
    legacy = build.match(sample["subject"], sample["tags"], parents, rows)
    classifier = build.RoomClassifier(build.load_taxonomy(), rows)
    rebuilt = classifier.classify_room(room["room_id"], room["subject"], room["tags"])
    for name, data in (
        ("sample_output.json", legacy),
        ("sample_room_output.json", rebuilt),
    ):
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    stale = [
        name
        for name, ok in (
            ("sample_output.json", legacy == sample),
            ("sample_room_output.json", rebuilt == room),
        )
        if not ok
    ]
    if stale:
        raise RuntimeError(f"rebuilt output differs from committed {', '.join(stale)}")
    return f"both samples match the committed fixtures; copies in {out.name}/"


def stage_review(out):
    needs("openpyxl")
    workbook = ROOT / "review" / "ldnoobw_mapping_review.xlsx"
    return run("apply_review.py", workbook, "--status", "model_reviewed", "--dry-run")


def stage_evaluate(out):
    run("semantic_evaluation.py", "--output", out / "keyword_evaluation.json")
    return _concept_f1(out / "keyword_evaluation.json")


def stage_minilm(out):
    predictions = ROOT / "evaluation" / "local_semantic_predictions.jsonl"
    report = out / "minilm_recorded_evaluation.json"
    run("semantic_evaluation.py", "--predictions", predictions, "--output", report)
    return _concept_f1(report)


def stage_explorer(out):
    run("build_taxonomy_explorer.py", "--output", out / "taxonomy_explorer.html")
    return f"{out.name}/taxonomy_explorer.html"


def stage_semicircle(out):
    needs("matplotlib", "numpy")
    run("semicircle_taxonomy.py", "--output", out / "taxonomy_semicircle")
    return f"{out.name}/taxonomy_semicircle.png and .svg"


def stage_minilm_live(out):
    needs("torch", "transformers", "huggingface_hub")
    run("compare_semantic.py", "--output-dir", out / "semantic_comparison")
    return f"{out.name}/semantic_comparison/"


def stage_tests(out):
    return run("-W", "error::ResourceWarning", "-m", "unittest", "discover")


def _concept_f1(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    concepts = report["concepts"]
    return (
        f"{report['prediction_source']}: concept micro-F1 {concepts['micro_f1']:.3f} "
        f"on {report['count']} cases -> {path.parent.name}/{path.name}"
    )


RUNNERS = {
    "build": stage_build,
    "fixtures": stage_fixtures,
    "review": stage_review,
    "evaluate": stage_evaluate,
    "minilm": stage_minilm,
    "explorer": stage_explorer,
    "semicircle": stage_semicircle,
    "minilm-live": stage_minilm_live,
    "tests": stage_tests,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:13]),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--skip", action="append", choices=STAGES, default=[])
    parser.add_argument(
        "--with-minilm",
        action="store_true",
        help="also re-run the live MiniLM comparison (needs torch and a cached model)",
    )
    parser.add_argument("--tests", action="store_true", help="also run the unit tests")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    skipped = set(args.skip)
    if not args.with_minilm:
        skipped.add("minilm-live")
    if not args.tests:
        skipped.add("tests")

    failed = False
    for name in STAGES:
        if name in skipped:
            status, detail = "SKIPPED", "not requested"
        else:
            try:
                status, detail = "OK", RUNNERS[name](out)
            except Skip as reason:
                status, detail = "SKIPPED", str(reason)
            except Exception as error:  # report every stage, then exit non-zero
                status, detail, failed = "FAILED", str(error), True
        print(f"{status:<8} {name:<12} {detail}", flush=True)
    print(f"\nOutputs: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
