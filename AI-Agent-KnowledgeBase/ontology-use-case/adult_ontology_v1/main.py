#!/usr/bin/env python3
"""Build, use and check the room taxonomy. Each command is one README step.

  python main.py all            regenerate and check the whole use case
  python main.py build          regenerate taxonomy, schema and seed keywords
  python main.py classify ...   classify one room (or legacy keyword matches)
  python main.py import-words   add a word list as unmapped candidates
  python main.py review ...     apply the mapping-review workbook
  python main.py explorer       write the offline taxonomy explorer page
  python main.py chart          draw the keyword-coverage semicircle chart

Run "python main.py COMMAND --help" for options.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import utils

REVIEW_WORKBOOK = utils.ROOT / "review" / "ldnoobw_mapping_review.xlsx"
STAGES = (
    "build",
    "fixtures",
    "review",
    "evaluate",
    "minilm",
    "explorer",
    "chart",
    "minilm-live",
    "tests",
)


# ---- Commands -----------------------------------------------------------------


def cmd_build(args):
    utils.generate()
    return build_summary()


def cmd_classify(args):
    if args.self_declared_concepts and not args.room_id:
        raise SystemExit("--self-declared-concepts requires --room-id")
    parents, rows = utils.load()
    if args.room_id:
        result = utils.RoomClassifier(utils.load_taxonomy(), rows).classify_room(
            args.room_id,
            args.subject,
            args.tags,
            self_declared_concepts=args.self_declared_concepts,
        )
    else:
        result = utils.match(args.subject, args.tags, parents, rows)
    return json.dumps(result, indent=2)


def cmd_import_words(args):
    utils.import_words(args.path, args.language)


def cmd_review(args):
    utils.apply_review(args.workbook, args.status, args.dry_run, args.source)


def cmd_explorer(args):
    return str(utils.build_explorer(args.output))


def cmd_chart(args):
    taxonomy = utils.load_taxonomy()
    paths = utils.draw_taxonomy(
        utils.taxonomy_rows(taxonomy, utils.load()[1]),
        f"Room Taxonomy Keyword Coverage (v{taxonomy['version']})",
        args.output,
        utils.category_order(taxonomy),
    )
    return "Created " + " and ".join(paths)


def cmd_all(args):
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


# ---- Pipeline stages used by "all" ----------------------------------------------


class Skip(Exception):
    """Raised by a stage that cannot run here; the message says why."""


def needs(*modules):
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise Skip(f"needs {', '.join(missing)} (pip install {' '.join(missing)})")


def run(*args):
    """Run a Python command from this folder; return its last output line or raise."""
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=utils.ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    if result.returncode:
        raise RuntimeError("\n".join(output[-5:]) or f"exit {result.returncode}")
    return output[-1] if output else "done"


def build_summary():
    taxonomy = utils.load_taxonomy()
    rows = utils.load_keyword_rows()
    keywords = utils.load()[1]
    policy = utils.load_policy_candidates()
    excluded = len(rows) - len(keywords) - len(policy)
    return (
        f"taxonomy {taxonomy['version']} ({len(taxonomy['nodes'])} concepts); "
        f"keywords.csv: {len(keywords)} keywords, {len(policy)} policy, {excluded} excluded"
    )


def stage_fixtures(out):
    """Rebuild both sample outputs and compare them with the committed ones."""
    sample = json.loads((utils.ROOT / "sample_output.json").read_text("utf-8"))
    room = json.loads((utils.ROOT / "sample_room_output.json").read_text("utf-8"))
    parents, rows = utils.load()
    legacy = utils.match(sample["subject"], sample["tags"], parents, rows)
    classifier = utils.RoomClassifier(utils.load_taxonomy(), rows)
    rebuilt = classifier.classify_room(room["room_id"], room["subject"], room["tags"])
    stale = []
    for name, data, committed in (
        ("sample_output.json", legacy, sample),
        ("sample_room_output.json", rebuilt, room),
    ):
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        if data != committed:
            stale.append(name)
    if stale:
        raise RuntimeError(f"rebuilt output differs from committed {', '.join(stale)}")
    return f"both samples match the committed fixtures; copies in {out.name}/"


def stage_review(out):
    needs("openpyxl")
    with contextlib.redirect_stdout(io.StringIO()):  # the summary is returned below
        summary = utils.apply_review(REVIEW_WORKBOOK, "model_reviewed", dry_run=True)
    return "Would apply " + ", ".join(f"{n} {d}" for d, n in summary.items())


def stage_evaluate(out):
    report = out / "keyword_evaluation.json"
    run("-m", "evaluation.semantic_evaluation", "--output", report)
    return concept_f1(report)


def stage_minilm(out):
    predictions = utils.ROOT / "evaluation" / "local_semantic_predictions.jsonl"
    report = out / "minilm_recorded_evaluation.json"
    run(
        "-m",
        "evaluation.semantic_evaluation",
        "--predictions",
        predictions,
        "--output",
        report,
    )
    return concept_f1(report)


def stage_explorer(out):
    return f"{out.name}/{utils.build_explorer(out / 'taxonomy_explorer.html').name}"


def stage_chart(out):
    needs("matplotlib", "numpy")
    run("main.py", "chart", "--output", out / "taxonomy_semicircle")
    return f"{out.name}/taxonomy_semicircle.png and .svg"


def stage_minilm_live(out):
    needs("torch", "transformers", "huggingface_hub")
    run(
        "-m", "evaluation.compare_semantic", "--output-dir", out / "semantic_comparison"
    )
    return f"{out.name}/semantic_comparison/"


def stage_tests(out):
    return run("-W", "error::ResourceWarning", "-m", "unittest", "discover")


def concept_f1(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    concepts = report["concepts"]
    return (
        f"{report['prediction_source']}: concept micro-F1 {concepts['micro_f1']:.3f} "
        f"on {report['count']} cases -> {path.parent.name}/{path.name}"
    )


RUNNERS = {
    "build": lambda out: cmd_build(None),
    "fixtures": stage_fixtures,
    "review": stage_review,
    "evaluate": stage_evaluate,
    "minilm": stage_minilm,
    "explorer": stage_explorer,
    "chart": stage_chart,
    "minilm-live": stage_minilm_live,
    "tests": stage_tests,
}


# ---- Command line -----------------------------------------------------------------


def parser():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="\n".join(__doc__.splitlines()[2:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("all", help="regenerate and check the whole use case")
    a.add_argument("--out", type=Path, default=utils.ROOT / "out")
    a.add_argument("--skip", action="append", choices=STAGES, default=[])
    a.add_argument(
        "--with-minilm",
        action="store_true",
        help="also re-run the live MiniLM comparison (needs torch and a cached model)",
    )
    a.add_argument("--tests", action="store_true", help="also run the unit tests")
    a.set_defaults(func=cmd_all)

    sub.add_parser(
        "build", help="regenerate taxonomy, schema and seed keywords"
    ).set_defaults(func=cmd_build)

    c = sub.add_parser("classify", help="classify one room")
    c.add_argument("--subject", default="Private show with cosplay and latex")
    c.add_argument("--tags", nargs="*", default=["bdsm", "lingerie"])
    c.add_argument("--room-id", help="return versioned room assignments")
    c.add_argument(
        "--self-declared-concepts",
        nargs="*",
        default=[],
        help="explicitly self-declared IDENTITY concept IDs (requires --room-id)",
    )
    c.set_defaults(func=cmd_classify)

    i = sub.add_parser("import-words", help="add a word list as unmapped candidates")
    i.add_argument("path")
    i.add_argument("--language", default="en")
    i.set_defaults(func=cmd_import_words)

    r = sub.add_parser("review", help="apply the mapping-review workbook")
    r.add_argument("workbook", nargs="?", type=Path, default=REVIEW_WORKBOOK)
    r.add_argument(
        "--status",
        default="human_reviewed",
        help="review_status for accepted and changed rows",
    )
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--source", default="ldnoobw_en")
    r.set_defaults(func=cmd_review)

    e = sub.add_parser("explorer", help="write the offline taxonomy explorer page")
    e.add_argument(
        "--output", type=Path, default=utils.ROOT / "visuals/taxonomy_explorer.html"
    )
    e.set_defaults(func=cmd_explorer)

    h = sub.add_parser("chart", help="draw the keyword-coverage semicircle chart")
    h.add_argument("--output", default="taxonomy_semicircle")
    h.set_defaults(func=cmd_chart)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = args.func(args)
    if isinstance(result, int):
        return result
    if result:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
