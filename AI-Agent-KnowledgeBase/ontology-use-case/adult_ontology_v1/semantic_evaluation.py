#!/usr/bin/env python3
"""Evaluate semantic room labels offline against independently authored gold."""
import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable, Sequence

from matching import norm
from taxonomy import validate_taxonomy

ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = ROOT / "evaluation" / "semantic_cases.json"
SPLITS = ("train", "validation", "test")


def load_benchmark(path=DEFAULT_BENCHMARK) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def score_labels(expected: Sequence[set], predicted: Sequence[set]) -> dict:
    """Micro counts and macro F1 over labels with gold or predicted support.

    All-empty label sets have zero precision/recall/F1 and perfect exact match.
    Empty selected datasets are rejected rather than reported as successful runs.
    """
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected non-empty, equally sized gold and prediction lists")
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    exact = 0
    for gold, guess in zip(expected, predicted):
        tp.update(gold & guess)
        fp.update(guess - gold)
        fn.update(gold - guess)
        support.update(gold)
        exact += gold == guess

    def metrics(t, f, n):
        return {
            "precision": t / (t + f) if t + f else 0.0,
            "recall": t / (t + n) if t + n else 0.0,
            "f1": 2 * t / (2 * t + f + n) if 2 * t + f + n else 0.0,
        }

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = {
        label: dict(metrics(tp[label], fp[label], fn[label]), support=support[label])
        for label in labels
    }
    totals = (sum(tp.values()), sum(fp.values()), sum(fn.values()))
    micro = metrics(*totals)
    return {
        "tp": totals[0],
        "fp": totals[1],
        "fn": totals[2],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": (
            sum(m["f1"] for m in per_label.values()) / len(labels) if labels else 0.0
        ),
        "macro_label_count": len(labels),
        "exact_match": exact / len(expected),
        "per_label": per_label,
    }


def _concept_set(value, nodes, context):
    if not isinstance(value, list) or any(
        not isinstance(c, str) or c not in nodes or nodes[c]["level"] == 0
        for c in value
    ):
        raise ValueError(
            f"{context}: expected known L1–L3 concept IDs, not binary/moderation labels"
        )
    if len(set(value)) != len(value):
        raise ValueError(f"{context}: duplicate concept IDs")
    return set(value)


def _validate_benchmark(benchmark, taxonomy, nodes):
    if not isinstance(benchmark, dict) or benchmark.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported benchmark schema_version")
    if benchmark.get("taxonomy_version") != taxonomy["version"]:
        raise ValueError("Benchmark taxonomy_version does not match loaded taxonomy")
    cases = benchmark.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must contain labeled cases")
    ids, groups, texts = set(), {}, {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object")
        for name in ("id", "group_id", "slice"):
            if not isinstance(case.get(name), str) or not case[name].strip():
                raise ValueError(f"Case requires non-empty {name}")
        if case["id"] in ids:
            raise ValueError(f'Duplicate case ID: {case["id"]}')
        ids.add(case["id"])
        split = case.get("split")
        if split not in SPLITS:
            raise ValueError("Unknown case split")
        if (
            not isinstance(case.get("subject"), str)
            or not isinstance(case.get("tags"), list)
            or len(case["tags"]) > 5
            or any(not isinstance(t, str) for t in case["tags"])
        ):
            raise ValueError("Case requires subject and at most five text tags")
        gold = _concept_set(case.get("expected_concepts"), nodes, "Gold")
        declarations = _concept_set(
            case.get("self_declared_concepts", []), nodes, "Declaration"
        )
        if any(nodes[c]["category_id"] != "IDENTITY" for c in declarations):
            raise ValueError("Declarations must reference IDENTITY concepts")
        if any(
            nodes[c]["category_id"] == "IDENTITY" and c not in declarations
            for c in gold
        ):
            raise ValueError("Gold identity concept requires self-declaration")
        # Normalize solely for leakage checks; raw text remains available for evaluation.
        fingerprint = (
            norm(case["subject"]),
            tuple(sorted(norm(t) for t in case["tags"])),
            tuple(sorted(declarations)),
        )
        for table, key in ((groups, case["group_id"]), (texts, fingerprint)):
            if key in table and table[key] != split:
                raise ValueError(f'Cross-split leakage in case {case["id"]}')
            table[key] = split
    return cases


def evaluate(
    benchmark: dict,
    taxonomy: dict,
    *,
    predictions: Iterable[dict] = None,
    classifier=None,
    split: str = "test",
) -> dict:
    """Score either external predictions or the prepared keyword baseline."""
    validate_taxonomy(taxonomy)
    nodes = {node["id"]: node for node in taxonomy["nodes"]}
    cases = _validate_benchmark(benchmark, taxonomy, nodes)
    cases = [case for case in cases if case["split"] == split]
    if not cases:
        raise ValueError(f"No cases for split {split}")
    if (predictions is None) == (classifier is None):
        raise ValueError(
            "Provide exactly one prediction source: classifier or predictions"
        )
    guesses = {}
    reviews = {}
    if predictions is not None:
        for row in predictions:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("case_id"), str)
                or row["case_id"] in guesses
            ):
                raise ValueError("Prediction case IDs must be unique strings")
            if row.get("taxonomy_version") != taxonomy["version"]:
                raise ValueError(
                    "Prediction taxonomy_version does not match loaded taxonomy"
                )
            if "requires_context_review" in row:
                if type(row["requires_context_review"]) is not bool:
                    raise ValueError("Prediction review flag must be boolean")
                reviews[row["case_id"]] = row["requires_context_review"]
            guesses[row["case_id"]] = _concept_set(
                row.get("concept_ids"), nodes, "Prediction"
            )
        if set(guesses) != {case["id"] for case in cases}:
            raise ValueError(
                "Prediction IDs must exactly match the selected split; no missing/extra cases"
            )
    else:
        if classifier.version != taxonomy["version"]:
            raise ValueError(
                "Classifier taxonomy_version does not match loaded taxonomy"
            )
        for case in cases:
            result = classifier.classify_room(
                case["id"],
                case["subject"],
                case["tags"],
                self_declared_concepts=case.get("self_declared_concepts", []),
            )
            guesses[case["id"]] = {a["concept_id"] for a in result["assignments"]}
            reviews[case["id"]] = result["requires_context_review"]

    def expand(concepts):
        expanded = set()
        for concept in concepts:
            while concept is not None:
                if nodes[concept]["level"] > 0:
                    expanded.add(concept)
                concept = nodes[concept]["parent_id"]
        return expanded

    gold = [set(case["expected_concepts"]) for case in cases]
    predicted = [guesses[case["id"]] for case in cases]

    def categories(items):
        return {nodes[c]["category_id"] for c in items}

    negative_indices = [i for i, expected in enumerate(gold) if not expected]
    rows = [
        {
            "case_id": case["id"],
            "slice": case["slice"],
            "expected": sorted(expected),
            "predicted": sorted(guess),
            "requires_context_review": reviews.get(case["id"]),
            "missing": sorted(expected - guess),
            "extra": sorted(guess - expected),
        }
        for case, expected, guess in zip(cases, gold, predicted)
    ]
    slices = {}
    for name in sorted({case["slice"] for case in cases}):
        indices = [i for i, case in enumerate(cases) if case["slice"] == name]
        slices[name] = {
            "count": len(indices),
            "concepts": score_labels(
                [gold[i] for i in indices], [predicted[i] for i in indices]
            ),
        }
    return {
        "taxonomy_version": taxonomy["version"],
        "split": split,
        "count": len(cases),
        "prediction_source": (
            "keyword_baseline" if classifier is not None else "external_predictions"
        ),
        "concepts": score_labels(gold, predicted),
        "categories": score_labels(
            [categories(g) for g in gold], [categories(p) for p in predicted]
        ),
        "hierarchical": score_labels(
            [expand(g) for g in gold], [expand(p) for p in predicted]
        ),
        "review_flagged_count": sum(flag is True for flag in reviews.values()),
        "review_flag_coverage": len(reviews) / len(cases),
        "taxonomy_negative_count": len(negative_indices),
        "taxonomy_negative_false_positive_rate": (
            sum(bool(predicted[i]) for i in negative_indices) / len(negative_indices)
            if negative_indices
            else None
        ),
        "slices": slices,
        "cases": rows,
        "failures": [row for row in rows if row["missing"] or row["extra"]],
    }


def _load_predictions(path):
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Predictions JSON must be an array (or use JSONL)")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="External semantic-model predictions, JSON array or JSONL; no API is called",
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument(
        "--output", type=Path, help="Write full metrics and per-case errors as JSON"
    )
    args = parser.parse_args(argv)
    from build import RoomClassifier, load, load_taxonomy

    try:
        taxonomy = load_taxonomy()
        kwargs = (
            {"predictions": _load_predictions(args.predictions)}
            if args.predictions
            else {"classifier": RoomClassifier(taxonomy, load()[1])}
        )
        report = evaluate(
            load_benchmark(args.benchmark), taxonomy, split=args.split, **kwargs
        )
        if args.output:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            f"{report['prediction_source']}: {report['count']} cases, taxonomy {report['taxonomy_version']}"
        )
        for key in ("concepts", "categories", "hierarchical"):
            metric = report[key]
            print(
                f"{key}: micro-F1={metric['micro_f1']:.3f}, exact-match={metric['exact_match']:.3f}"
            )
        print(
            f"Taxonomy-negative false-positive rate: {report['taxonomy_negative_false_positive_rate']}"
        )
        print(
            f"Cases with label errors: {len(report['failures'])}; this challenge fixture is not representative accuracy"
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Evaluation error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
