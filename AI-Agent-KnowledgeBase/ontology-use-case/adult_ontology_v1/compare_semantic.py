#!/usr/bin/env python3
"""Compare a cached local MiniLM prototype baseline with the keyword matcher."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

from local_encoder import CachedMiniLMEncoder, sha256_file
from local_semantic import (
    DEFAULT_MAX_LABELS,
    DEFAULT_THRESHOLD,
    SemanticBaseline,
    validate_settings,
)
from semantic_evaluation import (
    DEFAULT_BENCHMARK,
    evaluate,
    load_benchmark,
    select_cases,
)

ROOT = Path(__file__).resolve().parent


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def run_comparison(
    benchmark: dict,
    taxonomy: dict,
    rows: list,
    encoder,
    *,
    split="test",
    threshold=DEFAULT_THRESHOLD,
    max_labels=DEFAULT_MAX_LABELS,
) -> tuple:
    """Prepare label prototypes once, predict from inputs only, then score gold."""
    from room_classifier import RoomClassifier

    cases = select_cases(benchmark, taxonomy, split)
    model = SemanticBaseline(
        taxonomy, encoder, threshold=threshold, max_labels=max_labels
    )
    predictions = []
    for case in cases:
        # Do not pass expected labels, slice or group metadata into the predictor.
        prediction = model.predict(
            case["subject"],
            case["tags"],
            self_declared_concepts=case.get("self_declared_concepts", []),
        )
        predictions.append(
            {
                "case_id": case["id"],
                "taxonomy_version": taxonomy["version"],
                **prediction,
            }
        )
    semantic_report = evaluate(
        benchmark, taxonomy, predictions=predictions, split=split
    )
    keyword_report = evaluate(
        benchmark, taxonomy, classifier=RoomClassifier(taxonomy, rows), split=split
    )
    comparison = {
        "method": "taxonomy_prototype_cosine_similarity",
        "configuration": {
            "threshold": threshold,
            "max_semantic_labels": max_labels,
            "score_aggregation": "maximum_over_subject_and_first_five_tags",
            "ancestor_selection": "remove_qualified_ancestors_before_top_k",
            "prototype_source": "taxonomy_labels_and_parent_paths_only",
            "threshold_calibration": "uncalibrated_settings_no_fitting_in_this_command",
            "uses_default_settings": threshold == DEFAULT_THRESHOLD
            and max_labels == DEFAULT_MAX_LABELS,
        },
        "taxonomy_version": taxonomy["version"],
        "taxonomy_sha256_canonical": _json_hash(taxonomy),
        "prototypes_sha256_canonical": _json_hash(model.prototypes),
        "keyword": keyword_report,
        "semantic": semantic_report,
    }
    return comparison, predictions, model.prototypes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Complete local mean-pooling MiniLM folder; default is a pinned cached snapshot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Cosine cutoff; not a probability (default 0.5)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=DEFAULT_MAX_LABELS,
        help="Maximum semantic candidates, before adding explicit identity declarations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Write comparison JSON, predictions JSONL and prototype JSON here",
    )
    args = parser.parse_args(argv)
    from build import load, load_taxonomy

    try:
        validate_settings(args.threshold, args.max_labels)
        taxonomy = load_taxonomy()
        benchmark = load_benchmark(args.benchmark)
        select_cases(
            benchmark, taxonomy, args.split
        )  # Fail bad inputs before loading weights.
        rows = load()[1]
        encoder = CachedMiniLMEncoder(args.model_path)
        report, predictions, prototypes = run_comparison(
            benchmark,
            taxonomy,
            rows,
            encoder,
            split=args.split,
            threshold=args.threshold,
            max_labels=args.max_labels,
        )
        report.update(
            model=encoder.metadata,
            python=platform.python_version(),
            benchmark_sha256=sha256_file(args.benchmark),
            vocabulary_sha256=sha256_file(ROOT / "keywords.csv"),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "semantic_comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "semantic_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        (args.output_dir / "semantic_prototypes.json").write_text(
            json.dumps(prototypes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{report['keyword']['count']} cases; taxonomy {taxonomy['version']}; no model downloads"
        )
        for name in ("keyword", "semantic"):
            result = report[name]
            print(
                f"{name}: concept micro-F1={result['concepts']['micro_f1']:.3f}, exact-match={result['concepts']['exact_match']:.3f}, negative candidate FPR={result['taxonomy_negative_false_positive_rate']}"
            )
        print(
            "Small synthetic challenge set; cosine scores are not calibrated probabilities. See per-case errors."
        )
        return 0
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
    ) as error:
        parser.exit(2, f"Local comparison error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
