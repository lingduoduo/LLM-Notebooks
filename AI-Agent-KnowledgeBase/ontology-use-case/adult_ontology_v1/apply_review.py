"""Write reviewer decisions from the mapping-review workbook back to the vocabulary.

The workbook is the source of truth for every keyword it decides. Each decided
row sets the outcome for that imported keyword (source ldnoobw_en by default):

- accept: keep it under "Mapped concept_id" with the given review_status.
- change: keep it under "Corrected concept_id" with the given review_status.
- move to policy: keep it out of the vocabulary and list it in
  policy_candidates_ldnoobw_en.csv with its "Policy route".
- drop: keep it out of the vocabulary and the policy candidates.

Because outcomes are set rather than edited, re-applying a workbook changes
nothing, and overriding an applied drop or policy move re-creates the keyword
in place. "Set context_required" (true/false) overrides the flag on kept rows;
a re-created row otherwise uses the workbook's "context_required" column, or
"true" when that column is absent. Rows without a decision are left untouched,
and every decision is validated before any file is written.

Install: pip install openpyxl
Run: python apply_review.py review/ldnoobw_mapping_review.xlsx --status human_reviewed
"""

import argparse
import csv

import build
from matching import Matcher, norm

POLICY_FILE = "policy_candidates_ldnoobw_en.csv"
POLICY_ROUTES = (
    "minor_safety",
    "non_consent",
    "incest",
    "bestiality",
    "violence_extremism",
    "solicitation",
    "hate_slur",
)
DECISIONS = ("accept", "change", "move to policy", "drop")
COLUMNS = {
    "keyword": "Keyword",
    "concept": "Mapped concept_id",
    "decision": "Decision",
    "corrected": "Corrected concept_id",
    "route": "Policy route",
    "context": "Set context_required",
}
# Optional: the original flag, used when a keyword has to be re-created.
OPTIONAL_COLUMNS = {"original_context": "context_required"}


def read_decisions(workbook_path):
    from openpyxl import load_workbook

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        rows = workbook["Review"].iter_rows(values_only=True)
        header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
        missing = [name for name in COLUMNS.values() if name not in header]
        if missing:
            raise ValueError(f"Review sheet is missing columns: {missing}")
        index = {key: header.index(name) for key, name in COLUMNS.items()}
        index.update(
            (key, header.index(name))
            for key, name in OPTIONAL_COLUMNS.items()
            if name in header
        )
        decided = []
        for row in rows:
            record = {
                key: "" if i >= len(row) or row[i] is None else str(row[i]).strip()
                for key, i in index.items()
            }
            if record["decision"]:
                decided.append(record)
        return decided
    finally:
        workbook.close()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def apply_review(workbook_path, status, dry_run=False, source="ldnoobw_en"):
    """Validate every decision, then rewrite the vocabulary and policy files."""
    taxonomy = build.load_taxonomy()
    nodes = {node["id"]: node for node in taxonomy["nodes"]}
    vocab_path = build.ROOT / build.VOCABULARY_FILE
    policy_path = build.ROOT / POLICY_FILE
    vocab_fields, vocabulary = read_csv(vocab_path)
    policy_fields, policy = read_csv(policy_path)
    current = {norm(r["keyword"]): r for r in vocabulary if r["source"] == source}

    outcomes = {}  # normalized keyword -> vocabulary row, policy row, or None
    summary = {decision: 0 for decision in DECISIONS}
    for d in read_decisions(workbook_path):
        keyword = norm(d["keyword"])
        where = f"{d['keyword']!r} ({d['concept']})"
        if not keyword:
            raise ValueError("Decided rows need a keyword")
        if keyword in outcomes:
            raise ValueError(f"Keyword decided more than once: {where}")
        if d["decision"] not in DECISIONS:
            raise ValueError(f"Unknown decision {d['decision']!r} for {where}")
        if d["context"] not in ("", "true", "false"):
            raise ValueError(f"Set context_required must be true or false for {where}")
        concept = d["corrected"] if d["decision"] == "change" else d["concept"]
        if d["decision"] in ("accept", "change"):
            target = nodes.get(concept)
            if target is None or target["level"] == 0:
                raise ValueError(f"Concept must be an L1-L3 concept for {where}")
            if target["category_id"] == "IDENTITY":
                raise ValueError(f"Keywords may not map to IDENTITY concepts: {where}")
            previous = current.get(keyword, {})
            context = (
                d["context"]
                or previous.get("context_required")
                or d.get("original_context")
                or "true"
            )
            outcomes[keyword] = (
                "vocabulary",
                {
                    "keyword": previous.get("keyword", keyword),
                    "language": previous.get("language", "en"),
                    "concept_id": concept,
                    "source": source,
                    "context_required": context,
                    "review_status": status,
                },
            )
        elif d["decision"] == "move to policy":
            if d["route"] not in POLICY_ROUTES:
                raise ValueError(
                    f"Policy route must be one of {POLICY_ROUTES} for {where}"
                )
            outcomes[keyword] = (
                "policy",
                {
                    "keyword": current.get(keyword, {}).get("keyword", keyword),
                    "language": "en",
                    "policy_route": d["route"],
                    "source": source,
                    "review_status": "policy_review_required",
                },
            )
        else:
            outcomes[keyword] = ("drop", None)
        summary[d["decision"]] += 1

    def settle(rows, kind):
        """Replace decided rows in place; append decided rows not yet present."""
        settled, seen = [], set()
        for row in rows:
            key = norm(row["keyword"])
            if row["source"] != source or key not in outcomes:
                settled.append(row)
                continue
            seen.add(key)
            where, new = outcomes[key]
            if where == kind:
                settled.append({**row, **new})
        for key, (where, new) in outcomes.items():
            if where == kind and key not in seen:
                settled.append(new)
        return settled

    vocabulary, policy = settle(vocabulary, "vocabulary"), settle(policy, "policy")
    build.check_vocabulary(vocabulary)
    Matcher({i: n["parent_id"] for i, n in nodes.items()}, vocabulary)
    print(
        ("Would apply" if dry_run else "Applied"),
        ", ".join(f"{count} {decision}" for decision, count in summary.items()),
    )
    if dry_run:
        return summary
    # Same dialects as the generator: CRLF vocabulary, LF policy candidates.
    with vocab_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=vocab_fields)
        writer.writeheader()
        writer.writerows(vocabulary)
    with policy_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=policy_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(policy)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook")
    parser.add_argument(
        "--status",
        default="human_reviewed",
        help="review_status for accepted and changed rows",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", default="ldnoobw_en")
    args = parser.parse_args()
    apply_review(args.workbook, args.status, args.dry_run, args.source)
