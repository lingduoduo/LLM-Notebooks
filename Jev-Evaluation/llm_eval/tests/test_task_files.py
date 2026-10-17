"""The intro notebook reads tasks/public_all.jsonl and tasks/smoke.jsonl;
both must exist, load, and stay consistent with the raw public files."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_eval.task import load_tasks  # noqa: E402

TASKS = ROOT / "tasks"


def test_public_all_has_231_unique_tasks():
    tasks = load_tasks(str(TASKS / "public_all.jsonl"))
    assert len(tasks) == 231
    assert len({t.id for t in tasks}) == 231


def test_public_all_matches_raw_tier_files():
    raw = sum(
        sum(1 for _ in open(TASKS / "public" / f"{tier}.jsonl"))
        for tier in ("easy", "original", "hard")
    )
    assert raw == 231


def test_smoke_is_a_subset_of_public_all():
    public = {
        json.loads(line)["id"]: line.rstrip("\n")
        for line in open(TASKS / "public_all.jsonl")
    }
    for line in open(TASKS / "smoke.jsonl"):
        rec = json.loads(line)
        assert public[rec["id"]] == line.rstrip("\n")


def test_smoke_covers_every_question_type_and_one_paraphrase_pair():
    smoke = load_tasks(str(TASKS / "smoke.jsonl"))
    assert len(smoke) == 6
    assert {t.question_type for t in smoke} == {"choice", "noul", "score"}
    groups = Counter(t.group for t in smoke if t.group)
    assert [n for n in groups.values() if n >= 2] == [2]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
