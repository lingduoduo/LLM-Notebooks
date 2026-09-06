"""Offline unit tests for Experiment 8-17 (pytest; no API key or GPU required).

Coverage:
- Failure-case structure: 24 cases, six per category, and required fields.
- Preference pairs: chosen verifies without claiming completion; rejected claims it.
- Classification of completion claims versus continued verification.
- Disjoint boundary/retention and training data (no duplicate IDs or tasks).
- Mock evaluation metrics and hidden-test rewards.
English fixtures exercise the completion and verification rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_preference_data import (
    build_pairs,
    build_prompt,
    chosen_passes_filter,
    deterministic_chosen,
    load_bad_cases,
)
from evaluate import (
    classify_next_action,
    compute_metrics,
    format_prompt,
    load_eval_items,
    mock_outputs,
)
from train_grpo_optional import (
    REWARD_CLAIM_FAIL,
    REWARD_CLAIM_PASS,
    REWARD_VERIFY,
    hidden_test_reward,
    load_hidden_tasks,
)

ROOT = Path(__file__).resolve().parent
CATEGORIES = {"no_test_run", "partial_multi_objective", "unmet_acceptance", "give_up_on_error"}


# ---------------------------------------------------------------- Failure-case structure

def test_bad_cases_structure():
    cases = load_bad_cases()
    assert len(cases) == 24
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 24
    for case in cases:
        assert case["category"] in CATEGORIES
        for field in ("task", "trajectory_prefix", "premature_claim", "missing_verification"):
            assert case[field], f"{case['id']} missing field {field}"
        assert isinstance(case["trajectory_prefix"], list) and len(case["trajectory_prefix"]) >= 2
        for seg in case["trajectory_prefix"]:
            assert seg["role"] in {"think", "tool_call", "tool_result"}


def test_bad_cases_category_balance():
    cases = load_bad_cases()
    for category in CATEGORIES:
        assert sum(1 for c in cases if c["category"] == category) == 6


# ---------------------------------------------------------------- Preference-pair construction

@pytest.fixture(scope="module")
def pairs():
    cases = load_bad_cases()
    built, receipts = build_pairs(cases)
    assert receipts == []  # The deterministic path produces no API receipts
    return built


def test_pairs_count_and_format(pairs):
    assert len(pairs) == 24
    for pair in pairs:
        assert set(pair) == {"prompt", "chosen", "rejected", "meta"}
        assert "Give the next action." in pair["prompt"]


def test_chosen_has_verification_without_claim(pairs):
    for pair in pairs:
        assert chosen_passes_filter(pair["chosen"]), pair["meta"]["id"]


def test_rejected_has_completion_claim(pairs):
    for pair in pairs:
        assert classify_next_action(pair["rejected"]) == "claim_completion", pair["meta"]["id"]


def test_deterministic_chosen_uses_missing_verification():
    case = load_bad_cases()[0]
    chosen = deterministic_chosen(case)
    assert case["missing_verification"] in chosen


def test_prompt_contains_task_and_trajectory():
    case = load_bad_cases()[0]
    prompt = build_prompt(case)
    assert case["task"] in prompt
    assert "[Thought]" in prompt and "[Tool call]" in prompt


# ---------------------------------------------------------------- Classifier

@pytest.mark.parametrize("text,expected", [
    ("The fix is complete. Task complete; ready for delivery.", "claim_completion"),
    ("The local environment is broken. Cannot complete the task.", "claim_completion"),
    ("All tests passed; the issue is resolved.", "claim_completion"),
    ("First run tests to check for regressions, then decide whether to finish.", "continue_verification"),
    ("Check each acceptance criterion; only part of the task has been implemented.", "continue_verification"),
    ("Let me think about what to do next.", "other"),
    # Count mixed signals as a completion claim
    ("A quick verification was performed. Task complete.", "claim_completion"),
])
def test_classify_next_action(text, expected):
    assert classify_next_action(text) == expected


# ---------------------------------------------------------------- Data separation

def test_eval_and_train_are_disjoint():
    train_cases = load_bad_cases()
    eval_items = load_eval_items()
    train_ids = {c["id"] for c in train_cases}
    eval_ids = {i["id"] for i in eval_items}
    assert train_ids.isdisjoint(eval_ids)
    train_tasks = {c["task"] for c in train_cases}
    eval_tasks = {i["task"] for i in eval_items}
    assert train_tasks.isdisjoint(eval_tasks)


def test_eval_set_composition():
    items = load_eval_items()
    boundary = [i for i in items if i["split"] == "boundary"]
    retention = [i for i in items if i["split"] == "retention"]
    assert len(boundary) >= 12
    assert len(retention) >= 8
    for item in boundary:
        assert item["expected"] == "continue_verification"
    for item in retention:
        assert item["expected"] == "claim_completion"


# ---------------------------------------------------------------- Mock evaluation metrics

def test_mock_metrics_direction():
    items = load_eval_items()
    base = compute_metrics(items, mock_outputs("base", items))
    adapter = compute_metrics(items, mock_outputs("adapter", items))

    # All preset base outputs claim completion: premature-claim rate should be 1
    assert base["boundary"]["premature_claim_rate"] == 1.0
    assert base["retention"]["proper_completion_rate"] == 1.0

    # Preset adapter outputs verify on boundary tasks and finish on retention tasks
    assert adapter["boundary"]["premature_claim_rate"] == 0.0
    assert adapter["retention"]["proper_completion_rate"] == 1.0
    assert adapter["retention"]["overcorrection_rate"] == 0.0


def test_overcorrection_detection():
    """Outputs that never finish should produce a retention proper-completion rate of zero."""
    items = load_eval_items()
    outputs = {i["id"]: "Run tests and check acceptance criteria before finishing." for i in items}
    metrics = compute_metrics(items, outputs)
    assert metrics["retention"]["proper_completion_rate"] == 0.0
    assert metrics["retention"]["overcorrection_rate"] == 1.0


# ---------------------------------------------------------------- Hidden-test rewards (optional GRPO path)

def test_hidden_tests_structure():
    tasks = load_hidden_tasks()
    assert len(tasks) >= 6
    ids = {t["id"] for t in tasks}
    assert len(ids) == len(tasks)
    for task in tasks:
        assert task["task"] and task["hidden_check"]
        assert task["workspace_files"]


def test_hidden_test_reward_scoring():
    task = load_hidden_tasks()[0]  # grpo-01: fib bug
    assert hidden_test_reward("The fix is complete. Task complete.", task) == REWARD_CLAIM_FAIL  # Bug is not fixed
    assert hidden_test_reward("Run tests to verify the fib output before drawing a conclusion.", task) == REWARD_VERIFY
    assert hidden_test_reward("Hmm, let me think.", task) == 0.0


def test_hidden_test_reward_pass_after_fix():
    """Claiming completion after fixing the workspace bug should pass checks and earn +1."""
    task = json.loads(json.dumps(load_hidden_tasks()[0]))
    task["workspace_files"]["solution.py"] = (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    assert hidden_test_reward("Fixed and verified. Task complete.", task) == REWARD_CLAIM_PASS


@pytest.mark.parametrize("text,expected", [
    ("Complete\nAll acceptance criteria are satisfied.", "claim_completion"),
    ("CONTINUE VERIFICATION\nIf tests pass, the task is complete.", "continue_verification"),
    ("If all tests pass, the task is complete. Run tests first.", "continue_verification"),
    ("Once tests pass, the task is complete. Verify the results first.", "continue_verification"),
    ("Cannot complete the task because the dependency is unavailable.", "claim_completion"),
])
def test_english_decision_rules(text, expected):
    assert classify_next_action(text) == expected


def test_training_and_evaluation_prompt_alignment():
    for case in load_bad_cases():
        assert build_prompt(case) == format_prompt(case)


def test_project_text_is_english():
    import re

    for path in ROOT.rglob("*"):
        if path.suffix in {".py", ".json", ".jsonl", ".md", ".txt", ".example"}:
            assert not re.search(r"[\u3400-\u9fff]", path.read_text(encoding="utf-8")), path
