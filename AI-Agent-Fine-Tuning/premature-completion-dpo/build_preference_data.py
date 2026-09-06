"""Build DPO preference pairs from premature-completion failure cases.

Two paths generate chosen responses:
1. Offline deterministic path (default): combine templates with each case's
   missing_verification field. No API key is required.
2. Teacher model path (--teacher): use an OpenAI-compatible API and rule-based
   rejection sampling. Chosen responses must include verification actions
   (running tests or checking acceptance criteria) without claiming completion.
   Raw requests, responses, and token usage are saved as evidence receipts.

Write output/preference_pairs.jsonl with prompt/chosen/rejected on each line.
The prompt contains the trajectory prefix and a request for the next action;
the rejected response is the original premature completion claim.
Model-facing text and fixtures use English throughout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import has_completion_claim, has_verification_action

ROOT = Path(__file__).resolve().parent
BAD_CASES_PATH = ROOT / "data" / "bad_cases.json"
OUTPUT_PATH = ROOT / "output" / "preference_pairs.jsonl"

# Offline chosen templates: explain why finishing is premature, then specify missing verification.
CHOSEN_TEMPLATES = {
    "no_test_run": (
        "Continue verification: the code changes have not been tested. "
        "Next action: {missing_verification}. Report results after verification passes."
    ),
    "partial_multi_objective": (
        "Continue verification: some task objectives remain unmet, so it is too early to finish. "
        "Next action: {missing_verification}. Summarize after all objectives are met and verified."
    ),
    "unmet_acceptance": (
        "Continue verification: acceptance criteria remain unmet and require further checks. "
        "Next action: {missing_verification}. Finish after verifying every acceptance criterion."
    ),
    "give_up_on_error": (
        "Continue verification: an error is no reason to give up or fabricate a passing result. "
        "Next action: {missing_verification}. Identify and fix the root cause, then verify again."
    ),
}

TEACHER_SYSTEM = (
    "You are generating preference-optimization training data for a coding agent. Given a task and its execution trajectory, "
    "the agent is about to claim completion prematurely. Write the correct next action: perform the missing verification first "
    "(run tests or check each acceptance criterion), without claiming completion. Write one to three sentences in English."
)


def load_bad_cases(path: Path = BAD_CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(case: dict[str, Any]) -> str:
    """Combine task, trajectory prefix, and next-action request, matching evaluate.format_prompt."""
    lines = [f"Task: {case['task']}", "", "Execution trajectory:"]
    for seg in case["trajectory_prefix"]:
        role = seg["role"]
        if role == "think":
            lines.append(f"[Thought] {seg['content']}")
        elif role == "tool_call":
            lines.append(f"[Tool call] {seg['tool']}({json.dumps(seg.get('arguments', {}), ensure_ascii=False)})")
        else:
            lines.append(f"[Tool result] {seg['content']}")
    lines += [
        "",
        "Give the next action. If all acceptance criteria in the trajectory are satisfied, state that the task is complete; ",
        "if any criterion is unverified or a test fails, continue verification. Write only \"Complete\" or \"Continue verification\" on the first line, followed by one sentence explaining why.",
    ]
    return "\n".join(lines)


def deterministic_chosen(case: dict[str, Any]) -> str:
    """Build an offline chosen response from a template and missing_verification."""
    template = CHOSEN_TEMPLATES[case["category"]]
    return template.format(missing_verification=case["missing_verification"])


def chosen_passes_filter(text: str) -> bool:
    """Accept only responses with verification actions and no completion claim."""
    return has_verification_action(text) and not has_completion_claim(text)


def teacher_chosen(
    case: dict[str, Any],
    client: Any,
    backend: dict[str, Any],
    model: str,
    max_attempts: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate and filter teacher responses; return (chosen, receipts)."""
    from llm_client import chat_with_receipt

    prompt = build_prompt(case)
    receipts: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": prompt + f"\n\nMissing verification (for reference): {case['missing_verification']}"},
            ],
            "temperature": 0.7 if attempt else 0,
        }
        content, receipt = chat_with_receipt(client, backend, request)
        receipt["rejection_sampling"] = {"case_id": case["id"], "attempt": attempt + 1}
        receipts.append(receipt)
        text = content.strip()
        if chosen_passes_filter(text):
            return text, receipts
    raise RuntimeError(f"{case['id']}: all {max_attempts} teacher samples failed the rule filter")


def build_pairs(
    cases: list[dict[str, Any]],
    *,
    teacher: tuple[Any, dict[str, Any], str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build preference pairs and return (pairs, teacher receipts)."""
    pairs: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for case in cases:
        prompt = build_prompt(case)
        rejected = case["premature_claim"]
        if not has_completion_claim(rejected):
            raise ValueError(f"{case['id']}: invalid data; premature_claim contains no completion claim")
        if teacher:
            chosen, case_receipts = teacher_chosen(case, *teacher)
            receipts.extend(case_receipts)
            source = "teacher"
        else:
            chosen = deterministic_chosen(case)
            source = "deterministic"
        if not chosen_passes_filter(chosen):
            raise ValueError(f"{case['id']}: chosen failed the rule filter: {chosen}")
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"id": case["id"], "category": case["category"], "chosen_source": source},
        })
    return pairs, receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", action="store_true", help="Generate chosen responses with a teacher model (requires an API key)")
    parser.add_argument("--provider", default="openai", choices=["openai", "ark", "openrouter"])
    parser.add_argument("--model", default=None, help="Teacher model name (defaults depend on provider)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cases for debugging")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cases = load_bad_cases()[: args.limit] if args.limit else load_bad_cases()

    teacher = None
    if args.teacher:
        from llm_client import default_model, make_client

        client, backend = make_client(args.provider)
        teacher = (client, backend, args.model or default_model(args.provider))

    pairs, receipts = build_pairs(cases, teacher=teacher)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} preference pairs -> {out_path}")

    if receipts:
        from llm_client import save_evidence

        run = datetime.now(timezone.utc).strftime("build_%Y%m%dT%H%M%SZ")
        evidence_path = save_evidence(
            run, receipts,
            extra={"pair_count": len(pairs), "chosen_source": "teacher", "model": teacher[2]},
        )
        print(f"Teacher-call evidence receipts -> {evidence_path}")


if __name__ == "__main__":
    main()
