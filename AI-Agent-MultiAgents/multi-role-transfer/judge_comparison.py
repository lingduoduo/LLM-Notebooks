#!/usr/bin/env python3
"""Blind, position-swapped quality review for retained Experiment 10-1 pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


JUDGE_PROMPT = """You are an independent quality reviewer. Evaluate only the user task and the two anonymous candidate answers for:
factual/calculation correctness, compliance with all constraints, auditability, and clarity. Do not infer hidden reasoning or
trust claims of tool use without evidence in the answer; missing evidence counts as missing. Output exactly three lines:
WINNER: A or B or TIE
SCORE_A: integer from 0 to 4
SCORE_B: integer from 0 to 4

User task:
{task}

Candidate A:
{answer_a}

Candidate B:
{answer_b}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--request-timeout", type=float, default=60.0)
    return p.parse_args()


def position_swap_flags(rng: random.Random) -> tuple[bool, bool]:
    """Return the swap flag for each of a pair's two repeats.

    The pair is judged twice and the two repeats must place the arms in
    opposite positions, otherwise the repeat controls for nothing.  Only the
    arm that leads is random, so draw once per pair and invert it for the
    second repeat; drawing inside the repeat loop leaves the two repeats
    independent and silently unswapped about half the time.
    """
    first = bool(rng.randrange(2))
    return first, not first


def parse_judgment(text: str) -> dict[str, Any]:
    winner = re.search(r"WINNER\s*:\s*(A|B|TIE)", text, re.I)
    scores = [re.search(rf"SCORE_{name}\s*:\s*([0-4])", text, re.I) for name in ("A", "B")]
    return {
        "winner": winner.group(1).upper() if winner else None,
        "score_a": int(scores[0].group(1)) if scores[0] else None,
        "score_b": int(scores[1].group(1)) if scores[1] else None,
        "parse_ok": bool(winner and all(scores)),
    }


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("quality judge requires OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    tasks = {str(item["id"]): item["prompt"] for item in campaign["tasks"]}
    by_pair: dict[str, dict[str, dict]] = {}
    for run in campaign["runs"]:
        by_pair.setdefault(str(run["pair_id"]), {})[run["path"]] = run
    rng = random.Random(101)
    pairs = []
    for pair_id, arms in sorted(by_pair.items()):
        if set(arms) != {"transfer", "skill"}:
            continue
        judgments = []
        swap_flags = position_swap_flags(rng)
        for repeat in range(2):
            swapped = swap_flags[repeat]
            transfer = arms["transfer"]
            skill = arms["skill"]
            shown = [("skill", skill), ("transfer", transfer)] if swapped else [("transfer", transfer), ("skill", skill)]
            prompt = JUDGE_PROMPT.format(
                task=tasks[str(transfer["task_id"])],
                answer_a=shown[0][1].get("final_answer", ""),
                answer_b=shown[1][1].get("final_answer", ""),
            )
            kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 300,
            }
            started = datetime.now(timezone.utc).isoformat()
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "temperature" not in str(exc).lower():
                    raise
                kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            parsed = parse_judgment(content)
            judgments.append({
                "repeat": repeat + 1,
                "shown_order": [shown[0][0], shown[1][0]],
                "request": kwargs,
                "response": response.model_dump(mode="json"),
                "response_id": getattr(response, "id", None),
                "captured_at": started,
                "judgment": parsed,
            })
        # Convert each position-swapped judgment back to the architecture labels.
        normalized = []
        for item in judgments:
            winner = item["judgment"]["winner"]
            if winner == "TIE":
                normalized.append("tie")
            elif winner:
                normalized.append(item["shown_order"][0 if winner == "A" else 1])
            else:
                normalized.append(None)
        pairs.append({
            "pair_id": pair_id,
            "task_id": arms["transfer"]["task_id"],
            "transfer_deterministic_pass": bool(arms["transfer"]["outcome"]["pass"]),
            "skill_deterministic_pass": bool(arms["skill"]["outcome"]["pass"]),
            "judgments": judgments,
            "normalized_winners": normalized,
            "parse_complete": all(item["judgment"]["parse_ok"] for item in judgments),
        })
    all_judgments = [item for pair in pairs for item in pair["judgments"]]
    output = {
        "schema_version": 1,
        "experiment": "10-1",
        "judge_model": args.model,
        "judge_base_url": args.base_url,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_n": len(pairs),
        "position_swapped_repeats": 2,
        "judge_receipt_count": len(all_judgments),
        "unique_response_ids": len({item.get("response_id") for item in all_judgments}),
        "parse_complete": all(item["parse_complete"] for item in pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}: {len(pairs)} pairs, {len(all_judgments)} swapped judgments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
