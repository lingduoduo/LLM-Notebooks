"""
CoT distillation data collection script (companion code for experiments 7-9).

Method (step one, "collect trajectories", of the three-step flow in experiments 7-9):
  1. Read math problems with reference answers from problems.jsonl (a task
     distribution that a rule-based verifier can check);
  2. Call a frontier teacher model through OpenRouter (Claude by default) with
     reasoning enabled to obtain the full "thinking + answer" trajectory (the
     Claude 4 family returns summarized thinking -- a high-fidelity summary of the
     raw chain of thought produced by another model; the raw chain of thought only
     exists inside the encrypted signature field);
  3. Check the final answer with a rule-based verifier and write only the correct
     trajectories as SFT training data (messages format:
     "question -> <think>thinking</think> + final answer").

Note: this experiment only uses the reasoning/thinking capabilities exposed by each
vendor's official API to obtain chains of thought; it does not involve any means of
bypassing a vendor's safety mechanisms. The raw trajectories (including the ones
that fail verification) are saved to raw_trajectories.jsonl to make the teacher's
error patterns easy to analyze.
"""

import argparse
import asyncio
import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI

ANSWER_SUFFIX = "\n\nReason step by step, and give the final answer on the last line in the format \"Final Answer: <number>\" (the number only, no units)."

def load_problems(path: str) -> list[dict]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def extract_predicted_number(text: str) -> Optional[float]:
    """Parse the final answer value out of the model output.

    Prefers the Final Answer marker and otherwise falls back to the last number.
    The full-width colon stays in the pattern so trajectories collected with the
    earlier Chinese answer suffix still parse.
    """
    m = re.findall(r"Final Answer[:：]\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if not m:
        m = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def verify(text: str, gold: float, tol: float = 1e-6) -> bool:
    """Rule-based verifier: check the final answer against the reference answer."""
    pred = extract_predicted_number(text)
    if pred is None:
        return False
    return abs(pred - float(gold)) <= tol * max(1.0, abs(float(gold)))


def get_reasoning(message) -> str:
    """Extract the chain of thought from the returned message.

    Tries in order: OpenRouter's reasoning / reasoning_details fields, and the
    reasoning_content field used by native APIs such as Moonshot and DeepSeek.
    """
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return reasoning
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return reasoning_content
    details = getattr(message, "reasoning_details", None) or []
    parts = []
    for d in details:
        if isinstance(d, dict):
            parts.append(d.get("text") or d.get("summary") or "")
        else:
            parts.append(getattr(d, "text", None) or getattr(d, "summary", None) or "")
    return "\n".join(p for p in parts if p)


async def distill_one(client: AsyncOpenAI, problem: dict, args, semaphore) -> dict:
    """Call the teacher model on a single problem and return the full trajectory record."""
    record = {
        "id": problem["id"],
        "question": problem["question"],
        "gold_answer": problem["answer"],
        "model": args.model,
        "content": None,
        "reasoning": None,
        "verified": False,
        "usage": None,
        "error": None,
    }
    async with semaphore:
        for attempt in range(args.max_retries + 1):
            try:
                kwargs = {}
                if args.reasoning_effort:
                    # OpenRouter style: request the chain of thought by effort (adaptive-thinking
                    # models such as Claude Opus 4.8)
                    kwargs["extra_body"] = {"reasoning": {"effort": args.reasoning_effort}}
                elif args.reasoning_max_tokens:
                    # OpenRouter style: request the chain of thought by token budget (manual-budget
                    # models such as Claude Sonnet 4.5)
                    kwargs["extra_body"] = {"reasoning": {"max_tokens": args.reasoning_max_tokens}}
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": problem["question"] + args.answer_suffix}],
                        max_tokens=args.max_tokens,
                        # Raise the temperature on retries to get a different trajectory, except for
                        # models locked to temperature=1 such as Kimi K3
                        temperature=args.temperature + (0.2 * attempt if args.temperature < 1.0 else 0),
                        **kwargs,
                    ),
                    timeout=args.request_timeout,  # Hard timeout: keeps a half-open connection from hanging
                )
                msg = resp.choices[0].message
                record["content"] = msg.content or ""
                record["reasoning"] = get_reasoning(msg)
                record["usage"] = resp.usage.model_dump() if resp.usage else None
                record["verified"] = verify(record["content"], problem["answer"])
                break
            except Exception as e:
                record["error"] = f"attempt {attempt}: {type(e).__name__}: {e}"
        status = "OK" if record["verified"] else ("ERR" if record["error"] else "WRONG")
        print(f"  [{status}] {record['id']}", flush=True)
        return record


def to_sft_sample(record: dict) -> dict:
    """Turn a verified trajectory into an SFT training sample (messages format, thinking wrapped in <think> tags)."""
    if record["reasoning"]:
        assistant = f"<think>\n{record['reasoning'].strip()}\n</think>\n\n{record['content'].strip()}"
    else:
        assistant = record["content"].strip()
    return {
        "messages": [
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": assistant},
        ]
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Distill CoT trajectories from a frontier cloud model (via OpenRouter) into SFT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="./problems.jsonl", help="Problem file (JSONL with question/answer fields)")
    parser.add_argument("--sft_output", default="./data/sft_cot_distill.jsonl", help="Output path for the SFT training data")
    parser.add_argument("--raw_output", default="./data/raw_trajectories.jsonl", help="Output path for the raw trajectories (including failed samples)")
    parser.add_argument("--model", default="anthropic/claude-opus-4.8", help="Teacher model ID")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible API endpoint")
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY", help="Name of the environment variable holding the API key")
    parser.add_argument("--reasoning_effort", default="",
                        help="OpenRouter-style reasoning effort (high/medium/low); when set it takes precedence "
                             "over --reasoning_max_tokens. For models that only support adaptive thinking, "
                             "such as Claude Opus 4.8")
    parser.add_argument("--reasoning_max_tokens", type=int, default=4096,
                        help="Maximum chain-of-thought tokens (OpenRouter-style reasoning parameter); 0 = do not "
                             "send the parameter, for native APIs such as Moonshot/DeepSeek that return "
                             "reasoning_content by default")
    parser.add_argument("--max_problems", type=int, default=0, help="Maximum number of problems to process (0 = all; for debugging)")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens per reply (must exceed the reasoning tokens)")
    parser.add_argument("--max_retries", type=int, default=1, help="Maximum retries after a failure or error")
    parser.add_argument("--request_timeout", type=float, default=600, help="Per-request timeout in seconds; a timeout is retried as a failure")
    parser.add_argument("--answer_suffix", default=ANSWER_SUFFIX, help="Answer-format instruction appended to each problem")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set the {args.api_key_env} environment variable first")

    problems = load_problems(args.input)
    if args.max_problems:
        problems = problems[: args.max_problems]
    print(f"{len(problems)} problems, teacher model: {args.model} @ {args.base_url}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key, timeout=args.request_timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.sft_output), exist_ok=True)

    # Incremental flush: write each raw trajectory as soon as its problem finishes, so a hung or
    # interrupted process does not lose the results already collected
    records = []
    tasks = [distill_one(client, p, args, semaphore) for p in problems]
    with open(args.raw_output, "w", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            record = await coro
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    passed = [r for r in records if r["verified"]]
    with open(args.sft_output, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(to_sft_sample(r), ensure_ascii=False) + "\n")

    total_in = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in records)
    total_out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in records)
    n_err = sum(1 for r in records if r["error"])
    print(f"\n{'=' * 50}")
    # Empty problems JSONL yields zero records; avoid ZeroDivisionError on the rate.
    pass_rate = (len(passed) / len(records) * 100) if records else 0.0
    print(f"Verified: {len(passed)}/{len(records)} ({pass_rate:.1f}%)")
    print(f"API errors: {n_err}  No chain of thought returned: {sum(1 for r in records if not r['reasoning'])}")
    print(f"Token usage: input {total_in}, output {total_out}")
    print(f"SFT data written to: {args.sft_output}")
    print(f"Raw trajectories written to: {args.raw_output}")


if __name__ == "__main__":
    asyncio.run(main())
