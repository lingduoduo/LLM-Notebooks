"""
Quantified "before vs after" comparison for prompt distillation.

This script answers the core question of experiments 7-8: after distilling a
"long prompt + thinking teacher" into a "no prompt, direct answer student", how
much did we actually save, and how much quality did we lose? It builds a real
before/after table from real data **without loading any large model and without
network access**:

  1. Input cost (tokens): every teacher call has to carry the full language
     classification prompt (~1k tokens), while the student only sees the raw text
     to classify. The token difference is the input overhead saved per call.
  2. Task quality: read evaluation_results.json produced by evaluate.py to get
     the student's agreement rate with the teacher's labels on the same inputs
     (i.e. distillation fidelity).
  3. Per-case table: pick a few real samples and show teacher tokens / student
     tokens / teacher label / student prediction / match side by side, so the
     before/after is visible across several concrete cases.

Design principle: every number comes from real data and a real tokenizer, none
are invented. Latency (wall-clock response time) has to be measured on a GPU, so
this script does not estimate it; it only reports the token cost and quality that
can be reproduced offline.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


VALID_LABELS = ["ar", "de", "el", "en", "es", "fr", "hi", "ru", "tr", "ur", "vi", "zh", "ot"]


def load_prompt_template(source_file: str) -> str:
    """Extract the teacher's language classification prompt template from create_data.py (avoids importing vllm)."""
    src = Path(source_file).read_text(encoding="utf-8")
    match = re.search(
        r'LANGUAGE_CLASSIFICATION_PROMPT\s*=\s*"""(.*?)"""',
        src,
        re.DOTALL,
    )
    if not match:
        raise ValueError(
            f"Could not find the LANGUAGE_CLASSIFICATION_PROMPT template in {source_file}; "
            f"use --prompt_source to point at the file that defines this constant."
        )
    return match.group(1)


def build_token_counter(tokenizer_name: Optional[str]) -> Tuple[Callable[[str], int], str]:
    """
    Build a token counting function with a priority-ordered fallback so it always works offline.

    Returns (counter, method_description):
      1) If --tokenizer is given, count exactly with a HuggingFace tokenizer (real Qwen token
         counts on a GPU machine).
      2) Otherwise approximate with tiktoken's o200k_base (the GPT-4o/o1 tokenizer), which is
         reproducible offline.
      3) Finally fall back to a rough "characters / 4" heuristic, clearly labeled as an estimate.
    Every method is named in the output; an approximation is never reported as an exact count.
    """
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return (lambda s: len(tok.encode(s))), f"HuggingFace tokenizer (exact): {tokenizer_name}"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] Could not load tokenizer {tokenizer_name} ({exc}); falling back to tiktoken.",
                file=sys.stderr,
            )

    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken o200k_base (approximate, offline-reproducible)"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tiktoken unavailable ({exc}); falling back to a character heuristic.", file=sys.stderr)

    return (lambda s: max(1, len(s) // 4)), "characters/4 (rough estimate)"


def load_texts(test_file: str) -> List[str]:
    with open(test_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_teacher_labels(train_data_file: str) -> Dict[str, str]:
    """Read the text -> teacher label mapping from the distillation training data (teacher annotations)."""
    mapping: Dict[str, str] = {}
    if not Path(train_data_file).exists():
        return mapping
    with open(train_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            if len(msgs) >= 2:
                mapping[msgs[0].get("content", "")] = msgs[1].get("content", "")
    return mapping


def load_eval_results(eval_file: str) -> Optional[Dict]:
    if not Path(eval_file).exists():
        return None
    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate(text: str, width: int = 42) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def compare(
    prompt_template: str,
    texts: List[str],
    teacher_labels: Dict[str, str],
    eval_results: Optional[Dict],
    count_tokens: Callable[[str], int],
    token_method: str,
    num_examples: int,
) -> Dict:
    n = len(texts)

    # Fixed prompt overhead (the template itself, excluding the text to classify)
    fixed_overhead = count_tokens(prompt_template.format(text=""))

    teacher_input_total = 0
    student_input_total = 0
    per_text_tokens: List[Tuple[int, int]] = []  # (teacher_tokens, student_tokens)
    for text in texts:
        teacher_prompt = prompt_template.format(text=text)
        t_tok = count_tokens(teacher_prompt)
        s_tok = count_tokens(text)
        teacher_input_total += t_tok
        student_input_total += s_tok
        per_text_tokens.append((t_tok, s_tok))

    if n == 0:
        teacher_avg = student_avg = reduction_pct = 0.0
        ratio = float("inf")
    else:
        teacher_avg = teacher_input_total / n
        student_avg = student_input_total / n
        reduction_pct = (
            100.0 * (1 - student_input_total / teacher_input_total)
            if teacher_input_total
            else 0.0
        )
        ratio = (
            teacher_input_total / student_input_total
            if student_input_total
            else float("inf")
        )

    # Student predictions (aligned line by line with test_file)
    student_preds: Optional[List[Optional[str]]] = None
    accuracy = None
    correct = evaluated = None
    if eval_results:
        student_preds = eval_results.get("predictions")
        summary = eval_results.get("summary", {})
        accuracy = summary.get("accuracy")
        correct = summary.get("correct")
        evaluated = summary.get("evaluated")

    # Per-case table: prefer covering different languages, and try to include both
    # matching and mismatching examples
    examples: List[Dict] = []
    seen_labels = set()
    for idx, text in enumerate(texts):
        teacher_label = teacher_labels.get(text, "?")
        student_pred = (
            student_preds[idx] if student_preds and idx < len(student_preds) else None
        )
        key = teacher_label
        if key in seen_labels and len(examples) >= num_examples:
            continue
        if len(examples) >= num_examples:
            break
        if key in seen_labels:
            continue
        seen_labels.add(key)
        t_tok, s_tok = per_text_tokens[idx]
        examples.append(
            {
                "text": text,
                "teacher_tokens": t_tok,
                "student_tokens": s_tok,
                "teacher_label": teacher_label,
                "student_pred": student_pred,
                "match": (student_pred == teacher_label) if student_pred else None,
            }
        )

    return {
        "num_cases": n,
        "token_method": token_method,
        "fixed_prompt_overhead": fixed_overhead,
        "teacher_input_total": teacher_input_total,
        "teacher_input_avg": teacher_avg,
        "student_input_total": student_input_total,
        "student_input_avg": student_avg,
        "input_token_reduction_pct": reduction_pct,
        "teacher_student_ratio": ratio,
        "student_accuracy": accuracy,
        "student_correct": correct,
        "student_evaluated": evaluated,
        "examples": examples,
    }


def print_report(r: Dict) -> None:
    line = "=" * 84
    print("\n" + line)
    print("Prompt Distillation: quantified before vs after comparison")
    print(line)
    print(f"Cases                 : {r['num_cases']}")
    print(f"Token counting method : {r['token_method']}")
    print(
        f"Fixed prompt overhead : {r['fixed_prompt_overhead']} tokens "
        f"(the template itself, paid again on every call)"
    )

    print("\n" + "-" * 84)
    print("1. Input cost (input tokens per call)")
    print("-" * 84)
    print(f"{'Metric':<30}{'Teacher (prompt+think)':>26}{'Student (no prompt)':>22}")
    print(f"{'Avg input tokens / call':<30}{r['teacher_input_avg']:>26.1f}{r['student_input_avg']:>22.1f}")
    print(f"{'Total input tokens':<30}{r['teacher_input_total']:>26,}{r['student_input_total']:>22,}")
    print(
        f"\n-> Input tokens reduced by {r['input_token_reduction_pct']:.1f}%"
        f" (the teacher costs {r['teacher_student_ratio']:.1f}x the student)."
    )
    print("   On an API billed per input token this lowers cost proportionally; the teacher also")
    print("   spends thinking (CoT) output tokens that are not counted here, so the real gap is")
    print("   larger. Latency must be measured on GPU and is deliberately not estimated here.")

    print("\n" + "-" * 84)
    print("2. Task quality (student agreement with the teacher labels = distillation fidelity)")
    print("-" * 84)
    if r["student_accuracy"] is not None:
        print(
            f"Teacher (reference) : 100.00%   Student (distilled) : {r['student_accuracy'] * 100:.2f}%"
            f"  ({r['student_correct']}/{r['student_evaluated']})"
        )
        print(
            f"-> Without any prompt or thinking, the student keeps about "
            f"{r['student_accuracy'] * 100:.1f}% of the teacher's judgments,"
        )
        print(
            f"   a quality loss of about {(1 - r['student_accuracy']) * 100:.1f} percentage points."
        )
    else:
        print("No evaluation_results.json found (the student has not been evaluated yet). Run")
        print("evaluate.py first, then come back for this section. Its absence does not affect")
        print("the input cost comparison above.")

    print("\n" + "-" * 84)
    print(f"3. Per-case table ({len(r['examples'])} cases)")
    print("-" * 84)
    print(f"{'Text to classify':<44}{'T-tok':>8}{'S-tok':>8}{'Teacher':>9}{'Student':>9}{'Match':>6}")
    for ex in r["examples"]:
        if ex["match"] is None:
            mark = "-"
        else:
            mark = "✓" if ex["match"] else "✗"
        pred = ex["student_pred"] if ex["student_pred"] else "-"
        print(
            f"{truncate(ex['text']):<44}"
            f"{ex['teacher_tokens']:>8}{ex['student_tokens']:>8}"
            f"{ex['teacher_label']:>9}{pred:>9}{mark:>6}"
        )
    print(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quantified before/after comparison for prompt distillation: "
        "input cost, task quality and a per-case table, computed offline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./example-data/multilingual.txt",
        help="File of texts to classify (one per line), used as the comparison input set",
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="./data/prompt_distillation_lang.jsonl",
        help="Distillation training data (teacher annotations), used as the quality reference labels",
    )
    parser.add_argument(
        "--eval_results",
        type=str,
        default="./evaluation_results.json",
        help="Evaluation results produced by evaluate.py, used to read the student agreement rate (optional)",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="./create_data.py",
        help="Source file that defines the teacher prompt template LANGUAGE_CLASSIFICATION_PROMPT",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional: HuggingFace tokenizer name/path (e.g. Qwen/Qwen3-30B-A3B-Instruct-2507). "
        "When given it is used for exact counts; otherwise tiktoken approximates so the script runs offline",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of per-case examples to show (tries to cover different languages)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: path to save the comparison result (including per-case examples) as JSON",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Text file to classify not found: {args.test_file}")
    if not os.path.exists(args.prompt_source):
        raise FileNotFoundError(f"Prompt template source file not found: {args.prompt_source}")

    prompt_template = load_prompt_template(args.prompt_source)
    texts = load_texts(args.test_file)
    teacher_labels = load_teacher_labels(args.train_data_file)
    eval_results = load_eval_results(args.eval_results)
    count_tokens, token_method = build_token_counter(args.tokenizer)

    if not teacher_labels:
        print(
            f"[warn] No teacher annotations read from {args.train_data_file}; "
            f"teacher labels in the per-case table will show as '?'.",
            file=sys.stderr,
        )
    if eval_results is None:
        print(
            f"[warn] {args.eval_results} not found; only the input cost comparison will be "
            f"reported and the quality section will be skipped.",
            file=sys.stderr,
        )

    report = compare(
        prompt_template=prompt_template,
        texts=texts,
        teacher_labels=teacher_labels,
        eval_results=eval_results,
        count_tokens=count_tokens,
        token_method=token_method,
        num_examples=args.num_examples,
    )

    print_report(report)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📁 Comparison results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
