"""Offline end-to-end teaching demo for Experiment 8-17.

No API key or GPU is required. Demonstrate the complete pipeline:
1. Build DPO preference pairs from premature-completion failure cases offline.
2. Display two example preference pairs.
3. Demonstrate metrics with mock evaluation of preset base/adapter outputs.
4. Print a pipeline summary and the next steps for real training.
"""

from __future__ import annotations

import json

from build_preference_data import build_pairs, load_bad_cases
from evaluate import compute_metrics, load_eval_items, mock_outputs


def main() -> None:
    print("=" * 60)
    print("Experiment 8-17: Fixing premature completion with DPO - offline end-to-end demo")
    print("=" * 60)

    # Step 1: Build preference pairs deterministically without an API
    cases = load_bad_cases()
    pairs, _ = build_pairs(cases)
    print(f"\n[1] Built {len(pairs)} DPO preference pairs from {len(cases)} failure cases")

    # Step 2: Display examples
    print("\n[2] Example preference pairs (first 2):")
    for pair in pairs[:2]:
        meta = pair["meta"]
        print(f"\n  --- {meta['id']} ({meta['category']})---")
        print("  prompt (truncated):")
        print("    " + pair["prompt"].splitlines()[0])
        print(f"  chosen  : {pair['chosen']}")
        print(f"  rejected: {pair['rejected']}")

    # Step 3: Demonstrate metrics with mock evaluation
    items = load_eval_items()
    print(f"\n[3] Evaluation set: boundary {sum(1 for i in items if i['split'] == 'boundary')} examples, "
          f"retention {sum(1 for i in items if i['split'] == 'retention')} examples (disjoint from training data)")
    for variant in ("base", "adapter"):
        metrics = compute_metrics(items, mock_outputs(variant, items))
        b, r = metrics["boundary"], metrics["retention"]
        print(f"  [{variant:7s}] boundary premature-claim rate {b['premature_claim_rate']:.0%}"
              f" | retention proper-completion rate {r['proper_completion_rate']:.0%}"
              f" | overcorrection rate {r['overcorrection_rate']:.0%}")

    # Step 4: Summary
    print("\n[4] Pipeline summary:")
    print("  Demonstrated offline: failure cases -> preference pairs -> evaluation metrics")
    print("  Real pipeline still to run (requires a GPU / API key):")
    print("    python build_preference_data.py --teacher --provider ark   # Generate chosen responses with a teacher model")
    print("    python train_dpo.py                                         # Single-GPU LoRA DPO training")
    print("    python evaluate.py --base-only                              # Baseline evaluation")
    print("    python evaluate.py                                          # Compare base vs base+adapter")
    print("    python train_grpo_optional.py                               # Optional RL path")
    print("\nExpected after training: boundary premature-claim rate decreases; retention proper-completion rate holds.")
    print("Report actual numbers only after running real training and evaluation; do not fabricate them.")


if __name__ == "__main__":
    main()
