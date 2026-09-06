"""Optional RL path: GRPO with hidden acceptance tests as rewards (Experiment 8-17).

For each end-to-end task, a completion claim triggers workspace restoration
in an isolated temporary directory and execution of the task's hidden checks:
- Claims completion and passes hidden tests: +1
- Claims completion but fails hidden tests: -1
- Performs verification without claiming completion: +0.3
- Otherwise: 0

Hidden tests are defined in data/hidden_tests.json. This optional script is
runnable but needs a GPU and costs more to train; DPO remains the main path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from evaluate import has_completion_claim, has_verification_action

ROOT = Path(__file__).resolve().parent
HIDDEN_TESTS_PATH = ROOT / "data" / "hidden_tests.json"

REWARD_CLAIM_PASS = 1.0
REWARD_CLAIM_FAIL = -1.0
REWARD_VERIFY = 0.3


def load_hidden_tasks(path: Path = HIDDEN_TESTS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_hidden_check(task: dict[str, Any], workdir: Path) -> bool:
    """Restore the workspace in a temporary directory and return whether hidden checks pass."""
    for rel_path, content in task["workspace_files"].items():
        target = workdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    try:
        result = subprocess.run(
            task["hidden_check"], shell=True, cwd=workdir,
            capture_output=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def hidden_test_reward(completion: str, task: dict[str, Any]) -> float:
    """Reward completion claims by hidden-test results and give verification a small reward."""
    claimed = has_completion_claim(completion)
    if not claimed:
        return REWARD_VERIFY if has_verification_action(completion) else 0.0
    with tempfile.TemporaryDirectory(prefix="grpo-hidden-") as tmp:
        passed = run_hidden_check(task, Path(tmp))
    return REWARD_CLAIM_PASS if passed else REWARD_CLAIM_FAIL


def build_dataset(tasks: list[dict[str, Any]]):
    """Build a GRPO dataset with prompts for the model and task IDs for the reward function."""
    from datasets import Dataset

    rows = [{
        "prompt": f"Task: {t['task']}\n\nComplete this task and state your conclusion at the end.",
        "task_id": t["id"],
    } for t in tasks]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "grpo_adapter"))
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument("--num-generations", type=int, default=8, help="Number of samples per prompt")
    args = parser.parse_args()

    tasks = load_hidden_tasks()
    task_by_id = {t["id"]: t for t in tasks}

    def reward_func(completions, task_id, **kwargs):
        """TRL GRPO reward callback; task_id is passed from the dataset as a keyword argument."""
        return [hidden_test_reward(c, task_by_id[tid]) for c, tid in zip(completions, task_id)]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=args.num_generations,
        max_completion_length=512,
        bf16=True,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=build_dataset(tasks),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"GRPO adapter saved to {args.output_dir} (optional path output)")


if __name__ == "__main__":
    main()
