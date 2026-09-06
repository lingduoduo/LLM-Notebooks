"""DPO training for Experiment 8-17 (main path; requires one GPU).

Single-GPU defaults: bf16, gradient checkpointing, LoRA r=16/alpha=32,
per-device batch size 1, gradient accumulation 2, learning rate 3e-5,
beta 0.1, and 4 epochs. Low accumulation ensures enough updates on the small
dataset. Override with --epochs, --gradient-accumulation, and --learning-rate.
Save only the LoRA adapter to output/adapter/ and a training receipt with
configuration, data hash, and timestamps to validation/<run>/training_receipt.json.

--smoke checks data loading, tokenization, and one forward pass without training.
It works without a GPU but requires downloading a small model; see --smoke-model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "preference_pairs.jsonl"
ADAPTER_DIR = ROOT / "output" / "adapter"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pairs(path: Path):
    """Load DPO pairs into a datasets.Dataset with prompt/chosen/rejected columns."""
    from datasets import Dataset

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No preference pairs in {path}; run python build_preference_data.py first")
    return Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])


def smoke(args) -> None:
    """Check data loading, tokenization, and one forward pass without training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_pairs(Path(args.data))
    print(f"Data loaded: {len(dataset)} preference pairs, columns = {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.smoke_model)
    text = dataset[0]["prompt"] + dataset[0]["chosen"]
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Tokenizer OK: first sample has {inputs['input_ids'].shape[-1]} tokens")

    model = AutoModelForCausalLM.from_pretrained(args.smoke_model)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Forward pass OK: logits shape = {tuple(outputs.logits.shape)}")
    print("Smoke check passed (no training performed).")


def train(args) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    data_path = Path(args.data)
    dataset = load_pairs(data_path)
    started = datetime.now(timezone.utc)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        beta=0.1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        max_length=1536,
        seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    adapter_dir = Path(args.output_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    finished = datetime.now(timezone.utc)

    run = started.strftime("train_%Y%m%dT%H%M%SZ")
    receipt_dir = ROOT / "validation" / run
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "experiment": "8-17 premature-completion-dpo",
        "run": run,
        "model": args.model,
        "data": str(data_path.relative_to(ROOT)),
        "data_sha256": sha256_file(data_path),
        "pair_count": len(dataset),
        "config": {
            "beta": 0.1, "learning_rate": args.learning_rate, "epochs": args.epochs,
            "lora_r": 16, "lora_alpha": 32,
            "per_device_batch_size": 1, "gradient_accumulation": args.gradient_accumulation,
            "bf16": True, "gradient_checkpointing": True, "seed": args.seed,
        },
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "training_loss": next(
            (entry.get("loss") for entry in reversed(trainer.state.log_history)
             if entry.get("loss") is not None),
            None,
        ),
    }
    (receipt_dir / "training_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "validation" / "latest.json").write_text(
        json.dumps({"run": run, "training_receipt": f"validation/{run}/training_receipt.json"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Adapter saved to {adapter_dir}; training receipt -> {receipt_dir / 'training_receipt.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model (override with --model)")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to preference_pairs.jsonl")
    parser.add_argument("--output-dir", default=str(ADAPTER_DIR), help="Adapter output directory")
    parser.add_argument("--seed", type=int, default=717)
    parser.add_argument("--epochs", type=float, default=4,
                        help="Training epochs; default 4 for enough updates on the small demo dataset")
    parser.add_argument("--gradient-accumulation", type=int, default=2,
                        help="Gradient accumulation steps; default 2 for the 24 demo examples")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="LoRA learning rate; default 3e-5 for the small demo dataset")
    parser.add_argument("--smoke", action="store_true", help="Check data, tokenization, and a forward pass without training")
    parser.add_argument("--smoke-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Small model for smoke mode (supports a CPU forward pass)")
    args = parser.parse_args()

    if args.smoke:
        smoke(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
