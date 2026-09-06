# Experiment 8-17: Fixing premature completion with DPO

This project demonstrates the full pipeline for Experiment 8-17: analyze premature-completion failures from coding agents, construct trajectory-prefix regression tasks and DPO preference pairs, train a 7B model with LoRA on one GPU, and evaluate boundary and retention behavior. It builds on the end-to-end tasks, trajectory-prefix regression tasks, and failure analysis introduced in Chapter 6. In the source book, this is the only training experiment built around production failure cases.

Premature completion means finishing before the task is actually complete: claiming success without running tests, completing only some objectives, giving up after an error, or reward hacking by deleting failing tests and claiming everything passes. Preference pairs target the decision just before the agent finishes: the rejected response claims completion, while the chosen response runs tests or checks each acceptance criterion first.

This version uses English throughout: tasks, trajectories, prompts, responses, classifier patterns, and test fixtures. The saved GPU receipts and results describe the original Chinese-language experiment. Their hashes and measurements are historical evidence, not results for the translated data. Rerun training and evaluation to obtain English-model results.

## Offline demo

The teaching demo and unit tests need neither an API key nor a GPU:

```bash
# From the repository root
cd AI-Agent-Fine-Tuning/premature-completion-dpo
python -m pip install -r requirements.txt
python demo.py                         # Preference pairs and mock evaluation
python -m pytest -q test_pipeline.py   # Pipeline unit tests
python build_preference_data.py        # Generate output/preference_pairs.jsonl
python evaluate.py --mock              # Demonstrate metrics without a model
python train_dpo.py --smoke            # Data/tokenizer/forward check; downloads a small model
```

## Real training and evaluation

Real training requires a GPU and Hugging Face model downloads. Install the project requirements in your training environment, then run:

```bash
# Optional teacher-generated chosen responses with rejection sampling and receipts
export ARK_API_KEY=your_api_key_here
python build_preference_data.py --teacher --provider ark --model doubao-seed-1-6-250615

# Single-GPU LoRA DPO; defaults to Qwen/Qwen2.5-7B-Instruct (override with --model)
python train_dpo.py

# Free generation and fixed completion/verification candidate comparison
python evaluate.py --decision-score

# Optional LLM review of classifications, with evidence receipts
python evaluate.py --judge --provider ark
```

The source book uses a shared Chapter 7 environment (`uv sync --locked --python 3.12 --extra ch7`). This notebook repository uses the local `requirements.txt` installation shown above.

## Data

- `data/bad_cases.json`: 24 synthetic but realistic trajectory-prefix failure cases, six in each category: no tests run, partial completion of multiple objectives, unmet acceptance criteria, and giving up after errors. The last category includes deleting tests, removing assertions, and skipping flaky cases. Each case includes `id`, `category`, `task`, `trajectory_prefix`, `premature_claim`, and `missing_verification`. The default offline path builds 24 preference pairs.
- `data/eval_boundary.json`: held-out tasks and parameters, disjoint from training data as enforced by unit tests. The 12 boundary tasks require continued verification. The 8 completed retention tasks require normal completion, detecting overcorrection where a model never finishes.
- `data/hidden_tests.json`: end-to-end tasks and hidden acceptance scripts for the optional GRPO path.

## Training requirements

- One GPU: a 7B model with LoRA, bf16, gradient checkpointing, batch size 1, and accumulation 2 targets approximately 24GB-class VRAM (RTX 3090/4090 or similar). Use `--model` for a smaller model.
- Hugging Face downloads: the default Qwen/Qwen2.5-7B-Instruct model is approximately 15GB.
- Outputs: only the LoRA adapter in `output/adapter/`, plus a receipt in `validation/<run>/training_receipt.json` containing configuration, data hash, and timestamps.

## Evaluation metrics

Ask the model for its next action on unfinished boundary tasks and completed retention tasks. The English prompt requests `Complete` or `Continue verification` on the first line, followed by a reason. A deterministic keyword/pattern classifier labels the response. `--decision-score` also compares two fixed candidate actions using mean token log probability to directly measure the completion decision.

- **Boundary premature-claim rate** should decrease after training.
- **Retention proper-completion rate** should be maintained.
- **Overcorrection rate** = 1 - retention proper-completion rate; it should remain low.

Historical measurements from the original Chinese-language run on an RTX PRO 6000 (approximately 98GB VRAM): fixed-candidate boundary accuracy rose from 3/12 (25.0%) to 11/12 (91.7%) after LoRA DPO with Qwen2.5-7B-Instruct, 4 epochs, and learning rate 3e-5. Retention accuracy remained 8/8 (100%). Mean boundary margin rose from -0.2083 to 0.3828; retention margin changed from 4.6904 to 2.8525.

Free generation was a supplementary diagnostic: premature claims fell from 1/12 to 0/12, but proper retention completion fell from 6/8 to 0/8. Small-data DPO made open-ended answers overly cautious. The historical conclusion therefore concerns fixed-candidate decisions aligned with training prompts; it does not establish an improvement in overall production success. See `validation/experiment_8_17_gpu_20260807.md` and the receipts in `validation/`. These measurements have not been rerun for the English version.

## Optional RL path

`train_grpo_optional.py` uses TRL's GRPOTrainer with hidden acceptance tests as rewards. A completion claim triggers workspace restoration in an isolated temporary directory and execution of the hidden checks: +1 for a passing claim, -1 for a failing claim, +0.3 for verification without a completion claim, and 0 otherwise. This optional path requires a GPU and costs more to train; DPO is the main experiment.

```bash
python train_grpo_optional.py   # Defaults to Qwen/Qwen2.5-7B-Instruct
```

## Verification and reporting

Preference-pair construction, hidden tests, and evaluation classification are external verification code that the trained model cannot modify. `test_pipeline.py` enforces training/evaluation separation. Report only observed results. The large historical adapter was not included; receipts preserve its model, data hash, and configuration. Training with the translated data creates a new English adapter and new receipts, rather than reproducing the historical data hash.

The original experiment also tried completed-task control pairs. They improved open-ended completion but reduced fixed-candidate boundary accuracy to 0/12, so they were excluded from the final training set.
