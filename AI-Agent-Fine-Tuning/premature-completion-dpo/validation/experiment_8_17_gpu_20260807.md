# Experiment 8-17 GPU results (2026-08-07)

This is an English translation of the report for the original Chinese-language experiment. The saved receipts, data hashes, and measurements refer to that original run. The English datasets and prompts have not been evaluated on a GPU; rerun training and evaluation to measure their results.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, approximately 98GB VRAM
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Training: LoRA DPO, bf16, 4 epochs, learning rate `3e-5`, gradient accumulation 2, seed 717
- Data: 24 preference pairs; training tasks fully disjoint from unfinished boundary and completed retention tasks
- Training receipt: `train_20260807T163349Z/training_receipt.json`

## Results

The decision-boundary evaluation compares mean token log probabilities of two fixed actions, translated here as "Complete" and "Continue verification":

| Model | Boundary correct | Retention correct | Boundary mean margin | Retention mean margin |
|---|---:|---:|---:|---:|
| Base | 3/12 (25.0%) | 8/8 (100%) | -0.2083 | 4.6904 |
| LoRA DPO | 11/12 (91.7%) | 8/8 (100%) | 0.3828 | 2.8525 |

Free generation is a supplementary diagnostic: boundary premature completion changed from 1/12 to 0/12, while proper retention completion fell from 6/8 to 0/8. Open-ended answers tended to contain long verification plans without explicitly finishing. This is why the main conclusion uses fixed candidates aligned with the training prompts instead of relying solely on free generation.

The LoRA adapter was approximately 158MB and was not included in the commit. Receipts preserve actual configuration, data hashes, and runtime. Different TRL versions may record final loss in different fields, so a single loss value is not treated as the experimental conclusion. The README commands now train on English data and produce a new adapter.

## Iterations

The first version evaluated only free generation. The base model already rarely claimed completion on these tasks, and premature-completion rates barely changed after training. Answer length and prompt adherence obscured the metric, so it could not establish whether DPO learned the completion decision.

Evaluation then switched to two fixed candidate continuations, translated as "Complete: the task is complete and ready for delivery" and "Continue verification: run acceptance tests and check each acceptance criterion." Comparing mean token log probabilities directly measures whether to finish now.

A small comparison of training strengths found little change at `5e-6`. At `5e-5`, fixed-candidate boundary accuracy reached 12/12, but open-ended answers became noticeably overcautious. The final choice of `3e-5` and 4 epochs produced 11/12 boundary accuracy while retaining 8/8 on completed tasks.

Adding 8 completed-task control pairs, then reducing them to 4, partially restored open-ended retention completion but reduced fixed-candidate boundary accuracy to 1/12 and 0/12 respectively. These controls were excluded from the final training set. The individual `training_receipt.json` files preserve this iteration history rather than presenting a single favorable parameter setting without context.

## Interpretation and limitations

This small experiment suggests that trajectory-prefix preference data can improve completion decisions while preserving correct fixed-candidate choices on completed tasks. It does not establish higher overall production success for coding agents: there are only 24 training tasks, and candidate comparison is not a full environment replay. Deployment evaluation should include more task families, real hidden acceptance tests, and general-capability regression checks. Translation changes the model inputs, so the historical measurements cannot be assumed to hold for this English version.
