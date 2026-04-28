# LLM Fine-Tuning Notebooks

A collection of notebooks and experiments for fine-tuning and running large language models locally.

## Structure

| Directory | Description |
|-----------|-------------|
| [Unsloth/](Unsloth/) | Supervised fine-tuning (SFT) notebooks using Unsloth — Gemma 4 (vision), GPT-OSS 20B, Llama 3.1 8B |
| [Mlx/](Mlx/) | Fine-tuning and inference on Apple Silicon via MLX |
| [vLLM/](vLLM/) | Inference notebooks using vLLM — Qwen 2.5 1.5B |
| [finetuning_exploration.ipynb](finetuning_exploration.ipynb) | Exploratory fine-tuning experiments |

## Model Deployment Reference

| Scenario | Recommended Format | Reason |
|----------|--------------------|--------|
| Production (dual A100 / H100) | Official FP8 | Highest fidelity, near lossless |
| Consumer single GPU (RTX 4090 / 3090) | AWQ-INT4 | Fits in ~15 GB VRAM, runs with vLLM |
| Blackwell GPUs (RTX 5090 / RTX PRO 6000) | NVFP4 | Fully utilizes FP4 compute |
| Windows / Linux PC with 24 GB VRAM | Unsloth UD-Q4_K_XL | Best dynamic quantization quality |
| LM Studio | GGUF (lmstudio-community) | Best tooling and ecosystem integration |
| Mac Studio / MacBook | MLX-4bit or NVFP4 | Full Metal acceleration via MLX |
| Low-end machine with large RAM | Unsloth UD-Q2_K_XL | Runs within ~15 GB memory |
