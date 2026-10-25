"""Optional CPU MiniLM encoder that never downloads model files."""

import hashlib
from importlib.metadata import version
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cached_model(model_path=None) -> Path:
    """Resolve the pinned MiniLM snapshot or an explicit complete local folder."""
    if model_path is None:
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                token=False,
            )
        except (ImportError, OSError, ValueError) as error:
            raise ValueError(
                f"Pinned MiniLM is not available locally: {error}. Supply --model-path to a complete local MiniLM folder; this command never downloads weights."
            ) from None
    root = Path(model_path).expanduser().resolve()
    required = ("model.safetensors", "config.json", "tokenizer.json")
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise ValueError(
            "Local MiniLM folder requires model.safetensors, config.json and tokenizer.json"
        )
    return root


class CachedMiniLMEncoder:
    """Attention-masked mean pooling + L2 normalization, pinned cache-only weights."""

    def __init__(self, model_path=None, *, batch_size=16, max_length=256, threads=2):
        if any(
            type(value) is not int or value <= 0
            for value in (batch_size, max_length, threads)
        ):
            raise ValueError(
                "batch_size, max_length and threads must be positive integers"
            )
        root = resolve_cached_model(model_path)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ValueError(
                f"Local inference needs optional torch, transformers and huggingface_hub dependencies: {error}"
            ) from None
        torch.set_num_threads(threads)
        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .to("cpu")
            .eval()
        )
        self.metadata = {
            "model_id": MODEL_ID if model_path is None else "local_model_folder",
            "revision": MODEL_REVISION if model_path is None else None,
            "artifacts_sha256": {
                name: sha256_file(root / name)
                for name in (
                    "model.safetensors",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                    "added_tokens.json",
                )
                if (root / name).is_file()
            },
            "pooling": "attention_masked_mean",
            "normalization": "L2",
            "device": "cpu",
            "max_length": max_length,
            "batch_size": batch_size,
            "threads": threads,
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "huggingface-hub")
            },
        }

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
        return vectors
