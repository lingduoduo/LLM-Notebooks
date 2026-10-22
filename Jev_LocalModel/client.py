"""Small Jev-shaped client around the downloaded Laya checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


MODEL_DIRS = {
    "laya": "",
    "laya-english": "",
    "laya-multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
}


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LayaClient:
    """Expose a single Laya checkpoint with Jev's ``state + questions`` shape.

    Chinese and other non-English input should use the multilingual checkpoint.
    Set ``LAYA_MODEL_DIR`` to point at a different local checkpoint directory.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent / "models"
        configured = model_dir or os.environ.get("LAYA_MODEL_DIR")
        self.model_dir = Path(configured).expanduser() if configured else root / "multilingual"
        self.model_dir = self.model_dir.resolve()
        if not (self.model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"No model.safetensors in {self.model_dir}; download a Laya checkpoint or set LAYA_MODEL_DIR"
            )

        if model_name:
            self.model_name = model_name
        elif self.model_dir.name == "multilingual":
            self.model_name = "laya-multilingual"
        elif self.model_dir.name == "typed-decisions":
            self.model_name = "laya-typed-decisions"
        else:
            self.model_name = "laya"

        # The inference modules live next to the `models/` checkpoint folders;
        # each checkpoint directory itself only contains its weights/config.
        module_path = str(Path(__file__).resolve().parent / "models")
        if module_path not in sys.path:
            sys.path.insert(0, module_path)
        from rl_agent_api import RLAgent

        self.device = device or _default_device()
        self._agent = RLAgent(str(self.model_dir), device=self.device)

    def system_one(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate independent typed questions in one model call."""
        if not isinstance(state, (str, dict, list)):
            raise TypeError("state must be a string, object, or array")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        for qid, question in questions.items():
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("question IDs must be non-empty strings")
            if not isinstance(question, dict):
                raise ValueError(f"question {qid!r} must be an object")
            kind = question.get("type")
            if kind not in ("choice", "score", "noul"):
                raise ValueError(f"question {qid!r}: type must be choice, score, or noul")
            if not isinstance(question.get("instructions"), (str, dict, list)):
                raise ValueError(f"question {qid!r}: instructions must be a string, object, or array")
            criteria = question.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, (dict, list)) or not criteria:
                    raise ValueError(f"question {qid!r}: choice criteria must be a non-empty object or array")
                if any(not isinstance(key, str) or not key.strip() for key in criteria):
                    raise ValueError(f"question {qid!r}: choice labels must be non-empty strings")
                if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
                    raise ValueError(f"question {qid!r}: choice labels must be unique")
                if isinstance(criteria, dict) and any(value is not None and not isinstance(value, str)
                                                       for value in criteria.values()):
                    raise ValueError(f"question {qid!r}: choice descriptions must be strings or null")
            elif kind == "score":
                if not isinstance(criteria, list) or not criteria or any(not isinstance(c, str) for c in criteria):
                    raise ValueError(f"question {qid!r}: score criteria must be a non-empty array of strings")
            elif criteria is not None:
                if not isinstance(criteria, dict) or any(key not in ("true", "false") for key in criteria):
                    raise ValueError(f"question {qid!r}: noul criteria must contain only true and false labels")
                if any(value is not None and not isinstance(value, str) for value in criteria.values()):
                    raise ValueError(f"question {qid!r}: noul descriptions must be strings or null")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string")
        if model is not None and model not in {self.model_name, "laya-latest"}:
            raise ValueError(
                f"This server has {self.model_name!r} loaded; start another instance to use {model!r}"
            )

        result = self._agent.system_one(state, questions)
        result["model"] = self.model_name
        # Keep the public shape close to Jev; expose no training/runtime internals.
        for answer in result["answers"].values():
            answer.pop("rl_agent", None)
        return result
