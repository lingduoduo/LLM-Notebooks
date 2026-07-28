"""Universal OpenRouter fallback for the collaboration tools' LLM clients.

Every LLM entry point in this experiment (sub-agent runs, intelligence tools,
browser-use) speaks the OpenAI-compatible API. This helper centralizes the
credential resolution so that:

  1. When OPENAI_API_KEY is present, behavior is unchanged (direct OpenAI, or a
     custom OPENAI_BASE_URL / OPENAI_MODEL if the user set them).
  2. When OPENAI_API_KEY is absent but OPENROUTER_API_KEY is present, requests
     transparently route through OpenRouter (base_url=https://openrouter.ai/api/v1)
     with the model id mapped to provider/model form.
  3. When neither is set, callers can detect "offline" and fall back to their
     deterministic mock paths (no fabricated model output).
"""

import os
from typing import Dict, Optional, Tuple


def map_model_for_openrouter(model: str) -> str:
    """Map a plain model id onto OpenRouter's `provider/model` form.

    Ids already containing "/" pass through unchanged; gpt-*/o1-*/o3-*/o4-*
    become openai/…; claude-* becomes anthropic/claude-opus-4.8.
    """
    if "/" in model:
        return model
    m = model.lower()
    if m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    if m.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    if m.startswith("kimi"):
        return "moonshotai/kimi-k2.6"
    return model


def has_llm() -> bool:
    """True when at least one usable LLM credential is configured."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def resolve_llm(default_model: str = "gpt-5.6-luna") -> Tuple[str, Optional[str], str]:
    """Resolve (api_key, base_url, model), applying the OpenRouter fallback.

    Raises RuntimeError listing the accepted keys when neither credential is set.
    """
    model = os.getenv("OPENAI_MODEL", default_model)

    or_key = os.getenv("OPENROUTER_API_KEY")
    # gpt-5.x (incl. gpt-5.6*) needs OpenAI org-verification on the direct API;
    # when an OpenRouter key is present, prefer routing these ids through it.
    if or_key and model.lower().startswith("gpt-5"):
        return or_key, "https://openrouter.ai/api/v1", map_model_for_openrouter(model)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key, os.getenv("OPENAI_BASE_URL"), model

    if or_key:
        return or_key, "https://openrouter.ai/api/v1", map_model_for_openrouter(model)

    raise RuntimeError(
        "No LLM key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
        "(universal fallback)."
    )


# ---------------------------------------------------------------------------
# Reasoning-model request shaping
#
# The default model here is a GPT-5 class reasoning model, and OpenAI's chat
# completions endpoint rejects two parameters for that family that every older
# model accepted. Both rejections are hard 400s, so a request carrying them
# never reaches the model at all. Mirrors execution-tools/llm_helper.py.
# ---------------------------------------------------------------------------

def is_reasoning_model(model) -> bool:
    """Whether the model belongs to the reasoning family (GPT-5, o-series).

    Normalizes "/" to "-" so OpenRouter ids (``openai/gpt-5.6-luna``) are
    classified the same as their plain counterparts.
    """
    m = str(model or "").lower().replace("/", "-")
    return m.startswith(("o1", "o3", "o4")) or "gpt-5" in m


def reasoning_safe_temperature(model, requested: float = 1.0):
    """Reasoning models only accept temperature=1.

    Returns 1 for those; otherwise the requested value, so models that do honor
    a temperature keep their tuned setting.
    """
    return 1 if is_reasoning_model(model) else requested


def token_limit_parameter(model, base_url: Optional[str] = None) -> str:
    """Name of the output-token cap parameter to send for this model/route.

    Reasoning models reject ``max_tokens`` on the direct OpenAI API:

        Unsupported parameter: 'max_tokens' is not supported with this model.
        Use 'max_completion_tokens' instead.

    OpenRouter normalizes ``max_tokens`` for every model it proxies, so the
    rename is applied only on the direct route and the fallback route keeps the
    parameter name OpenRouter documents.
    """
    via_openrouter = bool(base_url and "openrouter.ai" in base_url)
    if is_reasoning_model(model) and not via_openrouter:
        return "max_completion_tokens"
    return "max_tokens"


def token_limit_kwargs(model, limit: int, base_url: Optional[str] = None) -> Dict[str, int]:
    """Keyword argument carrying the output-token cap for this model/route."""
    return {token_limit_parameter(model, base_url): limit}
