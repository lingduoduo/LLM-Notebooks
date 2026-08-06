"""Global model and pricing configuration.

Prices are expressed in US dollars per million tokens. Prompt caching is
automatic; cached input is reported in ``prompt_tokens_details.cached_tokens``.
The client prefers OpenAI credentials and supports OpenRouter as a compatible
fallback, including model-ID mapping.
"""

import os
from dataclasses import dataclass

# Model used by the experiment; COST_DEMO_MODEL or --model can override it.
MODEL = os.environ.get("COST_DEMO_MODEL", "gpt-5.6-luna")

# OpenAI-compatible OpenRouter fallback endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _to_openrouter_model(model: str) -> str:
    """Map a model name to an OpenRouter model ID."""
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return "openai/" + model
    if model.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    return "openai/gpt-5.6-luna"


def make_client_and_model(model: str):
    """Build a compatible client and return ``(client, resolved_model)``."""
    from openai import OpenAI

    primary = os.environ.get("OPENAI_API_KEY", "").strip()
    orkey = os.environ.get("OPENROUTER_API_KEY", "").strip()
    prefer_openrouter = bool(orkey) and model.startswith("gpt-5")

    if not prefer_openrouter and primary:
        return OpenAI(timeout=60.0, max_retries=2), model
    if orkey:
        return (
            OpenAI(base_url=OPENROUTER_BASE_URL, api_key=orkey,
                   timeout=60.0, max_retries=2),
            _to_openrouter_model(model),
        )
    if primary:
        return OpenAI(timeout=60.0, max_retries=2), model
    raise RuntimeError(
        "No credentials found. Set OPENAI_API_KEY for OpenAI or "
        "OPENROUTER_API_KEY for the OpenRouter fallback, or use --offline."
    )

# USD per million tokens (gpt-4o-mini defaults).
PRICE_INPUT_PER_M = 0.15
PRICE_CACHED_PER_M = 0.075
PRICE_OUTPUT_PER_M = 0.60


@dataclass(frozen=True)
class Pricing:
    """A set of prices in USD per million tokens."""
    input_per_m: float
    cached_per_m: float
    output_per_m: float

    def cost_usd(self, prompt_tokens: int, cached_tokens: int,
                 completion_tokens: int) -> float:
        """Calculate cost in USD from prompt, cached, and completion tokens."""
        uncached_input = max(prompt_tokens - cached_tokens, 0)
        return (
            uncached_input / 1_000_000 * self.input_per_m
            + cached_tokens / 1_000_000 * self.cached_per_m
            + completion_tokens / 1_000_000 * self.output_per_m
        )


# Pricing presets in USD per million tokens.
PRICING_PRESETS = {
    "gpt-4o-mini": Pricing(0.15, 0.075, 0.60),
    "gpt-4o":      Pricing(2.50, 1.25, 10.00),
    "gpt-4.1-mini": Pricing(0.40, 0.10, 1.60),
    "gpt-4.1":     Pricing(2.00, 0.50, 8.00),
}


def default_pricing() -> Pricing:
    """Return pricing for MODEL, falling back to the module-level defaults."""
    return PRICING_PRESETS.get(
        MODEL, Pricing(PRICE_INPUT_PER_M, PRICE_CACHED_PER_M, PRICE_OUTPUT_PER_M)
    )


def cost_usd(prompt_tokens: int, cached_tokens: int, completion_tokens: int,
             pricing: "Pricing | None" = None) -> float:
    """Calculate token cost in USD, optionally using custom pricing."""
    p = pricing or Pricing(PRICE_INPUT_PER_M, PRICE_CACHED_PER_M, PRICE_OUTPUT_PER_M)
    return p.cost_usd(prompt_tokens, cached_tokens, completion_tokens)
