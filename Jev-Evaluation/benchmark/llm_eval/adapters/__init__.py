"""Adapter registry. Per-provider configs are thin shims over
OpenAICompatAdapter; their default base URL, default model and
published pricing live here so they stay easy to update.

This fork adds `laya_local` — a thin wrapper around `TypesafeAdapter` pointing
at the local `python -m laya.serve` loopback endpoint.
"""
from __future__ import annotations

from typing import Any

from .base import AdapterError
from .laya_local import LayaLocalAdapter
from .mock import MockAdapter
from .openai_compat import OpenAICompatAdapter, is_reasoning_model
from .typesafe import TypesafeAdapter


# Provider presets. base_url is the OpenAI-compatible chat completions endpoint.
# Models default to the latest known general-purpose model; override via --model.
# Prices are USD per 1M tokens (input, output). null == unmetered.
# `mode` flags whether the model returns its OWN probability distribution
# ('native', Jev-class) or is a generic LLM we ask to WRITE one out
# ('verbalized', JSON-schema constrained). The two are labelled separately
# per the JevBench convention.
PROVIDERS: dict[str, dict[str, Any]] = {
    # ── JevBench reference row ──────────────────────────────────────────
    # Native System One — returns its own probability distribution over
    # the exact label set. Not a verbalized model.
    "jev": {
        "base_url": "https://api.typesafe.ai",   # /v1/systemone is appended by adapter
        "default_model": "jev-latest",
        "price_in_per_m": 0.042,   # $0.042 per 1M input tokens
        "price_out_per_m": 0.0,    # output tokens are free
        "mode": "native",
        "notes": "TypeSafe AI's Jev (System One). https://docs.typesafe.ai/models",
    },
    "typesafe": {  # alias kept for backward compatibility
        "base_url": "https://api.typesafe.ai",
        "default_model": "jev-latest",
        "price_in_per_m": 0.042,
        "price_out_per_m": 0.0,
        "mode": "native",
        "notes": "Alias of `jev`. See https://docs.typesafe.ai/models",
    },
    # ── OpenAI evaluation set (verbalized via JSON schema) ──────────────
    # One preset for every OpenAI model; per-model prices live in
    # OPENAI_MODELS below and are picked up by resolve_prices(model=...).
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-6-luna",
        "price_in_per_m": 0.10,
        "price_out_per_m": 0.50,
        "mode": "verbalized",
        "notes": "OpenAI Chat Completions. Pick the model with --model; see OPENAI_MODELS for the evaluated lineup.",
    },
    # ── Laya local serve ──────────────────────────────────────────────────
    # Native Laya typed-decision endpoint, exposed by `python -m laya.serve`
    # on the loopback interface. Wire format identical to Jev's /v1/systemone,
    # so this points at the same `TypesafeAdapter` decoding logic.
    "laya_local": {
        "base_url": "http://127.0.0.1:8811",
        "default_model": "laya-multilingual",
        "price_in_per_m": None,
        "price_out_per_m": None,
        "mode": "native",
        "notes": "Local Laya serve (Python -m laya.serve). Same /v1/systemone wire format as Jev.",
    },
}


# OpenAI models evaluated by the notebook. Standard-tier USD per 1M tokens
# (short context), from https://developers.openai.com/api/docs/pricing
# (checked 2026-10-17). Unlisted models fall back to the `openai` preset price.
OPENAI_MODELS: dict[str, dict[str, float]] = {
    "gpt-6.1-sol":   {"price_in_per_m": 2.00, "price_out_per_m": 10.00},
    "gpt-6-luna":    {"price_in_per_m": 0.10, "price_out_per_m": 0.50},
    "gpt-5.6-terra": {"price_in_per_m": 2.00, "price_out_per_m": 12.00},
    "gpt-5.6-luna":  {"price_in_per_m": 0.20, "price_out_per_m": 1.20},
    "gpt-4.1-mini":  {"price_in_per_m": 0.40, "price_out_per_m": 1.60},
}

# Reasoning models spend output tokens on hidden reasoning before the answer,
# so a 200-token cap leaves them with an empty reply. Both the adapter and the
# runner's budget reservation use effective_max_output_tokens() so they agree.
REASONING_MIN_OUTPUT_TOKENS = 2048


def effective_max_output_tokens(model: str, requested: int) -> int:
    if is_reasoning_model(model):
        return max(int(requested), REASONING_MIN_OUTPUT_TOKENS)
    return int(requested)


def resolve_prices(
    name: str,
    price_in_per_m: float | None,
    price_out_per_m: float | None,
    model: str = "",
) -> tuple[float | None, float | None]:
    """Fill unspecified prices from the model table, then the provider preset.

    Without this, a preset adapter run without explicit prices records $0
    and reserves $0, so the budget cap never engages.
    """
    spec = dict(PROVIDERS.get(name, {}))
    if name == "openai":
        spec.update(OPENAI_MODELS.get(model or spec.get("default_model", ""), {}))
    if price_in_per_m is None:
        price_in_per_m = spec.get("price_in_per_m")
    if price_out_per_m is None:
        price_out_per_m = spec.get("price_out_per_m")
    return price_in_per_m, price_out_per_m


def get_adapter(
    *,
    name: str,
    model: str = "",
    endpoint: str = "",
    key: str = "",
    request_options: dict | None = None,
    price_in_per_m: float | None = None,
    price_out_per_m: float | None = None,
    max_input_tokens: int = 2000,
    max_output_tokens: int = 200,
    **kwargs,
):
    """Factory. Always returns an object exposing .call(task)."""
    if name == "mock":
        # If the model name starts with 'uniform', return a flat dist (mostly wrong).
        uniform = model.startswith("uniform")
        return MockAdapter(uniform=uniform)
    if name == "openai_compat":
        if not endpoint:
            raise ValueError("--endpoint required for openai_compat")
        return OpenAICompatAdapter(
            base_url=endpoint,
            model=model,
            key=key,
            request_options=request_options,
            max_tokens=effective_max_output_tokens(model, max_output_tokens),
        )
    if name in PROVIDERS:
        spec = PROVIDERS[name]
        base_url = endpoint if endpoint else spec["base_url"]
        mdl = model or spec["default_model"]
        # Native /v1/systemone adapters (Jev + Laya-local) share the same wire
        # format; both go through `TypesafeAdapter`. Laya-local inherits from it
        # but takes different defaults — keep them as separate factory branches
        # so future config (e.g. timeouts, headers) can diverge per runner.
        if name in ("jev", "typesafe"):
            return TypesafeAdapter(
                base_url=base_url,
                model=mdl,
                key=key,
                request_options=request_options,
            )
        if name == "laya_local":
            return LayaLocalAdapter(
                base_url=base_url,
                model=mdl,
                key=key,
                request_options=request_options,
            )
        return OpenAICompatAdapter(
            base_url=base_url,
            model=mdl,
            key=key,
            request_options=request_options,
            max_tokens=effective_max_output_tokens(mdl, max_output_tokens),
        )
    raise ValueError(f"unknown adapter: {name!r}")