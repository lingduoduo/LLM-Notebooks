"""Shared generation parameter compatibility for contextual retrieval."""


def reasoning_safe_temperature(model, requested=1.0):
    """GPT-5 models only accept temperature=1."""
    m = str(model or "").lower().replace("/", "-")
    return 1 if "gpt-5" in m else requested
