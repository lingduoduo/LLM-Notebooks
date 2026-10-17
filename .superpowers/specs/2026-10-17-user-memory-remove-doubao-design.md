# User Memory: Remove the Doubao Provider

## Goal

Remove ByteDance Doubao (Volcano Engine ARK) as an LLM provider in `AI-Agent-KnowledgeBase/user-memory`. Work that used Doubao uses OpenAI instead, which is already the default provider (`PROVIDER=openai`, model `gpt-5.6-luna`).

## Decision

Remove the provider rather than keep the name `doubao` pointing at OpenAI. OpenAI already has its own provider entry, so an alias would add a second name for the same thing and hide what is actually called.

## Scope

- **`config.py`:** remove `DOUBAO_API_KEY`, `DOUBAO_BASE_URL`, the `doubao` entry in `PROVIDER_DEFAULT_MODELS`, and the Doubao branches in `get_api_key`, `validate` and the settings printout. Drop Doubao from the comments that list non-reasoning providers and OpenRouter model-id mapping.
- **`agent.py`, `conversational_agent.py`:** remove the Doubao client branch. The unsupported-provider error lists `openai`, `siliconflow`, `openrouter`.
- **`main.py`, `memory_cli.py`:** remove `doubao` from `--provider` choices and from the missing-key hint.
- **`background_memory_processor.py`:** docstring only.
- **`env.example`:** remove `DOUBAO_API_KEY` and Doubao from the provider and default-model comments.

SiliconFlow, OpenRouter and the universal OpenRouter fallback are unchanged.

## Behaviour change

`PROVIDER=doubao` or `--provider doubao` is now rejected:

- argparse rejects it in the CLIs
- the agents raise `ValueError: Unsupported provider: doubao`
- `Config.validate("doubao")` returns `False`

A user who had `PROVIDER=doubao` must switch to `openai` and set `OPENAI_API_KEY`.

## Verification

- `test_no_doubao_provider.py` checks that:
  - `Config` has no Doubao attributes, `get_api_key("doubao")` is `None`, and `validate("doubao")` is `False` with no Doubao hint
  - both agents raise `Unsupported provider: doubao`
  - the `memory_cli` parser rejects `--provider doubao`
  - OpenAI keys still resolve
- 5 of the 6 tests fail against the previous code; the OpenAI check passes on both.
- The full existing suite still passes, and no `doubao` or `volces` reference remains in the project.
