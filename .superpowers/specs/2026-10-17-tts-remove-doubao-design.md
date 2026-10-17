# TTS Quality Eval: Remove the Doubao Provider

## Goal

Remove ByteDance Doubao (Volcano Engine TTS) from `AI-Agent-Evaluation/tts-quality-eval`, completing a removal that had been started locally. The local edit deleted Doubao from `config.PROVIDERS` only, which left the provider tables inconsistent:

- `PROVIDER_CONFIGS` still had a `doubao` preset, so `--providers doubao` was still accepted.
- `pipeline._SYNTH_DISPATCH` still called the Volcano Engine endpoint.
- `--list-providers` no longer showed Doubao.

The existing 27 tests passed in that state.

## Scope

- **`config.py`:** drop the `PROVIDERS["doubao"]` entry (the original local edit) and the `PROVIDER_CONFIGS["doubao"]` preset; `TTSConfig.supports_speed()` is now true only for `minimax` among non-OpenAI providers; docstrings and comments list four providers.
- **`pipeline.py`:** delete `_synth_doubao` and its `_SYNTH_DISPATCH` entry. Keep `import base64`, which the Gemini judge path still uses.
- **`demo.py`:** update the `--list-providers` heading and the `--providers` help text.
- **`README.md`, `env.example`:** remove Doubao and its `DOUBAO_APP_ID` / `DOUBAO_ACCESS_TOKEN` variables.

OpenAI stays the default provider. ElevenLabs, Fish Audio and Minimax are unchanged.

## Behaviour change

`--providers doubao` raises `ValueError: unknown provider 'doubao'`. A `TTSConfig` with `provider="doubao"` fails in `synthesize()` with `Unknown provider`, before any network call.

## Verification

- `test_provider_registry.py` (10 tests):
  - Every preset has a `PROVIDERS` entry and a synthesizer, and the three tables list the same providers. This guards against another half-done removal.
  - Doubao is absent from all three tables and rejected by `select_configs` and `synthesize`.
  - `supports_speed` behaves as expected for each provider.
- Against the half-removed local state, 6 of these fail, including both consistency tests. Against `origin/main`, the 4 Doubao-specific tests fail.
- Full suite: 37 passed.
