# Contextual Retrieval OpenAI-Only Model Setup

## Goal

Convert all LLM configuration and usage under `AI-Agent-KnowledgeBase/contextual-retrieval` to use the official OpenAI API exclusively. The application will authenticate with `OPENAI_API_KEY`, default to `gpt-5.6-terra`, and contain no active routing or instructions for Kimi, Moonshot, Doubao, SiliconFlow, OpenRouter, Groq, Together, or DeepSeek.

## Scope

The conversion covers:

- LLM configuration and environment loading
- OpenAI client construction
- Agent and contextual-chunk generation calls
- Command-line model overrides
- Evaluation metadata and console output
- Example environment configuration
- Tests and user documentation

Knowledge-base backends such as local retrieval, Dify, RAPTOR, and GraphRAG remain unchanged. They are retrieval integrations rather than LLM providers.

The existing Chat Completions request flow remains in place. Migrating to the Responses API is outside this change.

## Configuration

`LLMConfig` will expose:

- `model`, defaulting to `gpt-5.6-terra`
- `api_key`, optionally supplied directly and otherwise read from `OPENAI_API_KEY`
- existing generation settings such as temperature, maximum tokens, and streaming

Provider enums, provider-default maps, alternate API-key mappings, custom base URLs, and OpenRouter fallback logic will be removed. `Config.from_env()` will accept `LLM_MODEL` as an optional model override and will no longer read `LLM_PROVIDER`.

If neither an explicit API key nor `OPENAI_API_KEY` is available when an LLM client is needed, configuration will raise a clear error naming `OPENAI_API_KEY`.

## Runtime Behavior

`agent.py` and `contextual_chunking.py` will construct the official client directly:

```python
OpenAI(api_key=api_key)
```

Both paths will obtain the model from the same `LLMConfig`, ensuring that the default and environment override behave consistently.

Existing GPT reasoning-model request compatibility will be preserved where required. Provider-specific temperature handling and comments will be removed.

## Command-Line Interfaces

The following provider-selection options will be removed:

- `main.py --provider`
- `evaluation/evaluate.py --provider`
- `index_local_laws_contextual.py --llm-provider`

Model overrides remain available:

- `--model`
- `--llm-model`

Their help text will identify the value as an OpenAI model ID.

## Documentation and Examples

`env.example` will list only `OPENAI_API_KEY` for LLM authentication and `LLM_MODEL=gpt-5.6-terra` for the optional model selection.

README files and runnable examples will:

- instruct users to set `OPENAI_API_KEY`
- use `gpt-5.6-terra` in commands
- remove provider comparisons, alternate-provider keys, and Kimi defaults
- retain references to Anthropic's Contextual Retrieval technique where they describe the research method rather than an API dependency

## Evaluation and Output

Evaluation results will record the OpenAI model ID without an LLM-provider field. Console messages and basic test output will report the model rather than a configurable provider.

## Testing

Tests will be written before production changes and will cover:

1. `LLMConfig` defaults to `gpt-5.6-terra`.
2. `OPENAI_API_KEY` is used when no explicit key is supplied.
3. An explicit API key takes precedence.
4. `LLM_MODEL` overrides the default model.
5. Missing credentials produce an actionable `OPENAI_API_KEY` error.
6. Client configuration contains no alternate base URL or provider routing.
7. Active Python, environment-template, and primary documentation surfaces contain no removed provider configuration or model defaults.

Tests that make real API calls will remain conditional on `OPENAI_API_KEY`; unit tests will not require network access.

## Success Criteria

- Every active LLM path under `AI-Agent-KnowledgeBase/contextual-retrieval` uses the official OpenAI client and `OPENAI_API_KEY`.
- `gpt-5.6-terra` is the single default LLM model.
- Users may override the model but cannot select a different provider through configuration or CLI options.
- No non-OpenAI provider routing remains in executable code or setup instructions.
- Existing retrieval-backend choices continue to work.
- The focused test suite passes without making external API calls.
