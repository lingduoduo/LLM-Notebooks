# Execution Tools English Conversion Design

## Goal

Convert all maintained user-facing and developer-facing Chinese text in
`execution-tools` to clear English and replace its multi-provider Chinese-model
configuration with a direct OpenAI-only integration.

## Scope

The conversion covers:

- CLI descriptions, option help, errors, headings, and demo output
- Runtime messages returned by execution tools
- Python module, class, and function docstrings
- Inline and block comments
- Embedded demonstration content
- References to Chinese prose or chapter headings
- Tests or fixtures whose maintained text is Chinese
- Provider selection, API-key configuration, model defaults, documentation,
  and tests for the LLM integration

Remove support and references for SiliconFlow, Doubao, Kimi/Moonshot, and
OpenRouter. Use the direct OpenAI API, `OPENAI_API_KEY`, and `gpt-5.6` as the
default model. Preserve `MODEL` as an optional override. Already-English text
may receive consistency edits needed for this migration. The already-English
`perception-tools` package is outside this change.

## Compatibility

The conversion will preserve:

- Python identifiers and import paths
- CLI command and option names
- Unrelated environment-variable names
- JSON keys and result structures
- Non-LLM external APIs and integration behavior
- File paths and configuration semantics
- Existing safety checks and execution behavior

English runtime strings will intentionally replace Chinese runtime strings.
Tests that assert those messages will be updated to assert the English text.
The LLM configuration is an intentional compatibility change: obsolete
provider names, provider-specific API keys, base URLs, and the OpenRouter
fallback will no longer be accepted or documented.

## Implementation

Translate the Chinese text in `execution-tools/cli.py`,
`execution-tools/execution_tools.py`, and any other maintained source,
documentation, configuration, or test file found by a repository-wide Unicode
scan. Use concise technical English and consistent terminology:

- "execution tools" for 执行工具
- "workspace" for 工作区 or 工作目录
- "validation" for 校验
- "approval" for 审批
- "truncation and persistence" for 截断与持久化
- "offline demo" for 离线演示

No unrelated refactoring or packaging changes are included.

Replace the provider router in `execution-tools/config.py` with a single OpenAI
configuration:

- Read `OPENAI_API_KEY`.
- Default `MODEL` to `gpt-5.6`.
- Return an OpenAI provider configuration without a third-party `base_url`.
- Raise a clear configuration error when `OPENAI_API_KEY` is absent.
- Remove provider-specific keys, branches, fallbacks, and error messages.

Update `llm_helper.py` to describe OpenAI models only while preserving lazy
client creation and existing call behavior. Remove the CLI provider override
because the provider is no longer selectable. Update `env.example`,
`README.md`, `EXPERIMENT.md`, and affected tests to document and verify the
OpenAI-only configuration.

## Validation

Add an automated English-only test for maintained `execution-tools` text files.
The test will scan relevant source, test, documentation, and configuration
files for Han characters while excluding generated artifacts, caches, virtual
environments, and Git metadata.

Validation will include:

1. Running the English-only scan.
2. Running the full `execution-tools` test suite.
3. Exercising CLI help, tool listing, and the offline demo to confirm that
   visible output is English and behavior remains intact.
4. Testing OpenAI configuration with and without `OPENAI_API_KEY`, including
   the `MODEL` override.
5. Scanning maintained files for obsolete provider and model names.

## Success Criteria

- No Han characters remain in maintained `execution-tools` files.
- All user-visible CLI and runtime text is English.
- All comments and docstrings are English.
- Commands, APIs, structured result fields, and behavior remain compatible.
- The only supported LLM provider is direct OpenAI.
- `OPENAI_API_KEY` is the only LLM credential and `gpt-5.6` is the default.
- `MODEL` can override the default OpenAI model.
- No maintained file references SiliconFlow, Doubao, Kimi, Moonshot,
  OpenRouter, Qwen, Gemini, or their retired configuration variables.
- The full non-live test suite passes.
