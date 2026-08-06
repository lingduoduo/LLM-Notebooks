# Agent Cost Analysis English Validity Design

## Goal

Make `AI-Agent-Evaluation/agent-cost-analysis` a coherent English-language cost-analysis benchmark while preserving the original refund scenario and the intended 2x2 comparison of prompt caching and context compression.

## Behavioral requirements

- All user-facing text, prompts, fixture values, comments, documentation, and errors remain English.
- The stable system prompt is at least 1,024 tokens under the benchmark tokenizer so cache-enabled scenarios are eligible for prompt caching.
- Commerce values retain their original meaning and are consistently identified as CNY/RMB. Translation must not silently convert yuan amounts to US dollars.
- Every tool result remains valid JSON after Python string assembly.
- The four scenario keys and public Python entry points remain unchanged.
- Offline mode remains deterministic and requires no API credentials.

## Offline trace policy

The bundled trace will be regenerated as deterministic synthetic data representing the English workload. Its metadata will identify it as synthetic rather than claiming it was observed from a live model. The data will preserve the intended qualitative relationships:

- naive: no cached tokens and full context growth;
- KV-only: cache hits with full context growth;
- compression-only: reduced prompt growth without cache hits;
- both: cache hits plus reduced prompt growth.

Live runs saved through `--save-trace` remain observed measurements and will be labeled accordingly.

## Implementation

1. Add regression tests for English-only content, valid tool-result JSON, explicit CNY currency, cache-eligible prompt length, and trace provenance.
2. Introduce a deterministic tokenizer-independent minimum-length guard for the stable prompt, while continuing to use `tiktoken` for runtime tool-context estimates.
3. Expand the English prompt with relevant customer-support policy and tool-contract material until it exceeds the eligibility threshold.
4. Correct currency references throughout prompts, summaries, documentation, and fixtures.
5. Regenerate synthetic English trace counts and metadata, then update offline loading/reporting to preserve provenance.
6. Update README and environment documentation to describe the English benchmark and trace policy accurately.

## Error handling

- Invalid or incomplete trace scenarios continue to produce clear CLI errors.
- Trace metadata distinguishes `synthetic` from `observed` data.
- Tests fail if the prompt falls below the conservative cache-length invariant or if translated fixtures stop parsing as JSON.

## Verification

- Run the complete pytest suite.
- Compile all Python modules.
- Parse the bundled trace with the standard JSON parser.
- Run all four scenarios in offline mode.
- Scan the module for Chinese characters.
- Confirm CLI help and README descriptions match actual behavior.
