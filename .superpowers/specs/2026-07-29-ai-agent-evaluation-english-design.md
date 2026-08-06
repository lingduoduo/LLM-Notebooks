# AI Agent Evaluation English Conversion Design

## Goal

Convert `AI-Agent-Evaluation/tts-quality-eval` into a coherent English-language TTS evaluation project. This includes its source code, documentation, command-line interface, judge prompts, sample corpus, result schema, and tests.

## Scope

- Translate Chinese comments, docstrings, help text, status output, errors, prompts, examples, and environment-file guidance into English.
- Replace the Chinese test corpus with English samples covering:
  - numbers, percentages, and dates;
  - pronunciation-sensitive or heteronym wording;
  - a long news-style sentence;
  - proper nouns and excited delivery.
- Guide Whisper to transcribe English.
- Replace character error rate (CER) with case-insensitive word error rate (WER), using normalized word tokens and Levenshtein distance.
- Replace character accuracy with word accuracy, clamped to zero.
- Rename rubric dimensions and structured output keys to `clarity`, `naturalness`, `pacing`, and `overall`.
- Update the LLM and Gemini judging prompts for English text and English reasons.
- Update aggregation, sorting, tables, JSON output, tests, and README examples to use the English schema and terminology.

## Compatibility

This is an intentional schema conversion, not a compatibility layer. Existing Chinese rubric keys and CER-oriented result fields will not be retained as aliases. Provider identifiers, provider API payload formats, credential names, model names, and brand names remain unchanged.

## Evaluation Flow

1. A configured provider synthesizes an English corpus sample.
2. `ffprobe` measures audio duration.
3. Whisper transcribes the audio as English.
4. The pipeline normalizes reference and hypothesis text into lowercase word tokens.
5. Word-level Levenshtein distance produces WER and word accuracy.
6. The selected judge scores clarity, naturalness, pacing, and overall quality.
7. The CLI prints per-sample results and an aggregated configuration comparison, then writes structured JSON.

## Error Handling

The current per-sample isolation remains: a provider or evaluation failure creates an error record without stopping the full comparison. Offline commands remain usable without API credentials. Error messages will be English.

## Verification

- Run the unit tests after updating them for English rubric keys and fixtures.
- Add or update metric tests to confirm word normalization, WER, and word-accuracy behavior.
- Run Python compilation checks.
- Run offline CLI commands such as `--help`, `--list-providers`, and `--dump-rubric`.
- Scan the project for remaining Chinese text, permitting only unavoidable external proper nouns when needed.
