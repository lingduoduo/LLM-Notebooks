# TTS quality evaluation: final fixes report

Date: 2026-07-29

## Scope

Final review fixes for `AI-Agent-Evaluation/tts-quality-eval`, preserving the
previous English-conversion work and excluding pre-existing untracked files.

## Fixes

1. Audio cache paths now include the first 12 hexadecimal characters of the
   SHA-256 hash of the UTF-8 reference text. The same configuration and sample
   ID therefore cannot reuse stale audio after a corpus edit or a different
   `--text` value.
2. `word_error_rate` treats an empty normalized reference as perfect only when
   the normalized hypothesis is also empty. A non-empty hypothesis returns
   `edits=len(hypothesis_tokens)`, `wer=1.0`, `accuracy=0.0`, and `ref_len=0`.
3. LLM and Gemini responses share `parse_rubric_response`. Only literal
   integers from 1 through 5 are accepted; null, missing, malformed, boolean,
   and out-of-range values become the zero sentinel.
4. Gemini's audio prompt now embeds `RUBRIC_DESCRIPTIONS` and the practical
   2–3 words/second reading-rate guidance.
5. README failure behavior now matches the implementation: missing OpenAI key
   fails before a run; missing `ffprobe` or provider-specific credentials mark
   the affected cells failed while other cells continue. It also documents the
   text-hashed cache behavior.
6. Removed the unused `traceback` import and added concise annotations to the
   CLI helpers and rubric result fields.

## TDD evidence

New tests were written before the implementation changes. The first targeted
run failed in four expected ways: empty-reference WER returned a perfect score,
`audio_path` lacked the text argument, and both judge paths accepted invalid
scores. After the minimal implementation changes, the targeted suite passed.

## Verification

| Check | Result |
| --- | --- |
| `pytest -q` | 15 passed |
| `python -m compileall -q .` | passed |
| `python demo.py --help` | passed |
| `python demo.py --list-providers` | passed without API keys |
| `python demo.py --dump-rubric` | passed without API keys |
| CJK scan of source/docs | no matches |
| Legacy CER/Chinese-metric scan of source/docs | no matches |
| `git diff --check` | passed |

## Notes

No live TTS, transcription, or remote judge call was made because the
workspace has no provider credentials. The offline CLI paths and mocked judge
coverage exercised the changed behavior. A pre-existing untracked
`AI-Agent-Evaluation/tts-quality-eval/.gitignore` and untracked
`docs/superpowers/plans/` remain intentionally outside this fix commit.
