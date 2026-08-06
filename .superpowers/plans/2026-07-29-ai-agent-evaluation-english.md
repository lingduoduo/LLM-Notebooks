# AI Agent Evaluation English Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the TTS evaluator into an English-language benchmark with word error rate metrics and an English result schema.

**Architecture:** Preserve the existing provider adapters and orchestration flow. Change the language-specific boundary in `config.py` and `pipeline.py` from Chinese characters/CER to English words/WER, then propagate the new names through the CLI, structured output, tests, and documentation.

**Tech Stack:** Python 3, OpenAI Python SDK, standard-library `urllib`, `ffprobe`, pytest

## Global Constraints

- Use English for source comments, docstrings, CLI copy, errors, prompts, corpus content, tests, and documentation.
- Use normalized lowercase word tokens and word-level Levenshtein distance.
- Use rubric keys `clarity`, `naturalness`, `pacing`, and `overall`.
- Do not retain aliases for Chinese rubric keys or CER result fields.
- Do not change provider identifiers, API payload formats, credential names, model names, or brand names.
- Do not add dependencies.

---

### Task 1: Define and Test English Word Metrics

**Files:**
- Modify: `AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/pipeline.py`

**Interfaces:**
- Produces: `normalize_words(text: str) -> list[str]`
- Produces: `_edit_distance(a: list[str], b: list[str]) -> int`
- Produces: `ErrorRate(wer: float, accuracy: float, edits: int, ref_len: int)`
- Produces: `word_error_rate(reference: str, hypothesis: str) -> ErrorRate`

- [ ] **Step 1: Add failing word-metric tests**

Add tests asserting:

```python
def test_normalize_words_handles_case_punctuation_and_contractions():
    assert pipeline.normalize_words("Hello, WORLD! Don't stop.") == [
        "hello", "world", "don't", "stop"
    ]


def test_word_error_rate_counts_word_edits():
    result = pipeline.word_error_rate(
        "The quick brown fox", "The fast brown fox jumps"
    )
    assert result.edits == 2
    assert result.ref_len == 4
    assert result.wer == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)


def test_word_error_rate_empty_reference_is_perfect():
    assert pipeline.word_error_rate("", "").wer == 0.0
    assert pipeline.word_error_rate("", "").accuracy == 1.0
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `cd AI-Agent-Evaluation/tts-quality-eval && pytest -q test_judge_robustness.py`

Expected: FAIL because `normalize_words` and `word_error_rate` do not exist.

- [ ] **Step 3: Replace character normalization and CER**

Implement tokenization with:

```python
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())
```

Make `_edit_distance` operate on token sequences. Rename `ErrorRate.cer` to `wer`, rename `char_error_rate` to `word_error_rate`, divide edit distance by the reference word count, and clamp accuracy with `max(0.0, 1.0 - wer)`.

- [ ] **Step 4: Run focused tests**

Run: `cd AI-Agent-Evaluation/tts-quality-eval && pytest -q test_judge_robustness.py`

Expected: the new metric tests PASS; judge tests may still fail until Task 2 changes their schema.

- [ ] **Step 5: Commit the metric conversion**

```bash
git add AI-Agent-Evaluation/tts-quality-eval/pipeline.py AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py
git commit -m "feat: evaluate English transcripts with WER"
```

### Task 2: Convert the Corpus and Judge Schema

**Files:**
- Modify: `AI-Agent-Evaluation/tts-quality-eval/config.py`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/pipeline.py`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py`

**Interfaces:**
- Consumes: `word_error_rate(reference: str, hypothesis: str) -> ErrorRate`
- Produces: `RUBRIC_DIMENSIONS = ["clarity", "naturalness", "pacing", "overall"]`
- Produces: `judge_rubric(reference, emotion, hypothesis, duration, wer, model=None) -> Rubric`
- Produces: English `Sample` instances in `config.CORPUS`

- [ ] **Step 1: Convert judge-response fixtures to the English schema**

Use payloads such as:

```python
{
    "clarity": {"score": None, "reason": "Unable to determine"},
    "naturalness": {"score": 4, "reason": "The speaking rate is natural"},
    "pacing": {"score": 3},
    "overall": {"score": 5, "reason": "Usable overall"},
}
```

Change all assertions to the English keys and change sample judge inputs to English.

- [ ] **Step 2: Run the judge tests and confirm schema failures**

Run: `cd AI-Agent-Evaluation/tts-quality-eval && pytest -q test_judge_robustness.py`

Expected: FAIL because production rubric dimensions still use Chinese keys.

- [ ] **Step 3: Convert configuration and corpus**

Translate every comment, docstring, provider note, and label. Keep the provider key `doubao`, but display its established English brand as `Doubao (Volcengine)`. Replace the four samples with:

```python
Sample("num", "In the third quarter of 2026, revenue grew 37.5 percent, up 12 percentage points year over year.", "numbers, percentages, and dates", "neutral")
Sample("pronunciation", "The bass player caught a bass near the lead mine after he read the latest report.", "heteronyms and pronunciation-sensitive words", "neutral")
Sample("long", "According to the report, as artificial intelligence advances rapidly, more companies are applying large language models to customer service, content creation, and data analysis, significantly improving operational efficiency.", "long sentence and news style", "neutral")
Sample("emotion", "Fantastic! OpenAI's newly released model achieved an amazing result on the GAIA benchmark!", "proper nouns and excited delivery", "excited")
```

- [ ] **Step 4: Convert transcription and judging**

Rename `_ZH_PROMPT` to `_EN_PROMPT`, call Whisper with `language="en"`, and write all rubric descriptions and prompts in English. State a target English speaking rate of roughly 2–3 words per second. Pass `wer` to the judge and require English JSON reasons.

Translate all remaining comments, docstrings, exceptions, and Gemini defensive errors in `pipeline.py`, while leaving API wire data unchanged.

- [ ] **Step 5: Run the judge tests**

Run: `cd AI-Agent-Evaluation/tts-quality-eval && pytest -q test_judge_robustness.py`

Expected: PASS.

- [ ] **Step 6: Commit the English corpus and judge**

```bash
git add AI-Agent-Evaluation/tts-quality-eval/config.py AI-Agent-Evaluation/tts-quality-eval/pipeline.py AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py
git commit -m "feat: convert TTS corpus and rubric to English"
```

### Task 3: Propagate WER Through the CLI and Results

**Files:**
- Modify: `AI-Agent-Evaluation/tts-quality-eval/demo.py`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py`

**Interfaces:**
- Consumes: `ErrorRate.wer`
- Consumes: English `RUBRIC_DIMENSIONS`
- Produces: result records containing `wer`, `word_accuracy`, and `words_per_second`
- Produces: English CLI help, details, summaries, and errors

- [ ] **Step 1: Add orchestration assertions**

Stub synthesis, duration, transcription, and judging, call `evaluate_one`, and assert:

```python
assert record["wer"] == pytest.approx(0.25)
assert record["word_accuracy"] == pytest.approx(0.75)
assert record["words_per_second"] == pytest.approx(1.0)
assert "cer" not in record
assert "accuracy" not in record
assert "speed" not in record
```

Add a summary assertion that records sort by descending `overall`, then ascending `wer`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `cd AI-Agent-Evaluation/tts-quality-eval && pytest -q test_judge_robustness.py`

Expected: FAIL because `demo.py` still calls `char_error_rate` and emits CER fields.

- [ ] **Step 3: Convert orchestration fields**

In `evaluate_one`, call `word_error_rate`, pass `er.wer` to `judge_rubric`, and store `wer`, `word_accuracy`, and `words_per_second`. In summaries, average those fields and sort with:

```python
rows.sort(key=lambda row: (-row.get("overall", 0), row.get("wer", 1)))
```

- [ ] **Step 4: Translate CLI output**

Translate module documentation, comments, help, examples, errors, run headings, provider state, rubric output, per-record labels, summary table labels, and completion messages. Use `Reference`, `Transcript`, `Duration`, `Words/s`, `WER`, and `Word accuracy`.

- [ ] **Step 5: Run unit and offline CLI tests**

Run:

```bash
cd AI-Agent-Evaluation/tts-quality-eval
pytest -q
python demo.py --help
python demo.py --list-providers
python demo.py --dump-rubric
```

Expected: tests PASS and all command output is English.

- [ ] **Step 6: Commit the CLI conversion**

```bash
git add AI-Agent-Evaluation/tts-quality-eval/demo.py AI-Agent-Evaluation/tts-quality-eval/test_judge_robustness.py
git commit -m "feat: report English TTS evaluation results"
```

### Task 4: Finish Documentation and Translation Audit

**Files:**
- Modify: `AI-Agent-Evaluation/tts-quality-eval/README.md`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/env.example`
- Modify: `AI-Agent-Evaluation/tts-quality-eval/requirements.txt`

**Interfaces:**
- Consumes: CLI field names and behavior from Tasks 1–3
- Produces: English setup and usage documentation consistent with the code

- [ ] **Step 1: Update documentation**

Change README references from CER to WER, character accuracy to word accuracy, pause/rhythm to pacing, and the Chinese custom-text example to:

```bash
python demo.py --text "Revenue grew 37.5 percent in 2026."
```

Explain that normalization uses lowercase English word tokens and note that WER depends on Whisper accuracy. Translate every comment in `env.example` and `requirements.txt`.

- [ ] **Step 2: Scan for remaining Chinese text**

Run:

```bash
rg -n '[一-龥]|[ぁ-んァ-ン]|[가-힣]' AI-Agent-Evaluation/tts-quality-eval
```

Expected: no matches.

- [ ] **Step 3: Run complete verification**

Run:

```bash
cd AI-Agent-Evaluation/tts-quality-eval
python -m compileall -q .
pytest -q
python demo.py --help
python demo.py --list-providers
python demo.py --dump-rubric
```

Expected: compilation succeeds, tests PASS, and all offline output is English.

- [ ] **Step 4: Check formatting and scope**

Run:

```bash
git diff --check
git status --short
git diff -- AI-Agent-Evaluation/tts-quality-eval
```

Expected: no whitespace errors; only the planned evaluator files are changed.

- [ ] **Step 5: Commit documentation**

```bash
git add AI-Agent-Evaluation/tts-quality-eval/README.md AI-Agent-Evaluation/tts-quality-eval/env.example AI-Agent-Evaluation/tts-quality-eval/requirements.txt
git commit -m "docs: document English TTS evaluation workflow"
```
