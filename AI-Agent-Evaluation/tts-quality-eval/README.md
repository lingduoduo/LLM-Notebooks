# TTS Quality Evaluation Pipeline

This project implements an end-to-end benchmark pipeline for TTS quality across multiple providers and configurations. The same source scripts are synthesized, then evaluated with an LLM-as-a-Judge rubric.

It compares:
- provider differences (OpenAI, ElevenLabs, Fish Audio, Minimax, Doubao)
- model / voice / speed settings
- objective speech metrics and rubric-based subjective dimensions

The workflow is fully reproducible and can run offline checks when API keys are unavailable.

### Goals

Answer practical questions such as:
- How much difference exists between `tts-1` and `tts-1-hd`?
- What is the quality cost of changing voice or speed (for example 1.5x)?

The pipeline answers these through a single command and produces a structured comparison report.

### Evaluation dimensions

Per synthesized audio, both objective and judged dimensions are recorded:
- Clarity: transcription consistency with source text
- Naturalness: speaking rate vs target range
- Pacing: pause and rhythm appropriateness based on speech length
- Overall score: holistic quality

WER-based objective metrics are computed from normalized English word tokens:
text is lowercased and split into word tokens before comparison. Because the
transcript comes from Whisper, the measured WER also depends on Whisper's
transcription accuracy.

### Provider support

- TTS synthesis is implemented for multiple providers (OpenAI via SDK, others via REST).
- Default run covers 4 OpenAI configurations with only `OPENAI_API_KEY`.
- `--providers` enables cross-provider comparisons.
- Missing key -> that provider is skipped; the benchmark continues.

### Judge/backend details

- Default rubric path uses OpenAI: Whisper (`whisper-1`) for transcript + `gpt-5.6-luna` for scoring.
- OpenRouter fallback is supported only for rubric chat judging (`gpt-*` mapping handled).
- Optional `--gemini` enables multimodal direct scoring from Gemini if `GEMINI_API_KEY` is provided.

### Files

| File | Purpose |
|---|---|
| `config.py` | providers, model pricing, configs, corpus |
| `pipeline.py` | synthesis, ffprobe duration, transcription, WER, rubric scoring |
| `demo.py` | command entry, run grid, output summaries |
| `requirements.txt` / `env.example` | dependencies and env template |

### Run

```bash
pip install -r requirements.txt
brew install ffmpeg
export OPENAI_API_KEY=sk-...

python demo.py
python demo.py --quick
python demo.py --extra
python demo.py --gemini
python demo.py --fresh
python demo.py --providers openai,minimax,elevenlabs
python demo.py --text "Revenue grew 37.5 percent in 2026."
python demo.py --judge-model gpt-5.6-luna
python demo.py --output ./runs/exp1
python demo.py --list-providers
python demo.py --dump-rubric
```

Outputs are under `output/` (audio) and `output/results.json` (structured results).

### Robustness notes

- Required-key (`OPENAI_API_KEY`) and ffprobe checks fail fast with clear instructions; a provider-specific missing key only marks that provider's cells as failed without stopping the run.
- A single failed (provider, text) cell does not stop the full run.
- OpenAI SDK is configured with retries.

### Limitations

- Default rubric path does not directly hear audio, so tonal/voice authenticity is partly inferred.
- WER depends on Whisper accuracy.
- Scores are relative, not absolute quality certification.
