# Setup

## Requirements

- Python 3.11 or newer
- Optional: FFmpeg for audio/video operations
- Optional: Tesseract with English language data for OCR

Create an isolated environment from `AI-Agent-Tools/perception-tools`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,documents,media,data]"
```

Use `.[private]` only when Google Calendar or Notion is needed. The legacy
`requirements.txt` remains a convenient full runtime install, while
`pyproject.toml` is the canonical package definition.

Copy `env.example` to `.env` and set only the credentials you need. Open-Meteo
weather does not require a key. Hosted image/video analysis uses
`OPENAI_API_KEY` and optionally `PERCEPTION_VISION_MODEL`.

Verify the installation:

```bash
perception-tools list
perception-tools demo --offline
python -m pytest
```
