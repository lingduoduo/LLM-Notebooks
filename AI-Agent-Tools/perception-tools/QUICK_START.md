# Quick Start

```bash
cd AI-Agent-Tools/perception-tools
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[documents,media,data]"
perception-tools list
perception-tools demo --offline
```

Start the stdio MCP server with:

```bash
perception-tools-mcp
```

Inspect any tool before calling it:

```bash
perception-tools info weather
perception-tools run weather location=Boston
```

The weather call requires network access but no API key. OpenAI-backed media
analysis requires `OPENAI_API_KEY`.
