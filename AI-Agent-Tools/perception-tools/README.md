# Perception Tools

An installable Python package and MCP server with 53 perception, retrieval,
document, media, filesystem, public-data, and private-data tools for AI agents.
All user-facing text and documentation are in English.

## Install

From `AI-Agent-Tools/perception-tools`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core install supports the MCP server, CLI, search, filesystem, and basic
web tools. Install only the integrations you need:

```bash
python -m pip install -e ".[documents]"
python -m pip install -e ".[media]"
python -m pip install -e ".[data]"
python -m pip install -e ".[private]"
python -m pip install -e ".[documents,media,data,private]"
```

For development:

```bash
python -m pip install -e ".[dev,documents,media,data]"
```

## Use

```bash
perception-tools --help
perception-tools list
perception-tools list --category multimodal
perception-tools info image_analyze
perception-tools run grep pattern=ActionResponse directory=perception_tools
perception-tools demo --offline
perception-tools-mcp
```

The MCP server uses stdio. A client configuration can invoke the installed
entry point directly:

```json
{
  "mcpServers": {
    "perception-tools": {
      "command": "perception-tools-mcp"
    }
  }
}
```

## Configuration

Most public-data tools use services that do not require credentials. Copy
`env.example` to `.env` when using optional integrations.

- `OPENAI_API_KEY`: required for hosted image/video analysis and OpenAI audio
  fallback.
- `PERCEPTION_VISION_MODEL`: optional OpenAI vision model override.
- `NOTION_API_KEY`: required for Notion search.
- `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`: optional for Google Custom Search;
  DuckDuckGo is the fallback.
- Google Calendar uses local OAuth credentials.

Hosted model calls use the official OpenAI API only.

## Tool groups

The CLI is the authoritative catalog:

```bash
perception-tools list
```

It currently reports:

- Search: 4 tools
- Multimodal: 19 tools
- File System: 3 tools
- Public Data: 25 tools
- Private Data: 2 tools

Optional integrations fail with an actionable dependency message when their
extra is not installed; they do not prevent the package or server from loading.
Network-dependent tests are skipped by default. Enable them explicitly with
`RUN_LIVE_PERCEPTION_TESTS=1`.

## Development

```bash
python -m pytest
ruff check .
python -m build
```

See [SETUP.md](SETUP.md), [TOOL_REFERENCE.md](TOOL_REFERENCE.md), and
[ARCHITECTURE.md](ARCHITECTURE.md) for focused details.
