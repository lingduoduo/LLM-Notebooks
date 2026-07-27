# Perception Tools English Repository Integration

## Goal

Turn `AI-Agent-Tools/perception-tools` into a reliable, installable Python
project that runs from a fresh checkout, exposes stable CLI and MCP entry
points, uses English on every active surface, and uses the official OpenAI API
exclusively for hosted AI-model calls.

## Scope

The work covers:

- Python package structure and imports
- CLI and MCP server entry points
- dependency and optional-integration boundaries
- OpenAI vision and transcription configuration
- deterministic unit tests and explicitly gated live tests
- Docker execution
- environment templates and all project documentation
- repository path and command accuracy

Existing MCP tool names and the `ActionResponse` JSON shape remain compatible.
The internal implementation may move from `src` to an importable package.

The public data, local filesystem, document, media, Google Calendar, and Notion
tool families remain in scope. The project will not add new tool families or
redesign their public request/response contracts.

## Package Architecture

The implementation will use a conventional package:

```text
AI-Agent-Tools/perception-tools/
├── perception_tools/
│   ├── __init__.py
│   ├── cli.py
│   ├── server.py
│   └── tool modules
├── tests/
├── pyproject.toml
├── requirements.txt
├── env.example
├── Dockerfile
└── README.md
```

Internal imports will be package-relative. No production or test module will
modify `sys.path`.

`pyproject.toml` will expose:

- `perception-tools` for the direct CLI
- `perception-tools-mcp` for the stdio MCP server

Module execution will also work through:

```bash
python -m perception_tools.cli
python -m perception_tools.server
```

The MCP server will continue to use stdio transport.

## Tool Registration and Optional Dependencies

The package will retain the existing tool names and organize them under the
current search, multimodal, filesystem, public-data, and private-data
categories.

Tool registration must not make unrelated capabilities unusable because one
optional package is absent. Heavy or integration-specific imports will be lazy
where necessary. When a caller invokes a tool whose dependency is missing, the
tool will return a structured English `ActionResponse` identifying the missing
package or configuration.

Required dependencies will be sufficient to:

- import the package
- list and inspect all registered tools
- run the offline CLI demo
- start the MCP server
- execute deterministic unit tests

Optional extras may group media, document, data-source, and private-integration
dependencies. `requirements.txt` will remain as a convenient full-install
surface and will stay consistent with `pyproject.toml`.

## English-Only Surfaces

All active user and developer surfaces under `AI-Agent-Tools/perception-tools`
will be English:

- CLI help, category labels, tool descriptions, errors, and demo output
- docstrings, comments, fixtures, and test names
- README and supporting Markdown documents
- environment-template comments
- MCP instructions and tool descriptions
- Docker comments and operational messages

A static regression test will scan active source, tests, configuration, and
documentation for Han characters. External URLs and third-party proper names
do not require translation, but active explanatory text must be English.

## OpenAI-Only Hosted Model Calls

Hosted image, video, and transcription calls will use the official
`openai.OpenAI` client and `OPENAI_API_KEY`.

The vision model will default to the existing OpenAI model choice and may be
overridden with `PERCEPTION_VISION_MODEL`.

The following will be removed from active code and setup instructions:

- `OPENROUTER_API_KEY`
- OpenRouter model-ID mapping
- OpenRouter `base_url`
- alternate LLM routing
- `OPENAI_BASE_URL` custom-gateway behavior

Local Whisper remains an optional local execution path. If local Whisper is
not installed, transcription may fall back to the official OpenAI
transcription API when `OPENAI_API_KEY` is set.

Missing OpenAI credentials must produce an actionable English error naming
`OPENAI_API_KEY`. Unit tests will mock the client and never make live model
calls.

## CLI Behavior

The CLI retains the `list`, `info`, `run`, and `demo` commands.

- `list` groups the complete registry by the five existing categories.
- `info` shows the callable signature, dependency/configuration notes, and an
  invocation example.
- `run` parses `key=value` arguments, calls the selected async tool, and prints
  the standardized JSON result.
- `demo --offline` uses temporary local files and makes no network calls.
- `demo` may use public network services and must handle unavailable services
  without crashing.

Tool discovery and offline commands must work without optional private-service
credentials.

## MCP Server Behavior

The server entry point will build one FastMCP server with the existing name
`perception-tools`. Tool wrappers will preserve existing tool names and
parameters.

Startup must not emit user-facing output to stdout that would corrupt MCP
stdio framing. Operational logging belongs on stderr through Python logging.

## Tests

Tests will use pytest and live under `tests/`.

The suite will cover:

1. package imports from the repository root without `sys.path` mutation;
2. CLI help, registry listing, tool information, argument conversion, and
   offline demo behavior;
3. MCP server construction and expected tool registration;
4. OpenAI-only client construction and missing-key errors;
5. removal of OpenRouter and alternate LLM routing from active surfaces;
6. English-only static audit;
7. existing regressions for CSV limits, grep limits, negative lengths, page
   ranges, media resource cleanup, keyframe counts, wiki dates, YouTube,
   PubChem, and Yahoo Finance;
8. missing optional-dependency behavior;
9. Docker and package metadata smoke checks.

Network-dependent demonstrations will not run during the default test suite.
Any live smoke tests will require an explicit environment opt-in.

## Docker and Documentation

The Docker image will use an official supported Python base image, install
only the system libraries needed by the selected full dependency set, install
the project as a package, run as a non-root user, and start through
`perception-tools-mcp`.

Documentation will state the canonical path
`AI-Agent-Tools/perception-tools`, the actual tool count, core versus optional
installation commands, OpenAI configuration, CLI usage, MCP client
configuration, Docker usage, testing, and live-test gating.

Outdated claims about 18 tools, Google-only search, OpenWeather requirements,
Chinese CLI help, `chapter4/perception-tools`, and OpenRouter fallback will be
removed.

## Error Handling

Expected input, dependency, authentication, network, and file errors will
produce structured `ActionResponse` failures or concise CLI errors in English.
Programming errors must not be swallowed by broad executable test scripts.

Tests must not call `sys.exit` at module import time. CLI process exit codes
will be tested through callable entry points.

## Success Criteria

- The project installs successfully from its canonical repository directory.
- `perception-tools --help`, `perception-tools list`, and
  `perception-tools demo --offline` exit successfully.
- `perception-tools-mcp` starts the stdio MCP server without import errors.
- Default pytest collection and execution complete without network access.
- All hosted AI-model paths use the official OpenAI client and
  `OPENAI_API_KEY`.
- No active OpenRouter routing, alternate LLM base URL, or non-English
  explanatory text remains.
- Existing MCP tool names and `ActionResponse` response shape are preserved.
- README, package metadata, environment template, Dockerfile, and test commands
  agree with the implemented project.
