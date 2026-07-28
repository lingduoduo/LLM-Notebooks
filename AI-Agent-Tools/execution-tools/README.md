# Execution Tools MCP Server 

An MCP (Model Context Protocol) server that provides comprehensive execution tools with built-in safety mechanisms for AI agents.

This project corresponds to Experiment 4-2 in the book’s “Execution Tools” section. It focuses on layered safety (input validation, permission control, LLM pre-approval), automatic syntax verification and feedback loops, and truncation plus persistence of long outputs. Recommended start: `python cli.py demo`.

### Features

#### Safety Mechanisms

1. **LLM-Based Approval**: Irreversible operations require approval from a secondary LLM before execution
2. **Result Summarization**: Execution tool outputs larger than 10,000 characters are automatically summarized by an LLM for easier processing
3. **Automatic Verification**: Operations that can be verified (e.g., syntax checking) are automatically validated

#### Tool Categories

##### File System Tools
- **file_write**: Write content to files with automatic syntax verification
- **file_edit**: Edit existing files with diff preview and verification

##### Generic Execution Tools
- **code_interpreter**: Execute Python code in a sandboxed environment with result analysis
- **virtual_terminal**: Execute shell commands with error summarization

##### External System Integration Tools
- **google_calendar_add**: Add events to Google Calendar
- **github_create_pr**: Create GitHub Pull Requests with validation

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

1. Copy `env.example` to `.env`:
```bash
cp env.example .env
```

2. Configure your environment variables:
```
# LLM Configuration (for safety checks and summarization)
PROVIDER=kimi

# API Keys (set the one for your provider)
KIMI_API_KEY=your_kimi_key
# SILICONFLOW_API_KEY=your_siliconflow_key
# DOUBAO_API_KEY=your_doubao_key
# OPENROUTER_API_KEY=your_openrouter_key

# Model (optional, defaults to provider's default)
# MODEL=kimi-k3

# Model parameters
TEMPERATURE=0.7
MAX_TOKENS=4096

# External Services (optional)
GOOGLE_CALENDAR_CREDENTIALS_FILE=credentials.json
GITHUB_TOKEN=your_github_token

# Safety Settings
REQUIRE_APPROVAL_FOR_DANGEROUS_OPS=true
AUTO_SUMMARIZE_COMPLEX_OUTPUT=true
AUTO_VERIFY_CODE=true
```

**Supported Providers:**
- `siliconflow`: Qwen/Qwen3-235B-A22B-Thinking-2507
- `doubao`: doubao-seed-1-6-thinking-250715  
- `kimi`/`moonshot`: kimi-k3
- `openrouter`: google/gemini-3.5-flash (or openai/gpt-5.6-luna, anthropic/claude-sonnet-4.6)

> **Universal OpenRouter fallback**: when the configured `PROVIDER`'s key is
> missing but `OPENROUTER_API_KEY` is set, the LLM steps (approval,
> summarization, error/syntax analysis) transparently switch to `openrouter`
> via `Config.effective_provider()`. Set `MODEL` to a `provider/model` id for
> OpenRouter, e.g. `MODEL=openai/gpt-5.6-luna`.

### Usage

#### CLI entry (`cli.py`)

`cli.py` is the unified command-line entry for listing tools, calling each execution tool, and running end-to-end demos. It reuses the same tool implementations as the MCP server, so behavior matches.

```bash
# Overview and all subcommands
python cli.py --help

# List all execution tools
python cli.py list

# End-to-end offline demo (recommended first; no API key)
python cli.py demo

# Call a tool individually
python cli.py code --language python --code "print(2 ** 10)"
python cli.py shell "python3 --version"
python cli.py write --path notes.txt --content "hello" --overwrite
python cli.py edit --path notes.txt --search hello --replace world
```

Global flags (before the subcommand):

| Flag | Effect |
|------|------|
| `--provider` | Override LLM provider (`PROVIDER`) |
| `--workspace` | Override workspace directory (file ops restricted here) |
| `--no-approval` | Disable LLM pre-approval for dangerous ops |
| `--no-verify` | Disable auto syntax check for write/code |
| `--no-summarize` | Disable LLM summarization of long output (still truncates and persists) |

**Offline operation**: `list`, `demo`, and `code`/`shell`/`write`/`edit` with approval/summarize/non-Python verify off need no API key. API key is needed for: LLM pre-approval, LLM summarization of long output, non-Python syntax checks. `calendar` and `pr` also need their external credentials.

> **Warning — `--no-approval`**: this flag bypasses the LLM pre-approval check for dangerous operations. Use it only in controlled local demos (e.g. a throwaway workspace). Never combine it with real workspaces or destructive commands.
>
> **Long-output truncation and persistence**: when `code_interpreter` / `virtual_terminal` output exceeds the threshold (default 200 lines or 10000 characters), the tool keeps only the first and last 50 lines in context, writes the full output to a temp file, and returns the path in `stdout_file` / `stderr_file`. This path does **not** depend on an LLM and works offline.

#### Running the MCP Server

```bash
python server.py
```

#### Using with MCP Client

```python
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_tools():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Use file write tool
            result = await session.call_tool("file_write", {
                "path": "test.py",
                "content": "print('Hello, World!')"
            })

            # Use code interpreter
            result = await session.call_tool("code_interpreter", {
                "code": "import math\nprint(math.sqrt(16))"
            })

            # Use virtual terminal
            result = await session.call_tool("virtual_terminal", {
                "command": "ls -la"
            })


asyncio.run(use_tools())
```

#### Testing Individual Tools

```bash
# Test file operations
python test_file_tools.py

# Test execution tools
python test_execution_tools.py

# Test external integrations
python test_external_tools.py
```

### Architecture

The server implements a layered architecture:

1. **Safety Layer**: Intercepts dangerous operations and validates them
2. **Tool Layer**: Implements individual tool logic
3. **Verification Layer**: Validates outputs and provides feedback
4. **Integration Layer**: Connects to external services

### Examples

See `examples.py` for comprehensive usage examples.
