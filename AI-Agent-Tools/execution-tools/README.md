# Execution Tools MCP Server

An MCP (Model Context Protocol) server that provides execution tools with layered safety mechanisms for AI agents.

This project corresponds to Experiment 4-2 in the book's "Execution Tools" section. It focuses on layered safety (input validation, permission control, LLM pre-approval), automatic verification with feedback loops, and truncation plus persistence of long outputs. Recommended start: `python cli.py demo`.

### Features

#### Safety Mechanisms

1. **LLM-Based Approval**: Destructive and irreversible operations are reviewed by a separate LLM before execution. Detection is token-based, not substring-based: commands are parsed into their command word and flags, and source code is matched with word-boundary patterns keyed on a canonical language name, so aliases such as `python3` or `sh` cannot reach an empty rule table.
2. **Automatic Verification**: Python is checked locally with `compile()`. Other languages are checked by their own compiler or interpreter at execution time. Verification reports one of `passed`, `failed`, `unverified` or `skipped` — it never claims a check that did not happen.
3. **Long-Output Handling**: Output over the threshold is reduced for the agent's context — summarized by an LLM when enabled, otherwise trimmed to a head and tail — while the complete text is written to a file whose path is returned.
4. **Error Analysis**: Failed executions come back with an `error_analysis` field explaining the likely root cause. Disable with `AUTO_ANALYZE_ERRORS=false`.

> **Isolation**: code runs in a fresh temporary directory as a subprocess of the server, on the host. That is *not* a security sandbox: there are no resource limits, no network restrictions, and no privilege separation. The provided `Dockerfile` is the isolation boundary — run untrusted code inside the container, not on a workstation.

#### Tool Categories

##### File System Tools
- **file_write**: Write a file, with syntax verification. An existing file is only replaced when `overwrite=true`, and replacing one requires approval.
- **file_edit**: Search-and-replace edit with a diff preview and verification.
- **fs_read_file**, **fs_read_multiple_files**: Read text files with a size limit.
- **fs_list_directory**, **fs_directory_tree**, **fs_search_files**, **fs_get_file_info**: Inspect the workspace.
- **fs_move**, **fs_copy**, **fs_delete**, **fs_create_directory**: Mutate the workspace. Deleting, and replacing an existing destination, require approval.
- **fs_list_allowed_directories**: Report the directories file operations are confined to.

##### Generic Execution Tools
- **code_interpreter**: Execute code in Python, JavaScript, TypeScript, Go, Java, C++, Rust, PHP or Bash in a temporary directory.
- **virtual_terminal**: Execute a shell command with dangerous-command detection.

##### Stateful Terminal Session
- **terminal_execute**: Run a command in the session's current directory; unlike `virtual_terminal` this keeps a working directory and history across calls.
- **terminal_pwd**, **terminal_cd**, **terminal_history**: Manage the session.
- **terminal_insert_lines**, **terminal_delete_lines**, **terminal_update_line**: Line-level file edits.

##### External System Integration Tools
- **google_calendar_add**: Add events to Google Calendar.
- **github_create_pr**: Create GitHub pull requests with branch validation.

`server.py` and `cli.py` both build from `tool_registry.py`, so the MCP tool list and the CLI can never disagree.

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
# Direct OpenAI configuration (safety checks, summarization, error analysis)
OPENAI_API_KEY=your_openai_api_key

# Optional model override; defaults to gpt-5.6
# MODEL=gpt-5.6

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
AUTO_ANALYZE_ERRORS=true
```

The LLM steps use the OpenAI API directly. `MODEL` defaults to `gpt-5.6` and accepts any OpenAI model id.

### Usage

#### CLI entry (`cli.py`)

`cli.py` is the unified command-line entry for listing tools, calling each execution tool, and running an end-to-end demo. It reuses the same tool registry as the MCP server, so behavior matches.

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
| `--workspace` | Override workspace directory (file ops restricted here) |
| `--no-approval` | Disable LLM pre-approval for dangerous ops |
| `--no-verify` | Disable auto syntax check for write/code |
| `--no-summarize` | Disable LLM summarization of long output (still truncates and persists) |

**Offline operation**: `list`, `demo`, and `code`/`shell`/`write`/`edit` need no API key. An API key is needed for LLM pre-approval, summarization of long output, and error analysis. `calendar` and `pr` also need their external credentials.

> **Warning — `--no-approval`**: this flag bypasses the LLM pre-approval check for dangerous operations. Use it only in controlled local demos (e.g. a throwaway workspace). Never combine it with real workspaces or destructive commands.
>
> **Long-output truncation and persistence**: when a tool's output exceeds the threshold (default 200 lines or 10,000 characters), the complete output is written to a file and the path is returned in `stdout_file` / `stderr_file`. With `AUTO_SUMMARIZE_COMPLEX_OUTPUT=true` the context gets an LLM summary of the *full* text; otherwise it gets the first and last 50 lines. Persistence never depends on an LLM and works offline. Saved outputs live in a process-owned temporary directory and are pruned to the most recent 50, then removed at exit.

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

##### Passing auxiliary files

`code_interpreter` accepts a `files` mapping. Content is written as UTF-8 text
verbatim; binary payloads must opt in with an explicit `base64:` prefix.

```python
await session.call_tool("code_interpreter", {
    "code": "print(open('input.txt').read())",
    "files": {
        "input.txt": "plain text stays plain text",
        "blob.bin": "base64:aGVsbG8gYmluYXJ5",
    },
})
```

#### Running the tests

```bash
python -m pytest
```

Tests run offline and use an isolated temporary workspace, so they never touch the repository. Language cases whose toolchain is not installed are skipped.

### Architecture

The server implements a layered architecture:

1. **Safety Layer** (`safety.py`): tokenizes commands and matches code patterns to decide what needs approval
2. **Tool Layer** (`file_tools.py`, `execution_tools.py`, `filesystem_enhanced.py`, `terminal_controller.py`, `external_tools.py`): individual tool logic
3. **Verification Layer** (`llm_helper.py`): validates syntax and reports honestly when it could not
4. **Registry** (`tool_registry.py`): the single source of truth for the tool surface, consumed by both entry points

### Examples

See `examples.py` for comprehensive usage examples.
