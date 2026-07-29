# Event-Triggered AI Agent with MCP Tools

> Companion code for *AI Agents in Depth*, Chapter 4 — **Experiment 4-4 ★★★**. A FastAPI event-driven Agent that loads MCP tools asynchronously from the collaboration / execution / perception servers.

← [Chapter 4 index](../README.md)

---

A modern AI agent with **native async support** that responds to events from various sources. Built with **FastAPI**, powered by **OpenAI** models, and integrated with **118 MCP tools** for extra capabilities including browser automation, web search, and document processing.

## Features

### Core Capabilities
- ✅ **Native Async** — FastAPI with clean async/await support
- ✅ **118 MCP Tools** — Automatically loaded from 3 MCP servers (see `/mcp/status` for the live count)
- ✅ **Event-Driven** — Responds to web messages, emails, GitHub updates, timers
- ✅ **System Hints** — Timestamps, tool counters, TODO management
- ✅ **Auto API Docs** — Interactive Swagger UI at `/docs`
- ✅ **Background Tasks** — Process monitoring and system alerts

### MCP Tool Categories

**Collaboration Tools** (41 tools):
- Browser automation (navigate, screenshot, execute tasks)
- Notifications (email, Telegram, Slack, Discord)
- Human-in-the-loop (admin approval, input requests)
- Timer management (one-time, recurring)

**Execution Tools** (24 tools):
- File operations (write, edit with verification)
- Code execution (Python interpreter, shell commands)
- External integrations (Google Calendar, GitHub PRs)

**Perception Tools** (53 tools):
- Web search and content extraction
- Document reading (PDF, DOCX, PPTX)
- Multimodal parsing (images, videos, webpages)
- Public data (weather, stocks, Wikipedia, ArXiv)
- Private data (Google Calendar, Notion)

## Model Configuration

Everything in this project talks to the OpenAI API through the official
`openai` SDK. Configuration lives in a `.env` file next to the code, which every
entry point loads automatically (real environment variables win over `.env`):

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | no | `gpt-5.2` | Model override, e.g. `gpt-5.1`, `gpt-4.1` |
| `OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | Point at any OpenAI-compatible endpoint |
| `AGENT_PORT` | no | `8000` | Server port |
| `ENABLE_MCP_TOOLS` | no | `true` | Set to `false` for built-in tools only |
| `AGENT_HOST` | no | `127.0.0.1` | Bind address. See Security below before changing it |
| `USER_TIMEOUT_SECONDS` | no | `60` | Idle time before a `user_timeout` reminder |
| `PROCESS_TIMEOUT_SECONDS` | no | `30` | Runtime before a `process_timeout` alert |
| `MONITOR_INTERVAL_SECONDS` | no | `10` | How often the monitor checks |

> **Reasoning models**: the GPT-5 family and the o-series take
> `max_completion_tokens` rather than `max_tokens`, and several of them reject
> any non-default `temperature`. `agent.py` detects these models
> (`is_reasoning_model`) and shapes each request accordingly, so you can switch
> between `gpt-5.2` and `gpt-4.1` without changing any other code. The
> `temperature` setting therefore only applies to the non-reasoning models.

> **Model must support function tools.** This agent performs every action
> through function tools, so the model has to support them on
> `/v1/chat/completions`. The `gpt-5.6-*` models (`terra`, `luna`, `sol`) do
> **not** — they reject function tools there unless reasoning is turned off
> entirely — which is why the default is `gpt-5.2`. `agent.py` logs a warning
> at startup if you configure one of them.

## Event-driven demo (runs offline, no API key)

Before starting the HTTP server, run `event_loop_demo.py`. In one process it
demonstrates the core idea of the chapter — **the external world wakes the
Agent**. The script registers three trigger types; background threads push
structured events onto a shared queue when something happens; the event loop
dequeues them and wakes the Agent — "register → trigger → wake → handle":

| Trigger | Class | Concept |
|---|---|---|
| One-shot timer | `OneShotTimer` | `set_timer`, one-shot (e.g. "call the DMV at 10:00 on Monday") |
| Recurring timer | `RecurringTimer` | `set_timer`, recurring / heartbeat (e.g. "check the server hourly") |
| File watch | `FileWatchTrigger` | File-change triggers like the ones n8n offers |

`--mock` offline mode makes no LLM calls; it prints simulated actions whenever
the Agent is woken — **no API key required**:

```bash
# Offline: all triggers (one-shot + recurring + file watch)
python event_loop_demo.py --mock

# One-shot only; fire after 2s; run 6s total
python event_loop_demo.py --mock --trigger timer --delay 2 --duration 6

# Recurring every 3s
python event_loop_demo.py --mock --trigger recurring --interval 3 --duration 12

# Watch a directory. A simulated external writer drops a file in after a few
# seconds so the demo shows something on its own.
python event_loop_demo.py --mock --trigger file --watch-dir watched_dir

# Drive the directory by hand instead (other terminal: echo hello > watched_dir/a.txt)
python event_loop_demo.py --mock --trigger file --no-auto-write --duration 30
```

If a run ends with `handled 0 event(s)`, nothing triggered it — with
`--no-auto-write` you have to create or modify a file inside the watched
directory while the loop is still running.

Sample offline output (excerpt):

```
⏱️  [OneShotTimer(daily_backup_check)] registered: fires in 2s
🔁 [RecurringTimer(health_check)] registered: fires every 3s
🟢 Event loop started; running for 8s, waiting for events to wake the Agent...
⚡ [OneShotTimer(daily_backup_check)] fired event -> timer_trigger: One-shot timer fired: check whether the daily backup has finished.
📥 Event loop dequeued event #1 -> waking the Agent
🤖 Agent woken, received message: [Timer daily_backup_check triggered] One-shot timer fired: check whether the daily backup has finished.
🛠️  [simulated action] load the scheduled-task context -> run the routine check -> report back
✅ Agent finished: responded to the timer_trigger event
```

Drop `--mock` to use a real model (built-in tools only by default, no MCP):

```bash
export OPENAI_API_KEY='sk-...'          # or put it in .env
python event_loop_demo.py --trigger timer

# Pick a different model for this run
python event_loop_demo.py --trigger timer --model gpt-4.1
```

Full flags: `python event_loop_demo.py --help`.

## Quick Start

### Installation

```bash
cd AI-Agent-Tools/agent-with-event-trigger

# Install dependencies (includes FastAPI, uvicorn, MCP SDK)
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env and add your OpenAI API key
```

### Start the Server

```bash
python server.py
```

CLI flags override env vars; see `python server.py --help`:

```bash
python server.py --port 9000              # custom port
python server.py --model gpt-4.1          # pick a different OpenAI model
python server.py --no-mcp                 # built-in tools only, no MCP
```

Output:
```
🤖 EVENT-TRIGGERED AGENT SERVER (FastAPI)
✅ Starting server on 0.0.0.0:8000
📡 API Documentation: http://localhost:8000/docs
📊 ReDoc: http://localhost:8000/redoc

🚀 Starting Event-Triggered Agent Server (FastAPI)
✅ Agent initialized with OpenAI model: gpt-5.2
🔄 MCP tools enabled (default) - loading asynchronously...
✅ Discovered tools from 'collaboration': 41 tools
✅ Discovered tools from 'execution': 24 tools
✅ Discovered tools from 'perception': 53 tools
✅ MCP tools loaded: 118 tools available
✅ Server ready to receive events

INFO: Uvicorn running on http://0.0.0.0:8000
```

### Interactive API Documentation

Visit **http://localhost:8000/docs** to:
- 📖 Browse all available endpoints
- 🧪 Test API calls interactively
- 📝 See request/response schemas
- ⚡ Send events with one click

## API Endpoints

### Core Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Check MCP tools status
curl http://localhost:8000/mcp/status

# Send an event
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "web_message",
    "content": "Search the web for Python async best practices",
    "metadata": {"user": "demo"}
  }'

# Get agent status
curl http://localhost:8000/agent/status

# Reset agent state
curl -X POST http://localhost:8000/agent/reset

# Reload MCP tools
curl -X POST http://localhost:8000/mcp/reload

# Start / stop the background monitor that raises system-reminder events
curl -X POST http://localhost:8000/monitoring/start
curl -X POST http://localhost:8000/monitoring/stop
```

### System reminders

`POST /monitoring/start` runs a background monitor that turns *absence* of
activity into events — the thing a request/response API cannot express:

- no user interaction for `USER_TIMEOUT_SECONDS` raises a `user_timeout` event
- a process registered via `/process/register` that has run longer than
  `PROCESS_TIMEOUT_SECONDS` raises a `process_timeout` event

Each fires once per occurrence: the user reminder re-arms only after the user
interacts again, and each process is flagged so it is reported once. The
monitor is **off until you start it**, because every reminder it raises costs
an LLM call.

### Using the Interactive Docs

1. Open http://localhost:8000/docs
2. Click on any endpoint (e.g., `POST /event`)
3. Click "Try it out"
4. Fill in the request body
5. Click "Execute"
6. See the response instantly

## Usage Examples

### Running the Standalone Example

For a complete demonstration of MCP integration without the server, run:

```bash
python example_with_mcp.py
```

This standalone script:
- Initializes the agent with MCP tools enabled
- Loads every tool from the 3 MCP servers
- Processes a sample event (web search task)
- Shows the complete flow from tool discovery to execution
- Properly cleans up MCP connections

Useful for:
- Testing MCP integration without running a server
- Understanding the async tool loading flow
- Debugging MCP connection issues
- Learning how to use the agent programmatically

### Example 1: Web Search Task

```bash
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "web_message",
    "content": "Search for the latest FastAPI features and summarize them",
    "metadata": {"user": "demo"}
  }'
```

The agent will:
1. Use `perception_web_search` to find results
2. Parse the content with `perception_webpage_reader`
3. Summarize findings in the response

### Example 2: Browser Automation

```bash
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "web_message",
    "content": "Navigate to example.com and take a screenshot",
    "metadata": {}
  }'
```

Uses:
- `collaboration_mcp_browser_navigate`
- `collaboration_mcp_browser_screenshot`

### Example 3: Document Processing

```bash
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "web_message",
    "content": "Download and summarize the PDF from https://example.com/doc.pdf",
    "metadata": {}
  }'
```

Uses:
- `perception_download`
- `perception_document_reader`

### Example 4: Email Notification

```bash
curl -X POST http://localhost:8000/event \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "timer_trigger",
    "content": "Send daily report to admin@example.com",
    "metadata": {"scheduled": true}
  }'
```

Uses:
- `collaboration_mcp_send_email`

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_MODEL="gpt-5.2"           # Override the default model
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AGENT_PORT="8000"                # Server port (default: 8000)
export ENABLE_MCP_TOOLS="true"          # Enable MCP (default: true)
```

### Disable MCP Tools

If you only want built-in tools:

```bash
ENABLE_MCP_TOOLS=false python server.py
```

### Custom Port

```bash
AGENT_PORT=9000 python server.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│                   (Native Async)                         │
└───────────┬─────────────────────────────────────────────┘
            │
            ├─► Lifespan Events (Startup/Shutdown)
            │   └─► Load MCP Tools Asynchronously
            │
            ├─► Event Handler (Process incoming events)
            │   └─► EventTriggeredAgent  ──► OpenAI Chat Completions
            │       ├─► System Hints (timestamps, TODOs)
            │       ├─► Tool Execution (MCP + built-in)
            │       └─► Trajectory Saving
            │
            └─► MCP Server Manager
                ├─► Collaboration Tools (41 tools)
                ├─► Execution Tools (24 tools)
                └─► Perception Tools (53 tools)
```

## Project Structure

```
agent-with-event-trigger/
├── agent.py                 # Event-triggered agent with system hints (OpenAI client)
├── event_types.py           # Event type definitions
├── event_loop_demo.py       # Offline event-loop demo (timer / recurring / file-watch triggers)
├── server.py                # FastAPI server (main entry point)
├── server_fastapi.py        # Same server without the CLI wrapper
├── client.py                # Test client for sending events
├── quickstart.py            # Starts the server and prints next steps
├── example_with_mcp.py      # Standalone MCP example
├── test_demo.py             # Scripted three-event demo
├── requirements.txt         # Dependencies (FastAPI, uvicorn, MCP, openai)
├── env.example              # Environment template
└── README.md                # This file
```

## MCP Tools Reference

### Check Available Tools

```bash
curl http://localhost:8000/mcp/status
```

Response shows:
- `tools`: List of every tool name
- `tools_by_server`: Tools grouped by server
- `tools_count`: Total count
- `loaded`: Whether MCP tools are active

### Tool Naming Convention

MCP tools use underscore prefixes:
- `collaboration_*` — Collaboration tools
- `execution_*` — Execution tools
- `perception_*` — Perception tools

Built-in tools (no prefix):
- `read_file`
- `write_file`
- `code_interpreter`
- `execute_command`
- `rewrite_todo_list`
- `update_todo_status`

## Event Types

```python
class EventType(Enum):
    # External input events
    WEB_MESSAGE = "web_message"           # Web interface
    IM_MESSAGE = "im_message"             # Instant messaging
    EMAIL_REPLY = "email_reply"           # Email responses
    GITHUB_PR_UPDATE = "github_pr_update" # PR notifications
    TIMER_TRIGGER = "timer_trigger"       # Scheduled tasks (one-shot / recurring)
    FILE_CHANGE = "file_change"           # File watch trigger (created / modified)

    # System reminder events
    USER_TIMEOUT = "user_timeout"         # No user activity
    PROCESS_TIMEOUT = "process_timeout"   # Long-running process
    SYSTEM_ALERT = "system_alert"         # System warnings
```

### Event Format

```json
{
  "event_type": "web_message",
  "content": "Your task description",
  "metadata": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

## Using the Client

The included client provides easy testing:

```bash
# Interactive mode
python client.py --mode interactive

# Test scenarios
python client.py --mode test

# Send a single event (defaults to web_message)
python client.py --message "Create a Python hello world script"

# Send a single event of a specific type
python client.py --event-type timer_trigger --message "Check the daily backup"
```

## Security Considerations

### What this agent can do to your machine

`/event` has no authentication and drives tools that run arbitrary shell
commands (`execute_command`) and arbitrary Python (`code_interpreter`, a plain
`exec()` with no sandbox). Anything that can POST to the port can run code as
you. Two consequences:

- **The server binds `127.0.0.1` by default.** Set `AGENT_HOST=0.0.0.0` only on
  a network you trust, and put authentication in front of it first.
- **Destructive shell commands are refused by default.** `execute_command`
  blocks commands that move git branches, discard uncommitted work, or delete
  files recursively, and tells the model to inspect and report instead.

The guard exists because of a real incident, not a hypothetical one. Handling
the synthetic "GitHub PR #42 review" event from `client.py --mode test`, the
agent decided the helpful thing to do was:

```
git fetch origin pull/42/head:pr-42 && git stash push -u && git checkout pr-42
```

on a live repository, moving the working tree off the branch its user was on.
Nothing was lost — the changes were in the stash — but nothing warned about it
either. Set `SystemHintConfig(allow_destructive_commands=True)` to opt out.

This is a guard rail, not a security boundary: a shell tool cannot be made safe
by pattern matching. For anything beyond local experimentation, run the agent in
a container or VM whose destruction you do not mind.

### For production deployment

1. **HTTPS**: Use a reverse proxy (nginx, Caddy)
2. **Authentication**: Add API key validation — `/event` has none
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Sanitize all inputs
5. **CORS**: Configure allowed origins
6. **Secrets**: Keep `.env` out of version control (it is gitignored here) and use a secrets manager in production
7. **Isolation**: Run in a sandbox — the tools have your privileges

## Comparison: Flask vs FastAPI

| Feature | Old (Flask) | New (FastAPI) |
|---------|-------------|---------------|
| Framework | Flask (WSGI) | FastAPI (ASGI) |
| Async Support | ❌ Threads | ✅ Native async/await |
| MCP Integration | ⚠️ Complex | ✅ Clean |
| API Docs | ❌ Manual | ✅ Auto-generated |
| Performance | Good | **Better** (2-3x) |
| Port | 4242 | 8000 |
| Deprecation Warnings | N/A | ✅ Fixed (lifespan) |

## Troubleshooting

### Missing or Invalid API Key

```bash
# The agent reads OPENAI_API_KEY from the environment or the .env file
grep OPENAI_API_KEY .env

# Verify the key works and see which models it can reach
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Unsupported Parameter Errors

If a custom model rejects `max_completion_tokens` or `temperature=1`, check
whether `is_reasoning_model()` in `agent.py` classifies it correctly — it keys
off the `gpt-5*` / `o1` / `o3` / `o4` name prefixes.

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
AGENT_PORT=8001 python server.py
```

### MCP Tools Not Loading

```bash
# Check status
curl http://localhost:8000/mcp/status

# Look for error in response
# Common issue: Missing API keys for MCP servers

# Reload tools
curl -X POST http://localhost:8000/mcp/reload
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify FastAPI installed
python -c "import fastapi; print(fastapi.__version__)"
```

### Agent Not Responding

```bash
# Check health
curl http://localhost:8000/health

# View logs in server terminal
# Check trajectory file: event_agent_trajectory.json
```

## Tests

```bash
python -m pytest test_command_guard.py -q # destructive-command guard
python -m pytest test_env_int.py -q       # AGENT_PORT parsing
python -m unittest test_code_interpreter -v   # code interpreter namespace
python -m pytest test_openai_config.py -q # OpenAI model/parameter handling
```

## License

MIT License — See LICENSE file for details

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- MCP protocol by [Model Context Protocol](https://modelcontextprotocol.io/)
- Tool servers: [collaboration-tools](../collaboration-tools/), [execution-tools](../execution-tools/), [perception-tools](../perception-tools/)

---

## Notes

- Start with `python event_loop_demo.py --mock` (no API key needed).
- MCP servers expected: [collaboration-tools](../collaboration-tools/), [execution-tools](../execution-tools/), [perception-tools](../perception-tools/).
