# MCP (Model Context Protocol) Agent System

A compact MCP example with a FastAPI tool server, three agent entry points, and a streaming `BaseAgent` abstraction backed by LangGraph.

---

## Quick Start

```bash
pip install -r requirements.txt
```

Minimal install (server + simple clients only):
```bash
pip install fastapi uvicorn pydantic requests
```

With LangGraph agent:
```bash
pip install langgraph langchain langchain-openai langchain-core
```

Start the server, then run any client:

```bash
# Terminal 1
python mcp_server.py

# Terminal 2 — pick one
python agent_client.py
python llm_agent_flow.py
python agent_langgraph.py
```

---

## Architecture

```
agent_client.py / llm_agent_flow.py / agent_langgraph.py
          │
    agent_runtime.py          shared planner, prompt builder, answer formatter
    base_agent.py             streaming queue, BaseAgent.stream() / .invoke()
          │
    mcp_server.py             FastAPI: /mcp/tools, /mcp/call, /health
          │
    tools.py                  ToolRegistry + three built-in tool handlers
          │
    config.py                 env-overridable constants
```

| File | Role |
|---|---|
| `config.py` | Shared constants; all values env-overridable |
| `tools.py` | `ToolRegistry` with catalog caching; `get_weather`, `search_docs`, `add_numbers` |
| `mcp_server.py` | FastAPI endpoints for tool discovery and execution |
| `agent_client.py` | Schema-driven routing client with cached HTTP session and three-tier discovery |
| `agent_runtime.py` | Dynamic planner, cached prompt serialization, `run_agent_query()` |
| `base_agent.py` | `QueueEvent`, `AgentThought`, `AgentResult`, `BaseAgent` (ABC) |
| `llm_agent_flow.py` | CLI wrapper over `run_agent_query()` |
| `agent_langgraph.py` | `MCPLangGraphAgent` — explicit graph nodes publishing typed events |

---

## API

```bash
# Health (returns status + tools_registered count)
curl http://127.0.0.1:8000/health

# Unified manifest (capabilities + tools in one call)
curl http://127.0.0.1:8000/mcp/manifest

# Discover tools (REST)
curl http://127.0.0.1:8000/mcp/tools

# Discover tools (SSE stream — receives ready/capabilities/tools events then heartbeats)
curl -N -H "Accept: text/event-stream" http://127.0.0.1:8000/mcp/events

# Call a tool
curl -X POST http://127.0.0.1:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"name": "get_weather", "arguments": {"city": "Boston", "unit": "celsius"}}'
```

Optional API key auth (set `MCP_API_KEY` env var on the server):
```bash
curl -H "X-MCP-API-Key: $MCP_API_KEY" http://127.0.0.1:8000/mcp/manifest
```

---

## Tools

All tools embed a `display_hint` in their response payload so the formatter decouples rendering from tool name.

### `get_weather`
```json
{ "city": "string (required, ≤100 chars)", "unit": "celsius | fahrenheit" }
→ { "city": "string", "temperature": number, "unit": "string", "condition": "string", "display_hint": "weather" }
```

### `search_docs`
```json
{ "query": "string (required, ≤500 chars)" }
→ { "query": "string", "results": ["string"], "count": number, "display_hint": "search_results" }
```

### `add_numbers`
```json
{ "a": number (required), "b": number (required) }
→ { "a": number, "b": number, "result": number, "display_hint": "arithmetic" }
```

All parameters marked required are validated before the handler runs; missing params raise `ValueError` rather than silently defaulting.

---

## Streaming LangGraph Agent

`MCPLangGraphAgent` (in `agent_langgraph.py`) extends `BaseAgent` from `base_agent.py`.

**Graph nodes:** `discover → plan →(conditional)→ execute → plan (loop) → respond`

The `execute → plan` back-edge enables multi-step tool chaining: after each tool call the planner re-evaluates whether another tool is needed. The planner terminates the chain once `tool_history` is non-empty, then `respond_node` reconstructs the effective decision from the last history entry.

**Routing from `plan`:**
- `tool_request` present → `execute`
- direct answer or `tool_history` non-empty → `respond`
- max iterations reached → `END` (skips `respond` entirely)

**Event types published per node:**

| Node | Events |
|---|---|
| `discover` | `AGENT_THOUGHT` |
| `plan` | `AGENT_THOUGHT` or `AGENT_MESSAGE + AGENT_END` (on max-iter) |
| `execute` | `TOOL_CALL`, `TOOL_RESULT` (or `TOOL_RESULT` with error observation) |
| `respond` | `AGENT_MESSAGE`, `AGENT_END` |

**`BaseAgent` guarantees:**
- `stream(input)` → `Iterator[AgentThought]`; graph runs on a daemon thread
- `invoke(input)` → `AgentResult`; accumulates all thoughts including deduped `AGENT_MESSAGE` chunks
- Any unhandled graph exception is caught in the thread and published as `ERROR + AGENT_END`, so `stream()` always terminates

---

## Tool Discovery — Three-Tier Fallback

`discover_tools()` in `agent_client.py` implements the MCP plug-and-play principle: connect to any conforming server and get working tool calls without custom wiring.

| Tier | Endpoint | When used |
|---|---|---|
| 1 | `GET /mcp/manifest` | Preferred — returns capabilities + tools in one round-trip |
| 2 | `GET /mcp/tools` | REST fallback for servers without a manifest endpoint |
| 3 | `GET /mcp/events` | SSE fallback — reads the `tools` event then closes the stream |

The SSE path (`_discover_tools_via_sse`) makes the client compatible with any MCP server that supports streaming, without knowing its REST API surface in advance.

---

## Dynamic Schema-Driven Routing

`choose_tool()` in `agent_client.py` selects tools without hardcoded keyword rules. For each discovered tool it:

1. Builds intent hints from the tool name, description, and schema property names
2. Scores overlap between query tokens and hints (`_score_tool_match`)
3. Requires a minimum score of 2 (`_MIN_TOOL_MATCH_SCORE`) before routing
4. Builds arguments dynamically from the schema (`_build_tool_arguments`)

This means adding a new tool to the registry makes it immediately discoverable and routable with no client changes.

---

## Shared Runtime (`agent_runtime.py`)

`decide_next_action` delegates to `choose_tool` for schema-driven selection and uses `tool_history` to detect when chaining is complete.

`_serialize_tools_for_prompt` is `lru_cache`-backed — repeated calls with the same tool catalog produce no extra JSON serialization work.

`format_agent_answer` dispatches on `display_hint` embedded in the tool result, not on the tool name — so renaming or replacing a tool doesn't require updating the formatter.

---

## Configuration

```python
MCP_SERVER_HOST = "127.0.0.1"      # MCP_SERVER_HOST
MCP_SERVER_PORT = 8000             # MCP_SERVER_PORT
REQUEST_TIMEOUT = 10               # REQUEST_TIMEOUT (seconds)
TOOL_CALL_MAX_RETRIES = 3          # TOOL_CALL_MAX_RETRIES
TOOL_CALL_TIMEOUT = 30             # TOOL_CALL_TIMEOUT (seconds)
LLM_MODEL = "gpt-4o-mini"         # LLM_MODEL
LLM_TEMPERATURE = 0.0              # LLM_TEMPERATURE
SSE_HEARTBEAT_SECONDS = 15         # SSE_HEARTBEAT_SECONDS
```

---

## Verification

```bash
ruff check .
python -m compileall .
```

---

## Future Improvements

- Replace the deterministic planner with a real LLM call
- Add automated tests (unit for `tools.py`, event-stream tests for `base_agent.py`, integration with server running)
- Rate limiting, metrics, and tracing on the API server

---

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
