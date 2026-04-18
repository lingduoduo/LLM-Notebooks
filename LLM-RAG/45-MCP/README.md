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
| `agent_client.py` | Simple rule-based client with cached HTTP session and discovery |
| `agent_runtime.py` | `PLANNER_RULES`, cached prompt serialization, `run_agent_query()` |
| `base_agent.py` | `QueueEvent`, `AgentThought`, `AgentResult`, `BaseAgent` (ABC) |
| `llm_agent_flow.py` | CLI wrapper over `run_agent_query()` |
| `agent_langgraph.py` | `MCPLangGraphAgent` — explicit graph nodes publishing typed events |

---

## API

```bash
# Health (returns status + tools_registered count)
curl http://127.0.0.1:8000/health

# Discover tools
curl http://127.0.0.1:8000/mcp/tools

# Call a tool
curl -X POST http://127.0.0.1:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"name": "get_weather", "arguments": {"city": "Boston", "unit": "celsius"}}'
```

---

## Tools

### `get_weather`
```json
{ "city": "string (required, ≤100 chars)", "unit": "celsius | fahrenheit" }
→ { "city": "string", "temperature": number, "unit": "string", "condition": "string" }
```

### `search_docs`
```json
{ "query": "string (required, ≤500 chars)" }
→ { "query": "string", "results": ["string"], "count": number }
```

### `add_numbers`
```json
{ "a": number (required), "b": number (required) }
→ { "a": number, "b": number, "result": number }
```

All parameters marked required are validated before the handler runs; missing params raise `ValueError` rather than silently defaulting.

---

## Streaming LangGraph Agent

`MCPLangGraphAgent` (in `agent_langgraph.py`) extends `BaseAgent` from `base_agent.py`.

**Graph nodes:** `discover → plan →(conditional)→ execute → respond`

**Routing from `plan`:**
- `tool_request` present → `execute`
- direct answer → `respond`
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

## Shared Runtime (`agent_runtime.py`)

`PLANNER_RULES` is a single source of truth for keyword-to-tool routing used by all three agent entry points.

`_serialize_tools_for_prompt` is `lru_cache`-backed — repeated calls with the same tool catalog produce no extra JSON serialization work.

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
- Multi-step tool chaining across turns

---

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
