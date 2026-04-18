# MCP (Model Context Protocol) Agent System

MCP example project with a FastAPI server, simple routing agent, LLM-style agent flow, and a LangGraph workflow. The current version is optimized around startup correctness, lower per-request overhead, and a shared concrete agent runtime reused across flows.

---

## Quick Start

### Installation

```bash

# Install core dependencies
pip install fastapi uvicorn pydantic requests

# Optional: For LangGraph agent
pip install langgraph langchain langchain-openai langchain-core

# Or install all at once
pip install -r requirements.txt
```

### Running the System

**Terminal 1: Start the MCP Server**
```bash
python mcp_server.py
```

Expected output:
```
2024-01-15 10:30:45,123 - root - INFO - ToolRegistry initialized
2024-01-15 10:30:45,124 - root - INFO - Tool registered: get_weather
2024-01-15 10:30:45,124 - root - INFO - Tool registered: search_docs
2024-01-15 10:30:45,124 - root - INFO - Tool registered: add_numbers
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Terminal 2: Run an Agent**

Choose one:
```bash
# Simple routing agent
python agent_client.py

# LLM-based agent
python llm_agent_flow.py

# Advanced LangGraph agent
python agent_langgraph.py
```

---

## API Usage

### Health Check
```bash
curl http://127.0.0.1:8000/health
# Response: {"status": "ok"}
```

### List Available Tools
```bash
curl http://127.0.0.1:8000/mcp/tools
```

### Call a Tool
```bash
curl -X POST http://127.0.0.1:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "arguments": {"city": "Boston", "unit": "celsius"}
  }'
```

Response:
```json
{
  "ok": true,
  "tool_name": "get_weather",
  "result": {
    "city": "Boston",
    "temperature": 22,
    "unit": "celsius",
    "condition": "sunny"
  }
}
```

---

## Architecture

### System Design (5-tier)

```
Presentation  agent_client.py / llm_agent_flow.py / agent_langgraph.py
    │
API Gateway   mcp_server.py         FastAPI, validation, error handling
    │
Tool Layer    tools.py              Tool registry, validation, invocation
    │
Config        config.py             Centralized constants
    │
Storage       (Ready for PostgreSQL/Redis)
```

### File Organization

| File | Purpose | Key Features |
|------|---------|--------------|
| `config.py` | Configuration management | Centralized constants, environment overrides |
| `tools.py` | Tool registry & handlers | Validation, error handling, logging |
| `mcp_server.py` | FastAPI server | Request validation, response schemas, health check |
| `agent_client.py` | Simple routing agent | Rule-based tool selection, timeout handling |
| `llm_agent_flow.py` | LLM-based agent | Shared runtime, concrete planner output |
| `agent_langgraph.py` | Advanced LangGraph agent | Explicit state graph over shared runtime |
| `agent_runtime.py` | Shared runtime | Shared planner, executor, answer formatting |
| `requirements.txt` | Dependencies | All required and optional packages |

---

## Available Tools

### 1. get_weather
Get current weather for a city.

**Input Schema**:
```json
{
  "city": "string (required)",
  "unit": "string (optional: 'celsius' or 'fahrenheit', default: 'celsius')"
}
```

**Output**:
```json
{
  "city": "string",
  "temperature": "number",
  "unit": "string",
  "condition": "string"
}
```

### 2. search_docs
Search documentation for relevant information.

**Input Schema**:
```json
{
  "query": "string (required)"
}
```

**Output**:
```json
{
  "query": "string",
  "results": ["string"],
  "count": "number"
}
```

### 3. add_numbers
Add two numbers together.

**Input Schema**:
```json
{
  "a": "number (required)",
  "b": "number (required)"
}
```

**Output**:
```json
{
  "a": "number",
  "b": "number",
  "result": "number"
}
```

---

## Configuration

Use environment variables or edit `config.py` to customize behavior:

```python
# Server settings
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 8000
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"

# API settings
REQUEST_TIMEOUT = 10  # seconds
MCP_PROTOCOL_VERSION = "1.0"

# Tool calling settings
TOOL_CALL_MAX_RETRIES = 3
TOOL_CALL_TIMEOUT = 30

# Search database (customize with your docs)
SEARCH_DOCS_DATABASE = [
    "MCP is a protocol for tool discovery and interaction.",
    "Tool calling usually relies on pre-defined function schemas.",
    # Add your documents here
]

# LLM settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
```

Example:

```bash
export MCP_SERVER_PORT=8010
export REQUEST_TIMEOUT=20
export LLM_MODEL=gpt-4o-mini
```

---
