# 47-Human-in-the-loop

This example demonstrates a Human-in-the-Loop (HITL) agent workflow built around
LangGraph interrupts. The agent can search for products automatically, but a
purchase action pauses for human approval before it completes.

The code is intentionally modular:

| File | Purpose |
| --- | --- |
| `config.py` | Centralized settings, product catalog, and dict/dot-access config helpers. |
| `logger.py` | Structured logging and JSONL audit events. |
| `tools.py` | Product search and purchase tools with validation and HITL approval. |
| `agent.py` | LangGraph agent orchestration, tool routing, and checkpoint setup. |
| `cli.py` | Interactive terminal helpers for approval decisions. |
| `web.py` | FastAPI routes for chat, approvals, health checks, and pending approval management. |
| `hitl.py` | Main entry point for CLI, web, tests, and legacy demo mode. |
| `test_hitl.py` | Legacy regression tests for the demo components. |

## Flow

```text
User request
    |
    v
Agent reasoning
    |
    v
Tool call needed?
    |
    +-- search_product -> returns structured product candidates
    |
    +-- purchase_item -> interrupt -> human accept/edit/reject -> audit log
```

The important design boundary is that the model decides what should happen, but
purchase execution is gated by a human decision.

## Setup

From this folder:

```bash
pip install langchain langgraph langchain-openai fastapi uvicorn jinja2 python-multipart
```

Set an OpenAI key before running the real agent:

```bash
export OPENAI_API_KEY="openai-api-key"
```

The web module can be imported without an API key because `HITLAgent` is created
lazily. Actually running agent chat still requires `OPENAI_API_KEY`.

## Configuration

Preferred environment variables use the `HITL_` prefix:

| Variable | Default | Description |
| --- | --- | --- |
| `HITL_LLM_MODEL` | `gpt-4-mini` | Chat model used by `HITLAgent`. |
| `HITL_LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `HITL_LLM_MAX_TOKENS` | `1000` | Max response tokens. |
| `HITL_APPROVAL_TIMEOUT_HOURS` | `24` | Pending approval expiry window. |
| `HITL_WEB_INTERFACE` | `false` | Enables `hitl.py --web`. |
| `HITL_WEB_HOST` | `127.0.0.1` | Web host. |
| `HITL_WEB_PORT` | `8000` | Web port. |
| `HITL_LOG_LEVEL` | `INFO` | Logging level. |
| `HITL_AUDIT_LOG_FILE` | `hitl_audit.log` | Audit JSONL file. |

For notebook/test compatibility, `LLM_MODEL` and `APPROVAL_TIMEOUT_HOURS` are
also accepted as legacy aliases.

## Run

Run the main CLI shell:

```bash
python hitl.py
```

Run the legacy demo:

```bash
python hitl.py --legacy
```

Run the web API:

```bash
HITL_WEB_INTERFACE=true python hitl.py --web
```

Or directly with Uvicorn:

```bash
uvicorn web:app --host 127.0.0.1 --port 8000 --reload
```

If `templates/index.html` is not present, `/` returns a minimal API landing
page. Static files are mounted only when a `static/` directory exists.

## Tool Contracts

`search_product`

```python
search_product.invoke({"query": "MacBook"})
```

Returns a list of product dictionaries:

```python
[
    {
        "name": "MacBook Pro M3",
        "price": 2999.0,
        "vendor": "Apple Official Store",
        "currency": "US",
        "price_range": "15,999-25,999",
        "vendors": ["Apple Official Store", "JD.com", "AMAZON"],
    }
]
```

`purchase_item`

```python
purchase_item.invoke({
    "item": "MacBook Pro",
    "price": 15999.0,
    "vendor": "Apple Official Store",
    "thread_id": "demo-thread",
})
```

The tool calls `interrupt(...)` and accepts these resume decisions:

- `{"type": "accept"}` or `"approved"`: complete purchase.
- `{"type": "edit", "args": {"price": 14999.0}}`: modify and complete purchase.
- `{"type": "reject"}`: reject purchase.

Every approval decision is written to the audit log.

## Web API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic web/API landing page. |
| `POST` | `/api/chat` | Submit a chat message. |
| `POST` | `/api/approve` | Accept, reject, or edit a pending approval. |
| `GET` | `/api/approvals` | List pending approvals. |
| `DELETE` | `/api/approvals/{approval_id}` | Cancel an approval. |
| `GET` | `/api/health` | Health check. |

The current web chat endpoint is still demo-oriented and simulates agent output;
approval storage is in-memory. For production, replace it with persistent
storage and real graph resume handling.

## Verify

Compile all modules:

```bash
python -m compileall .
```

Lint source modules:

```bash
ruff check agent.py cli.py config.py hitl.py logger.py tools.py web.py
```

Run the legacy regression tests:

```bash
python -m unittest discover -s . -p "test_*.py"
```

Current status after optimization:

- Compile passes.
- Ruff passes for source modules.
- Most legacy tests pass.
- Two legacy-test issues may remain: one test expects direct control of the global audit logger file, and one test mocks `web.FastAPI` after `web` may already be imported.

## Notes

- `hitl_agent.log` is generated runtime output and can be deleted safely.
- This folder is currently untracked in git in the local workspace.
- The example is a teaching/demo system, not a complete production approval backend.
