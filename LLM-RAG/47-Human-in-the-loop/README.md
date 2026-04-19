# 47-Human-in-the-Loop

Human-in-the-loop (HITL) agent example built with LangGraph interrupts.

The agent can search for products automatically, but purchase execution is
paused until a human accepts, edits, or rejects the request. This keeps the LLM
in charge of reasoning while leaving sensitive actions under explicit human
control.

## Architecture

```text
User message
    |
    v
HITLAgent
    |
    +-- search_product -> returns product candidates
    |
    +-- purchase_item -> interrupt -> human decision -> audit log
```

| File | Role |
| --- | --- |
| `config.py` | Environment settings, product catalog, and config helpers. |
| `tools.py` | Search and purchase tools, validation, and tool dispatch. |
| `agent.py` | LangGraph state, routing, checkpointing, and tool execution. |
| `cli.py` | Terminal UI helpers for chat and approval decisions. |
| `web.py` | FastAPI endpoints for chat, approvals, health, and cleanup. |
| `logger.py` | Runtime logging and JSONL audit trail. |
| `hitl.py` | Main entry point for CLI, web, tests, and legacy demo mode. |
| `test_hitl.py` | Regression tests for config, tools, agent, CLI, and web models. |

## Setup

From this directory:

```bash
cd LLM-RAG/47-Human-in-the-loop
pip install langchain langgraph langchain-openai fastapi uvicorn jinja2 python-multipart
```

Set an OpenAI API key before running the real agent:

```bash
export OPENAI_API_KEY="openai-api-key"
```

`web.py` imports without an API key because the agent is created lazily. Running
real agent chat still requires `OPENAI_API_KEY`.

## Run

Start the interactive CLI:

```bash
python hitl.py
```

Run the legacy LangGraph demo:

```bash
python hitl.py --legacy
```

Run the FastAPI web app:

```bash
HITL_WEB_INTERFACE=true python hitl.py --web
```

Or run it directly with Uvicorn:

```bash
uvicorn web:app --host 127.0.0.1 --port 8000 --reload
```

If `templates/index.html` is missing, `/` returns a minimal API landing page.
Static files are mounted only when a `static/` directory exists.

## Configuration

Preferred variables use the `HITL_` prefix.

| Variable | Default | Description |
| --- | --- | --- |
| `HITL_LLM_MODEL` | `gpt-4o-mini` | Chat model used by `HITLAgent`. |
| `HITL_LLM_TEMPERATURE` | `0.2` | Sampling temperature. |
| `HITL_LLM_MAX_TOKENS` | `1000` | Maximum response tokens. |
| `HITL_APPROVAL_TIMEOUT_HOURS` | `24` | Pending approval expiry window. |
| `HITL_WEB_INTERFACE` | `false` | Enables `python hitl.py --web`. |
| `HITL_WEB_HOST` | `127.0.0.1` | Web server host. |
| `HITL_WEB_PORT` | `8000` | Web server port. |
| `HITL_LOG_LEVEL` | `INFO` | Runtime log level. |
| `HITL_AUDIT_LOG_FILE` | `hitl_audit.log` | JSONL audit log path. |

For notebook and test compatibility, `LLM_MODEL` and
`APPROVAL_TIMEOUT_HOURS` are also accepted as legacy aliases.

## Tool Contracts

### `search_product`

```python
search_product.invoke({"query": "MacBook"})
```

Returns a list of product dictionaries:

```python
[
    {
        "name": "MacBook Pro M3",
        "price": 15999.0,
        "vendor": "Apple Official Store",
        "currency": "CNY",
        "price_range": "15,999-25,999",
        "vendors": ["Apple Official Store", "JD.com", "Tmall"],
    }
]
```

### `purchase_item`

```python
purchase_item.invoke({
    "item": "MacBook Pro",
    "price": 15999.0,
    "vendor": "Apple Official Store",
    "thread_id": "demo-thread",
})
```

The tool calls `interrupt(...)` before completing. Resume with one of:

| Decision | Effect |
| --- | --- |
| `{"type": "accept"}` | Complete the purchase as requested. |
| `"approved"` | Complete the purchase as requested. |
| `{"type": "edit", "args": {"price": 14999.0}}` | Apply edits, then complete. |
| `{"type": "reject"}` | Reject the purchase. |

Audit logging records the initial purchase request and the final human decision.

## Web API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Basic web/API landing page. |
| `POST` | `/api/chat` | Submit a chat message. |
| `POST` | `/api/approve` | Accept, reject, or edit a pending approval. |
| `GET` | `/api/approvals` | List pending approvals. |
| `DELETE` | `/api/approvals/{approval_id}` | Cancel an approval. |
| `GET` | `/api/health` | Health check. |

The web chat endpoint invokes the real agent via `asyncio.to_thread` and
handles `GraphInterrupt` to capture the approval payload. Pending approvals are
stored in memory and cleaned up by a FastAPI lifespan background task. For
production, replace in-memory storage with Redis or a database and wire up
graph resume on the `/api/approve` path.

## Verify

Compile the modules:

```bash
python -m compileall .
```

Lint the source files:

```bash
ruff check agent.py cli.py config.py hitl.py logger.py tools.py web.py
```

Run tests:

```bash
python -m unittest discover -s . -p "test_*.py"
```

Or use the entry point:

```bash
python hitl.py --test
```

## Notes

- `hitl_agent.log` is runtime output and can be deleted safely.
- This is a teaching/demo system, not a production approval backend.
