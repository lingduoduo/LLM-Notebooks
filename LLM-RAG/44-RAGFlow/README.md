# RAGFlow Adapter

A minimal FastAPI adapter that exposes a standard retrieval API backed by RAGFlow, with a built-in mock mode for local development.

```
Client → POST /api/v1/retrieval → Adapter → RAGFlow retrieval API
                                      ↓
                               mock_knowledge.json  (mock mode)
```

## Quick Start

```bash
cd LLM-RAG/44-RAGFlow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

```bash
curl http://127.0.0.1:8080/healthz
```

## API

### `POST /api/v1/retrieval`

Header: `Authorization: Bearer <EXTERNAL_API_KEY>`

Request:

```json
{
  "knowledge_id": "ragflow-kb-001",
  "query": "How does RAGFlow handle chunking?",
  "retrieval_setting": {
    "top_k": 3,
    "score_threshold": 0.3
  }
}
```

| Field | Type | Constraints |
|---|---|---|
| `knowledge_id` | string | required |
| `query` | string | required |
| `top_k` | int | 1–20 |
| `score_threshold` | float | 0.0–1.0 |
| `metadata_condition` | object | optional |

Response:

```json
{
  "records": [
    {
      "content": "Retrieved chunk text",
      "score": 0.92,
      "title": "source-document.md",
      "metadata": {
        "knowledge_id": "ragflow-kb-001",
        "source": "ragflow"
      }
    }
  ]
}
```

### `GET /healthz`

```json
{
  "status": "ok",
  "mode": "mock",
  "ragflow_base_url": "http://127.0.0.1:9380",
  "search_path": "/api/v1/retrieval"
}
```

## Configuration

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `EXTERNAL_API_KEY` | `demo-key` | Bearer token clients must send |
| `RAGFLOW_MODE` | `mock` | `mock` or `ragflow` |
| `RAGFLOW_BASE_URL` | `http://127.0.0.1:9380` | RAGFlow base URL |
| `RAGFLOW_API_KEY` | _(empty)_ | Bearer token for the downstream RAGFlow API |
| `RAGFLOW_SEARCH_PATH` | `/api/v1/retrieval` | Retrieval path on the RAGFlow deployment |
| `REQUEST_TIMEOUT_SECONDS` | `15` | Downstream request timeout |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Modes

**`mock`** — serves results from `mock_knowledge.json`. No RAGFlow instance needed. Good for local development and integration testing.

**`ragflow`** — forwards requests to a live RAGFlow deployment:

```bash
EXTERNAL_API_KEY=your-secret-key
RAGFLOW_MODE=ragflow
RAGFLOW_BASE_URL=http://127.0.0.1:9380
RAGFLOW_API_KEY=your-ragflow-api-key
```

> If your RAGFlow response shape differs, update `normalize_ragflow_records()` in [app.py](app.py).

## Testing

```bash
# Smoke test (mock mode)
curl -X POST http://127.0.0.1:8080/api/v1/retrieval \
  -H "Authorization: Bearer demo-key" \
  -H "Content-Type: application/json" \
  -d '{"knowledge_id":"ragflow-kb-001","query":"RAGFlow setup","retrieval_setting":{"top_k":3,"score_threshold":0.3}}'

# Unit tests
python -m unittest tests.test_adapter
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```

## Troubleshooting

| Status | Cause |
|---|---|
| `401` | Bearer token does not match `EXTERNAL_API_KEY` |
| `502` | Adapter could not reach RAGFlow |
| `504` | RAGFlow request timed out (`REQUEST_TIMEOUT_SECONDS`) |
| Empty `records` | Wrong `knowledge_id`, `score_threshold` too high, or no matching content in the knowledge base |
