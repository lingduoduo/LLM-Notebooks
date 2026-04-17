# Multi-Tenant LangGraph Demo

A production-oriented multi-tenant AI customer support system demonstrating Dify-style tenant isolation with LangGraph, FastAPI, JWT authentication, and compliance controls.

Runtime-generated secrets, audit logs, and session files are written to `.runtime/` by default so they stay separate from source code.

## Architecture

Five layered tiers, each with a clear boundary:

```
Presentation  gradio_demo.py / langgraph_demo.py (CLI)
    │
API Gateway   backend/api.py          FastAPI, Bearer token auth, rate limiting
    │
Business      backend/service.py      Tenant resolution, session ownership, workflow routing
    │
Execution     backend/graph_engine.py LangGraph state machine, thread-safe tenant isolation
    │
Storage/Model backend/storage.py      JSON-backed session state (PostgreSQL/Redis-ready)
              backend/model_provider.py Model routing with DashScope + rule-based fallback
```

Supporting modules:

| File | Responsibility |
|---|---|
| `backend/auth.py` | JWT verification, RBAC scope enforcement, rate limiting |
| `backend/security.py` | RSA-2048 key management, JWT (RS256), AES-256-GCM encryption |
| `backend/audit.py` | Append-only JSONL audit log |
| `backend/compliance.py` | GDPR-style deletion lifecycle tracking |
| `backend/models.py` | Shared dataclasses: `TenantContext`, `TenantConfig`, `ChatState`, `ChatResult` |
| `backend/config.py` | Path constants and default tenant definitions |
| `backend/compat.py` | LangGraph/LangChain compatibility shim |

## Security Controls

- **JWT RS256** — tenant-bound identity; private key cached in memory, rotatable via `RSAKeyManager.rotate_keys()`
- **AES-256-GCM** — encrypts `user_memory` at rest; key cached after first load
- **RBAC** — four roles (`viewer`, `analyst`, `editor`, `admin`) mapped to scopes (`chat:read/write`, `audit:read`, `compliance:write`)
- **Rate limiting** — per-tenant per-user sliding window (in-memory, hourly stale-key cleanup)
- **Audit logging** — JSONL append for every API success, denial, and compliance event
- **Compliance endpoints** — data export requires `audit:read` scope; deletion requires `compliance:write` scope

> Private key rotation invalidates all AES-encrypted session data — users must re-enter sensitive fields after rotation.

## Request Flow

```
Client
  → POST /api/v1/chat  (Bearer <JWT>)
  → ApiGatewayAuth: verify JWT → resolve tenant → check RBAC scope
  → InMemoryRateLimiter: sliding-window check
  → MultiTenantPlatformService: resolve session ID
  → DifyWorkflowService.run(context, tenant_config, message)
  → LangGraphExecutor: load_state → run_model → persist_state
  → ModelRouter: DashScope API or rule-based fallback
  ← ChatResult
```

## Quickstart

```bash
pip install fastapi uvicorn langgraph langchain-core pydantic cryptography
```

**CLI demo:**
```bash
python langgraph_demo.py
```

**Gradio UI:**
```bash
python gradio_demo.py
```

**FastAPI server:**
```bash
uvicorn backend.api:app --reload
```

**Tests:**
```bash
python -m unittest tests.test_multitenant
```

## API Usage

**Get a demo token:**
```bash
export MULTITENANT_ENABLE_DEMO_TOKEN_ISSUANCE=true
TOKEN=$(curl -s http://127.0.0.1:8000/api/v1/security/token/company-a/alice \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
```

**Chat:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "message": "My hometown is in Beijing."}'
```

**Export user data** (requires `audit:read` scope):
```bash
curl http://127.0.0.1:8000/api/v1/compliance/export/company-a/alice \
  -H "Authorization: Bearer ${TOKEN}"
```

**Delete user data** (requires `compliance:write` scope — admin role only):
```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/compliance/delete/company-a \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "reason": "gdpr_erasure"}'
```

## Default Tenants

Defined in `tenant_configs.json`:

| Tenant | Users | Roles | Rate limit |
|---|---|---|---|
| `company-a` | alice, bob | alice=admin, bob=viewer | 30 req/min |
| `company-b` | charlie, diana | charlie=editor, diana=viewer | 20 req/min |
| `enterprise-x` | manager1, manager2 | both=admin/editor | 60 req/min |

## Production Checklist

- [ ] Replace `backend/storage.py` JSON files with PostgreSQL + Redis hot-session cache
- [ ] Replace `DifyWorkflowService` internals with a live Dify app or workflow client
- [ ] Move RSA private key and AES key into HSM/KMS (AWS KMS, HashiCorp Vault, etc.)
- [ ] Add database-backed rate limiting (Redis) for multi-instance deployments
- [ ] Integrate corporate SSO/IdP at the JWT issuance layer
- [ ] Wire `tenant_configs.json` tokens to environment variables — never commit secrets

## Runtime Files

- `.runtime/security/`: generated RSA keypair, key id, and AES key
- `.runtime/audit/`: append-only audit logs and compliance lifecycle records
- `.runtime/sessions.json`: persisted session state
- `.runtime/session_map.json`: tenant-user session mapping

The legacy `security/` directory may still exist locally from older runs, but it is no longer the default path.

> This demo implements the control points for JWT+RBAC, encryption, audit logging, and compliance lifecycle, but is not independently audited or certified for GDPR, SOX, or HIPAA. Treat it as an engineering baseline.
