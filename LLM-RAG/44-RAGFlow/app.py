#!/usr/bin/env python3

from __future__ import annotations

import hmac
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
MOCK_KNOWLEDGE_FILE = BASE_DIR / "mock_knowledge.json"


def load_dotenv_file() -> None:
    dotenv_path = BASE_DIR / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_dotenv_file()


@dataclass
class Settings:
    api_key: str = os.getenv("EXTERNAL_API_KEY", "demo-key")
    ragflow_mode: str = os.getenv("RAGFLOW_MODE", "mock")
    ragflow_base_url: str = os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380")
    ragflow_api_key: str = os.getenv("RAGFLOW_API_KEY", "")
    ragflow_search_path: str = os.getenv("RAGFLOW_SEARCH_PATH", "/api/v1/retrieval")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


SETTINGS = Settings()
logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if SETTINGS.api_key == "demo-key":
    logger.warning(
        "EXTERNAL_API_KEY is using the insecure demo default — set it in .env before deploying"
    )


class RetrievalSetting(BaseModel):
    top_k: int = Field(..., ge=1, le=20)
    score_threshold: float = Field(..., ge=0.0, le=1.0)


class MetadataCondition(BaseModel):
    logical_operator: Optional[str] = "and"
    conditions: List[Dict[str, Any]] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    knowledge_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    retrieval_setting: RetrievalSetting
    metadata_condition: Optional[MetadataCondition] = None


class Record(BaseModel):
    content: str
    score: float
    title: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    records: List[Record]


class MockKnowledgeStore:
    def __init__(self, file_path: Path):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"mock knowledge file must be a JSON object, got {type(payload).__name__}")
        self._records: Dict[str, List[Dict[str, Any]]] = payload

    def search(self, knowledge_id: str, query: str, top_k: int, score_threshold: float) -> List[Dict[str, Any]]:
        entries = self._records.get(knowledge_id, [])
        query_terms = {term.lower() for term in query.split() if term.strip()}
        scored: List[Dict[str, Any]] = []
        for item in entries:
            text = f"{item.get('title', '')} {item.get('content', '')}".lower()
            overlap = sum(1 for term in query_terms if term in text)
            score = item.get("score", 0.0)
            adjusted_score = max(score, min(1.0, overlap / max(1, len(query_terms))))
            if adjusted_score >= score_threshold:
                scored.append(
                    {
                        "content": item["content"],
                        "score": adjusted_score,
                        "title": item.get("title", ""),
                        "metadata": {
                            **item.get("metadata", {}),
                            "knowledge_id": knowledge_id,
                            "source": "mock",
                        },
                    }
                )
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]


def normalize_ragflow_records(payload: Dict[str, Any], knowledge_id: str) -> List[Dict[str, Any]]:
    candidates = payload.get("records") or payload.get("data") or payload.get("docs") or []
    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        content = (
            item.get("content")
            or item.get("text")
            or item.get("chunk")
            or item.get("page_content")
            or ""
        )
        if not content:
            continue
        normalized.append(
            {
                "content": content,
                "score": float(item.get("score", item.get("similarity", 0.0)) or 0.0),
                "title": item.get("title", item.get("document_name", "")) or "",
                "metadata": {
                    **item.get("metadata", {}),
                    "knowledge_id": knowledge_id,
                    "source": "ragflow",
                },
            }
        )
    return normalized


class RagFlowClient:
    def __init__(self, settings: Settings):
        self._url = f"{settings.ragflow_base_url.rstrip('/')}{settings.ragflow_search_path}"
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}
        if settings.ragflow_api_key:
            self._headers["Authorization"] = f"Bearer {settings.ragflow_api_key}"
        self._client = httpx.AsyncClient(timeout=settings.request_timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        *,
        knowledge_id: str,
        query: str,
        top_k: int,
        score_threshold: float,
        metadata_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "knowledge_id": knowledge_id,
            "query": query,
            "retrieval_setting": {"top_k": top_k, "score_threshold": score_threshold},
        }
        if metadata_condition:
            body["metadata_condition"] = metadata_condition

        logger.info("Forwarding retrieval to RAGFlow: knowledge_id=%s top_k=%s", knowledge_id, top_k)
        try:
            response = await self._client.post(self._url, headers=self._headers, json=body)
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail={"error_code": 3002, "error_msg": "RAGFlow request timed out"},
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error_code": 3003, "error_msg": f"RAGFlow request failed: {exc.__class__.__name__}"},
            ) from exc

        if not response.is_success:
            logger.error(
                "RAGFlow request failed: status=%d url=%s body=%.500s",
                response.status_code,
                self._url,
                response.text,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": 3001,
                    "error_msg": f"RAGFlow request failed with status {response.status_code}",
                },
            )

        return normalize_ragflow_records(response.json(), knowledge_id)


mock_store = MockKnowledgeStore(MOCK_KNOWLEDGE_FILE)
ragflow_client = RagFlowClient(SETTINGS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await ragflow_client.close()


app = FastAPI(title="RAGFlow Adapter", version="1.0.0", lifespan=lifespan)


def validate_api_key(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error_code": 1001, "error_msg": "Invalid Authorization header format"},
        )
    token = authorization[len("Bearer "):]
    if not hmac.compare_digest(token, SETTINGS.api_key):
        raise HTTPException(
            status_code=401,
            detail={"error_code": 1002, "error_msg": "Authorization failed. Please check your API key."},
        )


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": SETTINGS.ragflow_mode,
        "ragflow_base_url": SETTINGS.ragflow_base_url,
        "search_path": SETTINGS.ragflow_search_path,
    }


@app.post("/api/v1/retrieval", response_model=RetrievalResponse)
async def retrieval(
    payload: RetrievalRequest,
    authorization: Optional[str] = Header(default=None),
) -> RetrievalResponse:
    validate_api_key(authorization)

    if SETTINGS.ragflow_mode == "mock":
        records = mock_store.search(
            knowledge_id=payload.knowledge_id,
            query=payload.query,
            top_k=payload.retrieval_setting.top_k,
            score_threshold=payload.retrieval_setting.score_threshold,
        )
    else:
        records = await ragflow_client.search(
            knowledge_id=payload.knowledge_id,
            query=payload.query,
            top_k=payload.retrieval_setting.top_k,
            score_threshold=payload.retrieval_setting.score_threshold,
            metadata_condition=payload.metadata_condition.model_dump() if payload.metadata_condition else None,
        )

    logger.info(
        "retrieval: knowledge_id=%s query=%r records=%d mode=%s",
        payload.knowledge_id,
        payload.query,
        len(records),
        SETTINGS.ragflow_mode,
    )
    return RetrievalResponse(records=[Record(**record) for record in records])
