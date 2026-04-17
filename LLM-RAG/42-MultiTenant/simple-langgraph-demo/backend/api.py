#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .auth import AuthenticationError, AuthorizationError, RateLimitExceeded
from .config import ALLOW_DEMO_TOKEN_ISSUANCE
from .service import MultiTenantPlatformService, global_platform_service


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    answer: str
    workflow: str
    model: str
    metadata: dict


class ComplianceRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    reason: str = Field(default="user_request", min_length=1)


def parse_bearer_token(authorization: str = Header(...)) -> str:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    return authorization[len(prefix) :]


def get_platform_service(request: Request) -> MultiTenantPlatformService:
    return request.app.state.platform_service


def create_app(platform_service: MultiTenantPlatformService | None = None) -> FastAPI:
    app = FastAPI(title="Multi-Tenant AI Gateway", version="1.0.0")
    app.state.platform_service = platform_service or global_platform_service
    app.state.allow_demo_token_issuance = ALLOW_DEMO_TOKEN_ISSUANCE

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "security": ["JWT-RS256", "AES-256-GCM", "RBAC", "audit-logging"],
            "demo_token_endpoint_enabled": bool(app.state.allow_demo_token_issuance),
        }

    @app.get("/api/v1/security/token/{tenant_id}/{user_id}")
    def issue_demo_token(
        tenant_id: str,
        user_id: str,
        platform_service: MultiTenantPlatformService = Depends(get_platform_service),
    ) -> dict:
        if not app.state.allow_demo_token_issuance:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Demo token issuance endpoint is disabled. "
                    "Set MULTITENANT_ENABLE_DEMO_TOKEN_ISSUANCE=true to enable it in development."
                ),
            )
        token = platform_service.issue_demo_token(tenant_id, user_id)
        return {"token": token}

    @app.post("/api/v1/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        token: str = Depends(parse_bearer_token),
        platform_service: MultiTenantPlatformService = Depends(get_platform_service),
    ) -> ChatResponse:
        try:
            result = platform_service.handle_authenticated_message(
                token=token,
                user_id=request.user_id,
                message=request.message,
                session_id=request.session_id,
            )
        except AuthenticationError as exc:
            platform_service.audit_logger.log(
                "api.chat.denied",
                {"user_id": request.user_id, "reason": str(exc), "status_code": 401},
            )
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except AuthorizationError as exc:
            platform_service.audit_logger.log(
                "api.chat.denied",
                {"user_id": request.user_id, "reason": str(exc), "status_code": 403},
            )
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RateLimitExceeded as exc:
            platform_service.audit_logger.log(
                "api.chat.denied",
                {"user_id": request.user_id, "reason": str(exc), "status_code": 429},
            )
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            platform_service.audit_logger.log(
                "api.chat.denied",
                {"user_id": request.user_id, "reason": str(exc), "status_code": 400},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ChatResponse(**result.__dict__)

    @app.get("/api/v1/compliance/export/{tenant_id}/{user_id}")
    def export_user_data(
        tenant_id: str,
        user_id: str,
        token: str = Depends(parse_bearer_token),
        platform_service: MultiTenantPlatformService = Depends(get_platform_service),
    ) -> dict:
        try:
            principal = platform_service.auth.authenticate(token, user_id, required_scopes=["audit:read"])
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if principal.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Token tenant mismatch")
        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "data": platform_service.storage.export_user_data(tenant_id, user_id),
        }

    @app.delete("/api/v1/compliance/delete/{tenant_id}")
    def delete_user_data(
        tenant_id: str,
        request: ComplianceRequest,
        token: str = Depends(parse_bearer_token),
        platform_service: MultiTenantPlatformService = Depends(get_platform_service),
    ) -> dict:
        try:
            principal = platform_service.auth.authenticate(token, request.user_id, required_scopes=["compliance:write"])
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except AuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if principal.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Token tenant mismatch")
        platform_service.storage.delete_user_data(tenant_id, request.user_id, request.reason)
        return {"status": "deleted", "tenant_id": tenant_id, "user_id": request.user_id}

    return app


app = create_app()
