#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from .security import JWTService, JwtVerificationError
from .models import TenantConfig
from .storage import TenantConfigRepository


@dataclass
class AccessPrincipal:
    tenant_id: str
    user_id: str
    scopes: list[str]
    role: str


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


ROLE_SCOPES = {
    "viewer": ["chat:read"],
    "analyst": ["chat:read", "chat:write", "audit:read"],
    "editor": ["chat:read", "chat:write"],
    "admin": ["chat:read", "chat:write", "audit:read", "compliance:write"],
}


class ApiGatewayAuth:
    def __init__(self, tenant_repo: TenantConfigRepository, jwt_service: JWTService):
        self.tenant_repo = tenant_repo
        self.jwt_service = jwt_service

    def authenticate(self, token: str, user_id: str, required_scopes: list[str] | None = None) -> AccessPrincipal:
        try:
            unsafe_claims = self.jwt_service.peek_claims(token)
        except JwtVerificationError as exc:
            raise AuthenticationError(str(exc)) from exc
        config = self.tenant_repo.get(unsafe_claims.get("tenant_id", ""))
        if not config:
            raise AuthenticationError("Unknown tenant in JWT")
        try:
            verified = self.jwt_service.verify_token(
                token,
                issuer=config.issuer,
                audience=config.audience,
            )
        except JwtVerificationError as exc:
            raise AuthenticationError(str(exc)) from exc

        claims = verified.claims
        if claims.get("sub") != user_id:
            raise AuthenticationError("JWT subject does not match requested user")
        if claims.get("tenant_id") != config.tenant_id:
            raise AuthenticationError("JWT tenant does not match configured tenant")
        if user_id not in config.allowed_users:
            raise AuthorizationError(f"User '{user_id}' is not allowed for tenant '{config.tenant_id}'")
        role = claims.get("role") or config.user_roles.get(user_id, "viewer")
        token_scopes = claims.get("scopes") or ROLE_SCOPES.get(role, ["chat:read"])
        if required_scopes:
            missing = [scope for scope in required_scopes if scope not in token_scopes]
            if missing:
                raise AuthorizationError(f"Missing required scopes: {', '.join(missing)}")
        return AccessPrincipal(
            tenant_id=config.tenant_id,
            user_id=user_id,
            scopes=token_scopes,
            role=role,
        )


class InMemoryRateLimiter:
    _CLEANUP_INTERVAL = 3600  # purge idle keys every hour

    def __init__(self):
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < self._CLEANUP_INTERVAL:
            return
        cutoff = now - 60
        stale: List[str] = [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]
        for k in stale:
            del self._hits[k]
        self._last_cleanup = now

    def check(self, config: TenantConfig, user_id: str) -> None:
        key = f"{config.tenant_id}:{user_id}"
        now = time.time()
        cutoff = now - 60
        with self._lock:
            self._maybe_cleanup(now)
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= config.rate_limit_per_minute:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for tenant '{config.tenant_id}', user '{user_id}'"
                )
            hits.append(now)
