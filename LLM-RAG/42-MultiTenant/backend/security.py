#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding_len = (-len(data)) % 4
    return base64.urlsafe_b64decode(data + ("=" * padding_len))


@dataclass
class VerifiedJwt:
    claims: Dict[str, Any]
    header: Dict[str, Any]


class JwtVerificationError(Exception):
    pass


class RSAKeyManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.private_key_path = base_dir / "jwt_private.pem"
        self.public_key_path = base_dir / "jwt_public.pem"
        self.kid_path = base_dir / "jwt_kid.txt"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_keys()

    def _ensure_keys(self) -> None:
        if self.private_key_path.exists() and self.public_key_path.exists() and self.kid_path.exists():
            return
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        self.private_key_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        self.public_key_path.write_bytes(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        self.kid_path.write_text(secrets.token_hex(8), encoding="utf-8")

    def load_private_key(self):
        if not hasattr(self, "_private_key"):
            self._private_key = serialization.load_pem_private_key(self.private_key_path.read_bytes(), password=None)
        return self._private_key

    def load_public_key(self):
        if not hasattr(self, "_public_key"):
            self._public_key = serialization.load_pem_public_key(self.public_key_path.read_bytes())
        return self._public_key

    def key_id(self) -> str:
        if not hasattr(self, "_kid"):
            self._kid = self.kid_path.read_text(encoding="utf-8").strip()
        return self._kid

    def rotate_keys(self) -> None:
        for attr in ("_private_key", "_public_key", "_kid"):
            self.__dict__.pop(attr, None)
        for path in (self.private_key_path, self.public_key_path, self.kid_path):
            if path.exists():
                path.unlink()
        self._ensure_keys()


class JWTService:
    def __init__(self, key_manager: RSAKeyManager):
        self.key_manager = key_manager

    def issue_token(
        self,
        *,
        tenant_id: str,
        user_id: str,
        role: str,
        scopes: list[str],
        issuer: str,
        audience: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        now = int(datetime.now(UTC).timestamp())
        payload = {
            "iss": issuer,
            "aud": audience,
            "sub": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "scopes": scopes,
            "iat": now,
            "nbf": now,
            "exp": now + expires_in_seconds,
        }
        header = {"alg": "RS256", "typ": "JWT", "kid": self.key_manager.key_id()}
        signing_input = ".".join(
            [
                _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
                _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
            ]
        )
        signature = self.key_manager.load_private_key().sign(
            signing_input.encode("ascii"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return f"{signing_input}.{_b64url_encode(signature)}"

    def verify_token(self, token: str, *, issuer: str, audience: str) -> VerifiedJwt:
        parts = token.split(".")
        if len(parts) != 3:
            raise JwtVerificationError("Malformed JWT")
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        try:
            header = json.loads(_b64url_decode(header_b64))
            claims = json.loads(_b64url_decode(payload_b64))
            signature = _b64url_decode(signature_b64)
        except Exception as exc:
            raise JwtVerificationError("Invalid JWT encoding") from exc

        if header.get("alg") != "RS256":
            raise JwtVerificationError("Unsupported JWT algorithm")

        try:
            self.key_manager.load_public_key().verify(
                signature,
                signing_input,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except Exception as exc:
            raise JwtVerificationError("JWT signature verification failed") from exc

        now = int(datetime.now(UTC).timestamp())
        if claims.get("iss") != issuer:
            raise JwtVerificationError("Invalid JWT issuer")
        if claims.get("aud") != audience:
            raise JwtVerificationError("Invalid JWT audience")
        if int(claims.get("nbf", 0)) > now or int(claims.get("exp", 0)) < now:
            raise JwtVerificationError("JWT is expired or not yet valid")
        return VerifiedJwt(claims=claims, header=header)

    def peek_claims(self, token: str) -> Dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            raise JwtVerificationError("Malformed JWT")
        try:
            return json.loads(_b64url_decode(parts[1]))
        except Exception as exc:
            raise JwtVerificationError("Invalid JWT encoding") from exc


class EncryptionKeyRotationRequired(Exception):
    pass


class AESGCMCipher:
    def __init__(self, key_file: Path):
        self.key_file = key_file
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_key()

    def _ensure_key(self) -> None:
        if self.key_file.exists():
            return
        self.key_file.write_bytes(secrets.token_bytes(32))

    @property
    def key(self) -> bytes:
        if not hasattr(self, "_key_bytes"):
            self._key_bytes = self.key_file.read_bytes()
        return self._key_bytes

    @property
    def _aesgcm(self) -> AESGCM:
        if not hasattr(self, "_aesgcm_instance"):
            self._aesgcm_instance = AESGCM(self.key)
        return self._aesgcm_instance

    def encrypt_json(self, payload: Dict[str, Any]) -> Dict[str, str]:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, json.dumps(payload, ensure_ascii=False).encode("utf-8"), None)
        return {
            "alg": "AES-256-GCM",
            "nonce": _b64url_encode(nonce),
            "ciphertext": _b64url_encode(ciphertext),
        }

    def decrypt_json(self, payload: Dict[str, str]) -> Dict[str, Any]:
        try:
            plaintext = self._aesgcm.decrypt(
                _b64url_decode(payload["nonce"]),
                _b64url_decode(payload["ciphertext"]),
                None,
            )
        except Exception as exc:
            raise EncryptionKeyRotationRequired(
                "Encrypted credentials or sensitive data can no longer be decrypted. "
                "If the key was rotated, all previously encrypted credentials must be re-entered."
            ) from exc
        return json.loads(plaintext.decode("utf-8"))
