#!/usr/bin/env python3
"""Offline tests for ModelRouter provider selection (no network, OpenAI client is faked)."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.compat import HumanMessage
from backend.model_provider import ModelRouter
from backend.models import TenantConfig, TenantContext


def _fake_openai_module(reply: str = "openai reply", raise_exc: bool = False):
    calls: list = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if raise_exc:
                raise RuntimeError("simulated API failure")
            message = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        def __init__(self, api_key=None, **_):
            calls.append({"api_key": api_key})
            self.chat = types.SimpleNamespace(completions=_Completions())

    module = types.ModuleType("openai")
    module.OpenAI = OpenAI
    return module, calls


class ModelRouterProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ModelRouter()
        self.context = TenantContext(tenant_id="company-a", user_id="alice", session_id="s1")

    def _state(self):
        return {"messages": [HumanMessage(content="My hometown is in Boston.")]}

    def _config(self, model: str) -> TenantConfig:
        return TenantConfig(tenant_id="company-a", display_name="A", allowed_users=["alice"], model=model)

    def test_fallback_model_never_calls_openai(self) -> None:
        fake, calls = _fake_openai_module()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), mock.patch.dict(sys.modules, {"openai": fake}):
            reply = self.router.generate(self._state(), self.context, self._config("fallback"))
        self.assertEqual(calls, [])
        self.assertIn("company-a", reply)

    def test_no_api_key_uses_fallback(self) -> None:
        fake, calls = _fake_openai_module()
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"openai": fake}):
            reply = self.router.generate(self._state(), self.context, self._config("gpt-4.1-mini"))
        self.assertEqual(calls, [])
        self.assertIn("gpt-4.1-mini", reply)

    def test_openai_model_routes_to_openai(self) -> None:
        fake, calls = _fake_openai_module(reply="hello from openai")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), mock.patch.dict(sys.modules, {"openai": fake}):
            reply = self.router.generate(self._state(), self.context, self._config("gpt-4.1-mini"))
        self.assertEqual(reply, "hello from openai")
        self.assertEqual(calls[0], {"api_key": "sk-test"})
        self.assertEqual(calls[1]["model"], "gpt-4.1-mini")
        self.assertEqual(calls[1]["messages"][0]["role"], "system")
        self.assertEqual(calls[1]["messages"][-1], {"role": "user", "content": "My hometown is in Boston."})

    def test_openai_error_falls_back(self) -> None:
        fake, _ = _fake_openai_module(raise_exc=True)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), mock.patch.dict(sys.modules, {"openai": fake}):
            reply = self.router.generate(self._state(), self.context, self._config("gpt-4.1-mini"))
        self.assertIn("Tenant 'company-a'", reply)


if __name__ == "__main__":
    unittest.main()
