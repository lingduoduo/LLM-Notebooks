#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from typing import Dict, List

from .models import ChatState, TenantConfig, TenantContext

_RE_HOMETOWN_1 = re.compile(r"hometown\s+is\s+in\s+([A-Za-z\s-]+)", re.IGNORECASE)
_RE_HOMETOWN_2 = re.compile(r"my hometown is ([A-Za-z\s-]+)", re.IGNORECASE)
_RE_NAME = re.compile(r"(?:my name is|call me)\s+([A-Za-z][A-Za-z\s-]{0,30})", re.IGNORECASE)
_RE_HOBBY = re.compile(r"(?:i like|my hobby is|i enjoy)\s+([A-Za-z\s-]+)", re.IGNORECASE)


def _extract_hometown(message: str) -> str:
    match = _RE_HOMETOWN_1.search(message)
    if match:
        return match.group(1).strip().rstrip(".")
    match = _RE_HOMETOWN_2.search(message)
    if match:
        return match.group(1).strip().rstrip(".")
    return ""


def _extract_name(message: str) -> str:
    match = _RE_NAME.search(message)
    return match.group(1).strip() if match else ""


def _extract_hobby(message: str) -> str:
    match = _RE_HOBBY.search(message)
    return match.group(1).strip().rstrip(".") if match else ""


def _update_memory(state: ChatState, user_message: str) -> None:
    memory = state.setdefault("user_memory", {})
    hometown = _extract_hometown(user_message)
    if hometown:
        memory["hometown"] = hometown
    name = _extract_name(user_message)
    if name:
        memory["name"] = name
    hobby = _extract_hobby(user_message)
    if hobby:
        memory.setdefault("hobbies", [])
        if hobby not in memory["hobbies"]:
            memory["hobbies"].append(hobby)


def _extract_topic(message: str) -> str:
    lowered = message.lower()
    if "hometown" in lowered or "city" in lowered:
        return "hometown"
    if "food" in lowered or "famous" in lowered:
        return "local_food"
    if "name" in lowered:
        return "identity"
    return "general"


class ModelRouter:
    def generate(self, state: ChatState, context: TenantContext, tenant_config: TenantConfig) -> str:
        user_message = state["messages"][-1].content if state.get("messages") else ""
        _update_memory(state, user_message)
        topic = _extract_topic(user_message)
        state.setdefault("last_topics", [])
        if topic not in state["last_topics"]:
            state["last_topics"].append(topic)
            state["last_topics"] = state["last_topics"][-5:]

        openai_key = os.getenv("OPENAI_API_KEY")
        if tenant_config.model != "fallback" and openai_key:
            response = self._try_openai(state, context, tenant_config)
            if response:
                return response
        return self._fallback_response(state, context, tenant_config, user_message)

    def _try_openai(self, state: ChatState, context: TenantContext, tenant_config: TenantConfig) -> str:
        """Call the official OpenAI Chat Completions API; return "" on any failure."""
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return ""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=tenant_config.model,
                messages=self._build_messages(state, context, tenant_config),
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def _build_messages(
        self,
        state: ChatState,
        context: TenantContext,
        tenant_config: TenantConfig,
    ) -> List[Dict[str, str]]:
        memory = state.get("user_memory", {})
        system_prompt = (
            "You are a multi-tenant support assistant.\n"
            f"Tenant: {context.tenant_id}\n"
            f"Workflow: {tenant_config.workflow_name}\n"
            f"Dify app: {tenant_config.dify_app_id or 'local'}\n"
            f"Known memory: {memory}\n"
            "Keep tenant data isolated and answer based only on this tenant session."
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in state.get("messages", [])[-8:]:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})
        return messages

    def _fallback_response(
        self,
        state: ChatState,
        context: TenantContext,
        tenant_config: TenantConfig,
        user_message: str,
    ) -> str:
        memory = state.get("user_memory", {})
        hometown = memory.get("hometown")
        name = memory.get("name") or context.user_id
        lowered = user_message.lower()
        if "where is my hometown" in lowered or "which city is my hometown" in lowered:
            if hometown:
                return (
                    f"{name}, based on the current tenant-isolated session, your hometown is {hometown}. "
                    f"This answer came from workflow '{tenant_config.workflow_name}' for tenant '{context.tenant_id}'."
                )
            return (
                f"{name}, I do not have your hometown in this session yet. "
                "Tell me where you are from and I will remember it inside your tenant workspace only."
            )
        if "food" in lowered and hometown:
            return (
                f"Because your hometown is {hometown}, I would route this through the business workflow and "
                f"generate a tenant-specific answer about local food for {hometown}."
            )
        return (
            f"Hello {name}. Tenant '{context.tenant_id}' is using workflow '{tenant_config.workflow_name}' "
            f"and model '{tenant_config.model}'. I received: '{user_message}'."
        )
