#!/usr/bin/env python3

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Optional

from .compat import AIMessage, END, HumanMessage, MemorySaver, StateGraph
from .model_provider import ModelRouter
from .models import ChatState, TenantConfig, TenantContext
from .storage import MultiTenantStorage


class LangGraphExecutor:
    def __init__(self, storage: MultiTenantStorage, model_router: ModelRouter):
        self.storage = storage
        self.model_router = model_router
        self.graph = self._create_graph()
        self._local = threading.local()

    def _create_graph(self):
        workflow = StateGraph(ChatState)
        workflow.add_node("load_state", self._load_state)
        workflow.add_node("run_model", self._run_model)
        workflow.add_node("persist_state", self._persist_state)
        workflow.set_entry_point("load_state")
        workflow.add_edge("load_state", "run_model")
        workflow.add_edge("run_model", "persist_state")
        workflow.add_edge("persist_state", END)
        return workflow.compile(checkpointer=MemorySaver())

    def invoke(self, context: TenantContext, tenant_config: TenantConfig, message: str) -> ChatState:
        self._local.context = context
        self._local.config = tenant_config
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={
                "configurable": {
                    "thread_id": f"{context.tenant_id}:{context.user_id}:{context.session_id}",
                    "checkpoint_ns": f"tenant:{context.tenant_id}",
                }
            },
        )
        return result

    def _load_state(self, state: ChatState) -> ChatState:
        context = self._require_context()
        tenant_state = self.storage.get_or_create_state(context)
        if state.get("messages"):
            tenant_state.setdefault("messages", [])
            tenant_state["messages"].extend(state["messages"])
        tenant_state["context"] = context
        return tenant_state

    def _run_model(self, state: ChatState) -> ChatState:
        context = self._require_context()
        tenant_config = self._require_config()
        answer = self.model_router.generate(state, context, tenant_config)
        state.setdefault("messages", [])
        state["messages"].append(AIMessage(content=answer))
        state["conversation_summary"] = (
            f"{len(state['messages']) // 2} turns; latest topics: {', '.join(state.get('last_topics', [])[-3:])}"
        )
        state.setdefault("metadata", {})
        state["metadata"].update(
            {
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "workflow_name": tenant_config.workflow_name,
                "model_name": tenant_config.model,
                "last_update": datetime.now(UTC).isoformat(),
                "message_count": len(state["messages"]),
            }
        )
        return state

    def _persist_state(self, state: ChatState) -> ChatState:
        context = self._require_context()
        self.storage.update_session(context, state)
        return state

    def _require_context(self) -> TenantContext:
        ctx = getattr(self._local, "context", None)
        if not ctx:
            raise RuntimeError("Tenant context is not set")
        return ctx

    def _require_config(self) -> TenantConfig:
        cfg = getattr(self._local, "config", None)
        if not cfg:
            raise RuntimeError("Tenant config is not set")
        return cfg
