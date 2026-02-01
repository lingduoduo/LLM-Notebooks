#!/usr/bin/env python3
"""
Multi-Tenant Customer Support Demo Built with LangGraph
Focus: session isolation and context/state tracking

Core features:
1) Multi-tenant session isolation
2) Context and state tracking
"""

# ============================================================================
# Imports
# ============================================================================
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from typing import TypedDict
from typing_extensions import Annotated
import json
import os

# ============================================================================
# Data models
# ============================================================================

@dataclass
class TenantContext:
    """Tenant context - tracks session state for each tenant."""
    tenant_id: str
    user_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thread_id: str = field(default_factory=lambda: str(threading.current_thread().ident))
    timestamp: datetime = field(default_factory=datetime.now)

    def display_info(self) -> str:
        return f"[tenant:{self.tenant_id}|user:{self.user_id}|session:{self.session_id}]"


class ChatState(TypedDict):
    """Chat state - LangGraph state container."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[TenantContext]
    metadata: Dict[str, Any]
    # Persistent memory fields
    user_memory: Dict[str, Any]          # user info memory
    conversation_summary: str            # conversation summary
    last_topics: List[str]               # most recent topics


# ============================================================================
# Multi-tenant storage system
# ============================================================================

class MultiTenantStorage:
    """Multi-tenant data storage with full isolation and persistence."""
    def __init__(self, storage_file: str = "sessions.json"):
        # {tenant_id: {user_id: {session_id: ChatState}}}
        self._storage: Dict[str, Dict[str, Dict[str, ChatState]]] = {}
        self._lock = threading.RLock()
        self.storage_file = storage_file
        self._load_from_file()

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata, handling non-JSON-serializable values like datetime."""
        if not metadata:
            return {}

        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif hasattr(value, "__dict__"):
                # Skip complex objects
                continue
            else:
                try:
                    json.dumps(value)  # check serializability
                    serialized[key] = value
                except (TypeError, ValueError):
                    continue
        return serialized

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize metadata and restore datetime objects (best-effort)."""
        if not metadata:
            return {}

        deserialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str) and key.endswith("_time"):
                try:
                    deserialized[key] = datetime.fromisoformat(value)
                except ValueError:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

    def _load_from_file(self):
        """Load session data from disk."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for tenant_id, tenant_data in data.items():
                    self._storage.setdefault(tenant_id, {})
                    for user_id, user_sessions in tenant_data.items():
                        self._storage[tenant_id].setdefault(user_id, {})
                        for session_id, session_data in user_sessions.items():
                            # rebuild messages
                            messages: List[BaseMessage] = []
                            for msg_data in session_data.get("messages", []):
                                if msg_data["type"] == "human":
                                    messages.append(HumanMessage(content=msg_data["content"]))
                                elif msg_data["type"] == "ai":
                                    messages.append(AIMessage(content=msg_data["content"]))

                            self._storage[tenant_id][user_id][session_id] = {
                                "messages": messages,
                                "context": None,  # rebuilt at runtime
                                "metadata": self._deserialize_metadata(session_data.get("metadata", {})),
                                "user_memory": session_data.get("user_memory", {}),
                                "conversation_summary": session_data.get("conversation_summary", ""),
                                "last_topics": session_data.get("last_topics", []),
                            }

                print(f"[storage] Loaded sessions from {self.storage_file}")
            except Exception as e:
                print(f"[storage] Failed to load sessions: {e}")

    def _save_to_file(self):
        """Persist session data to disk."""
        try:
            serializable_data: Dict[str, Any] = {}

            for tenant_id, tenant_data in self._storage.items():
                serializable_data[tenant_id] = {}
                for user_id, user_sessions in tenant_data.items():
                    serializable_data[tenant_id][user_id] = {}
                    for session_id, session_data in user_sessions.items():
                        # serialize messages
                        messages_payload = []
                        for msg in session_data.get("messages", []):
                            if isinstance(msg, HumanMessage):
                                messages_payload.append({"type": "human", "content": msg.content})
                            elif isinstance(msg, AIMessage):
                                messages_payload.append({"type": "ai", "content": msg.content})

                        serializable_data[tenant_id][user_id][session_id] = {
                            "messages": messages_payload,
                            "metadata": self._serialize_metadata(session_data.get("metadata", {})),
                            "user_memory": session_data.get("user_memory", {}),
                            "conversation_summary": session_data.get("conversation_summary", ""),
                            "last_topics": session_data.get("last_topics", []),
                            "message_count": len(messages_payload),
                        }

            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)

            print(f"[storage] Saved sessions to {self.storage_file}")
        except Exception as e:
            print(f"[storage] Failed to save sessions: {e}")

    def get_or_create_state(self, context: TenantContext) -> ChatState:
        """Get or create a tenant session state."""
        with self._lock:
            tenant_id, user_id, session_id = context.tenant_id, context.user_id, context.session_id

            self._storage.setdefault(tenant_id, {})
            self._storage[tenant_id].setdefault(user_id, {})

            if session_id not in self._storage[tenant_id][user_id]:
                state: ChatState = {
                    "messages": [],
                    "context": context,
                    "metadata": {},
                    "user_memory": {},
                    "conversation_summary": "",
                    "last_topics": [],
                }
                self._storage[tenant_id][user_id][session_id] = state
                self._save_to_file()
            else:
                # restore existing session context at runtime
                self._storage[tenant_id][user_id][session_id]["context"] = context

            return self._storage[tenant_id][user_id][session_id]

    def update_session(self, context: TenantContext, state: ChatState):
        """Update session state and persist."""
        with self._lock:
            tenant_id, user_id, session_id = context.tenant_id, context.user_id, context.session_id
            if (
                tenant_id in self._storage
                and user_id in self._storage[tenant_id]
                and session_id in self._storage[tenant_id][user_id]
            ):
                self._storage[tenant_id][user_id][session_id] = state
                self._save_to_file()

    def show_isolation_status(self):
        """
        Print multi-tenant isolation status.

        Monitoring:
        - shows each tenant's isolated namespace
        - shows per-user sessions
        - counts message volume per session
        """
        with self._lock:
            print("\n[isolation] Multi-tenant data overview:")
            for tenant_id, tenant_data in self._storage.items():
                print(f"  Tenant {tenant_id}:")
                for user_id, user_sessions in tenant_data.items():
                    print(f"    User {user_id}: {len(user_sessions)} session(s)")
                    for session_id, state in user_sessions.items():
                        print(f"      Session {session_id}: {len(state['messages'])} message(s)")


class SessionManager:
    """
    Session ID manager - persists session IDs for each tenant/user.
    """
    def __init__(self, file_path: str = "session_map.json"):
        self._file_path = file_path
        self._lock = threading.RLock()
        self._sessions: Dict[str, str] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._sessions = data
        except Exception as e:
            print(f"[SessionManager] Failed to load session map: {e}")

    def _save(self):
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(self._sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[SessionManager] Failed to save session map: {e}")

    def _key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}_{user_id}"

    def get_session_id(self, tenant_id: str, user_id: str) -> str:
        """Get (or create) the persistent session ID for a tenant/user."""
        with self._lock:
            key = self._key(tenant_id, user_id)
            if key not in self._sessions:
                self._sessions[key] = str(uuid.uuid4())[:8]
                self._save()
            return self._sessions[key]

    def get_all_sessions(self) -> Dict[str, str]:
        """Return all tenant-user session mappings (key=tenant_user, value=session_id)."""
        with self._lock:
            return dict(self._sessions)

    def clear_session(self, tenant_id: str, user_id: str) -> str:
        """Generate and persist a new session ID for a tenant/user."""
        with self._lock:
            key = self._key(tenant_id, user_id)
            new_id = str(uuid.uuid4())[:8]
            self._sessions[key] = new_id
            self._save()
            return new_id


# Global instances (also used by a Gradio UI)
global_session_manager = SessionManager(file_path="session_map.json")
global_storage = MultiTenantStorage(storage_file="sessions.json")

# Thread-local context for tenant scoping
_context_local = threading.local()


def get_current_context() -> Optional[TenantContext]:
    """Get tenant context for the current thread."""
    return getattr(_context_local, "current_context", None)


@contextmanager
def tenant_context(tenant_id: str, user_id: str, session_id: Optional[str] = None):
    """
    Establish tenant context so downstream graph nodes run in the correct tenant scope.
    If session_id is not provided, it is loaded from the global session manager.
    """
    prev = getattr(_context_local, "current_context", None)
    sid = session_id or global_session_manager.get_session_id(tenant_id, user_id)
    ctx = TenantContext(tenant_id=tenant_id, user_id=user_id, session_id=sid)
    _context_local.current_context = ctx
    try:
        yield ctx
    finally:
        _context_local.current_context = prev


# ============================================================================
# Graph nodes
# ============================================================================

def process_user_input(state: ChatState) -> ChatState:
    """
    User input node - entry point of the LangGraph workflow.

    Flow:
    1) Validate the current tenant context
    2) Fetch tenant-isolated state storage
    3) Append the latest user message to the tenant-isolated history
    4) Ensure messages are processed in the correct tenant space
    """
    context = get_current_context()
    if not context:
        raise ValueError("Must run inside a tenant_context")

    tenant_state = global_storage.get_or_create_state(context)

    if state["messages"]:
        latest_message = state["messages"][-1]
        tenant_state["messages"].append(latest_message)
        print(f"[user_input] {context.display_info()} message: {latest_message.content}")

    return tenant_state


def generate_ai_response(state: ChatState) -> ChatState:
    """
    AI response node - generate an intelligent response using tenant context and persistent memory.

    Mechanism:
    1) Extract user memory from tenant-isolated state
    2) Build a prompt including historical context
    3) Call an AI model to generate a targeted response
    4) Update conversation memory and topic tracking
    5) Ensure generation happens in the correct tenant context
    """
    context = get_current_context()
    if not context:
        return state

    print(f"[ai_response] {context.display_info()} generating response...")

    if not state["messages"]:
        return state

    user_message = state["messages"][-1].content

    # Update user memory
    update_user_memory(state, user_message)

    # Build memory-aware system prompt
    system_prompt = build_memory_aware_prompt(state, context)

    # Build conversation history
    conversation_history = build_conversation_history(state)

    # Initialize DashScope once (if available)
    if not hasattr(generate_ai_response, "_dashscope_initialized"):
        import dashscope
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key
            generate_ai_response._dashscope_initialized = True
        else:
            print("Warning: DASHSCOPE_API_KEY not set; using fallback response")

    ai_response = ""

    if hasattr(generate_ai_response, "_dashscope_initialized"):
        from dashscope import Generation
        from http import HTTPStatus

        state["metadata"].update(
            {
                "api_call_attempted": True,
                "user_message": user_message,
                "conversation_length": len(state["messages"]),
            }
        )

        try:
            response = Generation.call(
                model="qwen-turbo",
                messages=[{"role": "system", "content": system_prompt}] + conversation_history,
                result_format="message",
            )

            if response.status_code == HTTPStatus.OK:
                ai_response = response.output.choices[0].message.content
            else:
                ai_response = generate_memory_fallback_response(state, context, user_message)
        except Exception as e:
            print(f"DashScope API exception: {e}")
            ai_response = generate_memory_fallback_response(state, context, user_message)
    else:
        ai_response = generate_memory_fallback_response(state, context, user_message)

    state["messages"].append(AIMessage(content=ai_response))
    update_conversation_memory(state, user_message, ai_response)

    print(f"[ai_response] {context.display_info()} reply: {ai_response}")
    return state


def update_context_tracking(state: ChatState) -> ChatState:
    """
    Context tracking node - maintain topic continuity and context metadata.

    Mechanism:
    1) Extract key topics from recent messages
    2) Maintain a list of recent topics (max 5)
    3) Support continuity for future AI responses
    4) Keep metadata up-to-date
    """
    context = get_current_context()
    if context:
        state["metadata"].update(
            {
                "last_update": datetime.now(),
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "message_count": len(state["messages"]),
            }
        )
        print(f"[context_tracking] {context.display_info()} metadata updated")
    return state


# ============================================================================
# Memory helpers
# ============================================================================

def update_user_memory(state: ChatState, user_message: str):
    """Update persistent user memory (name/age/hobby extraction)."""
    state.setdefault("user_memory", {})

    name = extract_name_from_message(user_message)
    if name:
        state["user_memory"]["name"] = name

    age = extract_age_from_message(user_message)
    if age:
        state["user_memory"]["age"] = age

    hobby = extract_hobby_from_message(user_message)
    if hobby:
        state["user_memory"].setdefault("hobbies", [])
        if hobby not in state["user_memory"]["hobbies"]:
            state["user_memory"]["hobbies"].append(hobby)


def build_memory_aware_prompt(state: ChatState, context: TenantContext) -> str:
    """
    Build a system prompt with memory for personalization.
    """
    user_memory = state.get("user_memory", {})
    conversation_summary = state.get("conversation_summary", "")

    prompt = (
        f"You are the dedicated AI support assistant for tenant '{context.tenant_id}', "
        f"serving user '{context.user_id}'.\n\n"
        f"User memory:\n"
    )

    if user_memory:
        if "name" in user_memory:
            prompt += f"- Name: {user_memory['name']}\n"
        if "age" in user_memory:
            prompt += f"- Age: {user_memory['age']}\n"
        if "hobbies" in user_memory:
            prompt += f"- Hobbies: {', '.join(user_memory['hobbies'])}\n"
    else:
        prompt += "- No known user info yet\n"

    if conversation_summary:
        prompt += f"\nConversation summary: {conversation_summary}\n"

    prompt += (
        "\nPlease respond in a personalized and context-aware way. "
        "Remember what the user told you and reference it when appropriate."
    )
    return prompt


def build_conversation_history(state: ChatState) -> List[Dict[str, str]]:
    """Build a list of recent chat turns for the model."""
    conversation_history: List[Dict[str, str]] = []
    messages = state.get("messages", [])

    recent_messages = messages[:-1][-10:] if len(messages) > 1 else []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation_history.append({"role": "assistant", "content": msg.content})

    if messages:
        current_msg = messages[-1]
        if isinstance(current_msg, HumanMessage):
            conversation_history.append({"role": "user", "content": current_msg.content})

    return conversation_history


def update_conversation_memory(state: ChatState, user_message: str, ai_response: str):
    """Update last_topics and conversation_summary."""
    state.setdefault("last_topics", [])

    topic = extract_topic_from_message(user_message)
    if topic and topic not in state["last_topics"]:
        state["last_topics"].append(topic)
        if len(state["last_topics"]) > 5:
            state["last_topics"] = state["last_topics"][-5:]

    message_count = len(state.get("messages", []))
    if message_count > 2:
        recent = ", ".join(state["last_topics"][-3:])
        state["conversation_summary"] = f"{message_count // 2} turns so far; topics include {recent}."


def extract_topic_from_message(message: str) -> str:
    """
    Extract topic using a simple memory mechanism in the tenant state.
    Note: This is a placeholder/demo implementation.
    """
    context = get_current_context()
    if not context:
        return "general"

    tenant_state = global_storage.get_or_create_state(context)
    tenant_state.setdefault("topic_memory", {})

    message_key = message[:20].lower()

    if message_key not in tenant_state["topic_memory"]:
        default_topic = "general"
        if tenant_state.get("last_topics"):
            default_topic = tenant_state["last_topics"][-1]
        tenant_state["topic_memory"][message_key] = default_topic

    topic = tenant_state["topic_memory"].get(message_key, "general")

    tenant_state.setdefault("last_topics", [])
    if topic not in tenant_state["last_topics"]:
        tenant_state["last_topics"].append(topic)
        if len(tenant_state["last_topics"]) > 5:
            tenant_state["last_topics"] = tenant_state["last_topics"][-5:]

    return topic


def generate_memory_fallback_response(state: ChatState, context: TenantContext, user_message: str) -> str:
    """Memory-aware fallback response when no model API is available."""
    user_memory = state.get("user_memory", {})
    state.setdefault("response_memory", {})

    current_topic = state.get("last_topics", [""])[-1] if state.get("last_topics") else ""

    assistant_tag = f"I am the dedicated AI support assistant for tenant '{context.tenant_id}'."

    if current_topic == "personal_info" and user_memory.get("name"):
        response = f"Hello, {user_memory['name']}! {assistant_tag} How can I help you today?"
    elif current_topic == "age_info" and user_memory.get("age"):
        response = f"Based on what you shared, you are {user_memory['age']} years old. How can I assist?"
    elif current_topic == "hobbies" and user_memory.get("hobbies"):
        hobbies = ", ".join(user_memory["hobbies"])
        response = f"You mentioned you enjoy {hobbies}. Want to talk more about that?"
    elif current_topic == "poetry":
        response = "I can write a short poem for you—what theme would you like?"
    elif current_topic == "consulting":
        response = f"{assistant_tag} What would you like to ask?"
    else:
        if user_memory.get("name"):
            response = f"Hello, {user_memory['name']}! {assistant_tag} What can I help you with?"
        else:
            response = f"Hello! {assistant_tag} What can I help you with?"

    state["response_memory"][current_topic] = response
    return response


# ============================================================================
# Extraction functions (demo placeholders)
# ============================================================================

def extract_name_from_message(message: str) -> str:
    """
    Extract name from user message (demo placeholder using tenant memory).
    Current implementation only records the message; returns empty string.
    """
    context = get_current_context()
    if not context:
        return ""

    tenant_state = global_storage.get_or_create_state(context)
    tenant_state.setdefault("name_memory", {})

    message_key = message[:30].lower()

    if any(k in message.lower() for k in ["name", "my name", "call me"]):
        tenant_state["name_memory"][message_key] = message
        tenant_state.setdefault("user_memory", {})
        tenant_state["user_memory"]["has_name_info"] = True

    return ""


def extract_age_from_message(message: str) -> str:
    """Extract age from user message (demo placeholder)."""
    context = get_current_context()
    if not context:
        return ""

    tenant_state = global_storage.get_or_create_state(context)
    tenant_state.setdefault("age_memory", {})

    message_key = message[:30].lower()

    if any(k in message.lower() for k in ["age", "years old"]):
        tenant_state["age_memory"][message_key] = message
        tenant_state.setdefault("user_memory", {})
        tenant_state["user_memory"]["has_age_info"] = True

    return ""


def extract_hobby_from_message(message: str) -> str:
    """Extract hobbies from user message (demo placeholder)."""
    context = get_current_context()
    if not context:
        return ""

    tenant_state = global_storage.get_or_create_state(context)
    tenant_state.setdefault("hobby_memory", {})

    message_key = message[:30].lower()

    if any(k in message.lower() for k in ["like", "hobby", "interested in"]):
        tenant_state["hobby_memory"][message_key] = message
        tenant_state.setdefault("user_memory", {})
        tenant_state["user_memory"].setdefault("hobbies_info", [])
        tenant_state["user_memory"]["hobbies_info"].append(message_key)

    return ""


# ============================================================================
# Misc helpers (still kept for completeness)
# ============================================================================

def extract_user_info_from_history(user_messages: List[str]) -> str:
    """Extract user info from historical messages (placeholder)."""
    info_parts = []
    for msg in user_messages:
        if "my name" in msg.lower():
            name = extract_name_from_message(msg)
            if name:
                info_parts.append(f"Your name is {name}")
        if "years old" in msg.lower():
            age = extract_age_from_message(msg)
            if age:
                info_parts.append(f"You are {age} years old")
        if "i like" in msg.lower():
            hobby = extract_hobby_from_message(msg)
            if hobby:
                info_parts.append(f"You like {hobby}")
    return ", ".join(info_parts) if info_parts else ""


def extract_recent_topics(recent_messages: List[str]) -> str:
    """Extract recently discussed topics (simple heuristics)."""
    topics = []
    for msg in recent_messages:
        if len(msg) <= 2:
            continue
        if any(k in msg.lower() for k in ["question", "help", "consult"]):
            topics.append("questions/help")
        elif any(k in msg.lower() for k in ["product", "service", "feature"]):
            topics.append("product/service")
        elif any(k in msg.lower() for k in ["tech", "support", "bug", "issue"]):
            topics.append("technical support")
        else:
            topics.append(msg[:10] + "..." if len(msg) > 10 else msg)
    return " | ".join(topics[:3]) if topics else "general"


def summarize_previous_context(user_messages: List[str], ai_messages: List[str]) -> str:
    """Summarize prior conversation context."""
    if not user_messages:
        return ""
    if len(user_messages) == 1:
        return f"You previously mentioned '{user_messages[0][:20]}...'."
    return f"We discussed {len(user_messages)} topics, including '{user_messages[0][:15]}...'."


def get_conversation_context(user_messages: List[str], ai_messages: List[str]) -> str:
    """Return a short conversation context summary."""
    total_exchanges = min(len(user_messages), len(ai_messages))
    return f"{total_exchanges} turns"


# ============================================================================
# LangGraph workflow
# ============================================================================

def create_multi_tenant_graph():
    """
    Build a multi-tenant-aware LangGraph workflow.

    Steps:
    1) Define the StateGraph and nodes
    2) Set transitions
    3) Define entry and exit points
    4) Compile with a checkpointer
    """
    workflow = StateGraph(ChatState)

    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("generate_ai_response", generate_ai_response)
    workflow.add_node("update_context_tracking", update_context_tracking)

    workflow.set_entry_point("process_user_input")
    workflow.add_edge("process_user_input", "generate_ai_response")
    workflow.add_edge("generate_ai_response", "update_context_tracking")
    workflow.add_edge("update_context_tracking", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================================================
# CLI entry points
# ============================================================================

def interactive_session(tenant_id: str, user_id: str, session_id: Optional[str] = None):
    """Interactive CLI session with auto session recovery."""
    graph = create_multi_tenant_graph()

    existing_sessions: List[str] = []
    if tenant_id in global_storage._storage and user_id in global_storage._storage[tenant_id]:
        existing_sessions = list(global_storage._storage[tenant_id][user_id].keys())

    if session_id is None:
        if existing_sessions:
            session_id = existing_sessions[-1]
            print("\n[auto_recover] Found an existing session; continuing...")
        else:
            session_id = str(uuid.uuid4())[:8]
            print("\n[new_session] Creating a new chat session...")

    print(f"[session] tenant:{tenant_id} | user:{user_id} | session_id:{session_id}")
    print("Type 'exit' to quit\n")

    with tenant_context(tenant_id, user_id, session_id):
        while True:
            user_input = input("User >>> ")
            if user_input.lower() == "exit":
                break

            current_context = get_current_context()
            tenant_state = global_storage.get_or_create_state(current_context)

            user_message = HumanMessage(content=user_input)
            tenant_state["messages"].append(user_message)

            initial_state = {
                "messages": tenant_state["messages"].copy(),
                "context": current_context,
                "metadata": tenant_state.get("metadata", {}),
                "user_memory": tenant_state.get("user_memory", {}),
                "conversation_summary": tenant_state.get("conversation_summary", ""),
                "last_topics": tenant_state.get("last_topics", []),
            }

            print(f"[debug] Messages sent to workflow: {len(initial_state['messages'])}")

            try:
                result = graph.invoke(
                    initial_state,
                    config={
                        "configurable": {
                            "thread_id": f"{tenant_id}_{user_id}_{session_id}",
                            "checkpoint_ns": f"tenant_{tenant_id}",
                        }
                    },
                )

                print(f"[debug] Messages returned by workflow: {len(result.get('messages', []))}")

                if result:
                    tenant_state.update(
                        {
                            "messages": result.get("messages", []),
                            "metadata": result.get("metadata", {}),
                            "user_memory": result.get("user_memory", {}),
                            "conversation_summary": result.get("conversation_summary", ""),
                            "last_topics": result.get("last_topics", []),
                            "context": current_context,
                        }
                    )
                    global_storage.update_session(current_context, tenant_state)

                    if result.get("messages"):
                        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
                        if ai_messages:
                            print(f"\nAI >>> {ai_messages[-1].content}")
                        else:
                            print(f"\n[debug] No AI message found; total messages: {len(result['messages'])}")
                            for i, msg in enumerate(result["messages"]):
                                print(f"[debug] msg{i}: {type(msg).__name__} - {msg.content[:50]}...")
                    else:
                        print("\nNo messages in result")
                else:
                    print("\nWorkflow returned no result")

            except Exception as e:
                print(f"\nWorkflow error: {e}")
                import traceback
                traceback.print_exc()


def list_user_sessions(tenant_id: str, user_id: str) -> List[str]:
    """List all sessions for a user."""
    sessions: List[str] = []
    if tenant_id in global_storage._storage and user_id in global_storage._storage[tenant_id]:
        sessions = list(global_storage._storage[tenant_id][user_id].keys())
    return sessions


def main():
    """Main program entry."""
    print("=== Multi-tenant Customer Support Demo ===")

    while True:
        print("\nChoose an action:")
        print("1. Start chat (auto-recover existing session)")
        print("2. View session list")
        print("3. Exit")

        choice = input("Select (1-3): ")

        if choice == "1":
            tenant_id = input("Enter tenant_id: ")
            user_id = input("Enter user_id: ")
            interactive_session(tenant_id, user_id)
        elif choice == "2":
            tenant_id = input("Enter tenant_id: ")
            user_id = input("Enter user_id: ")
            sessions = list_user_sessions(tenant_id, user_id)
            if sessions:
                print(f"Sessions for user {user_id}: {sessions}")
            else:
                print(f"No sessions for user {user_id}")
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Try again.")

    global_storage.show_isolation_status()


if __name__ == "__main__":
    main()


class MultiTenantCustomerService:
    """
    Multi-tenant customer service wrapper for a simple message-processing interface (e.g., Gradio).

    Usage:
    - Call process_message(message) inside a tenant_context(tenant_id, user_id, session_id)
    - Automatically loads/updates state from global storage and returns the latest AI reply as a string
    """
    def __init__(self):
        self.graph = create_multi_tenant_graph()

    def process_message(self, message: str) -> str:
        context = get_current_context()
        if not context:
            raise RuntimeError("No tenant context detected; call inside tenant_context")

        tenant_state = global_storage.get_or_create_state(context)

        tenant_state["messages"].append(HumanMessage(content=message))

        initial_state = {
            "messages": tenant_state["messages"].copy(),
            "context": context,
            "metadata": tenant_state.get("metadata", {}),
            "user_memory": tenant_state.get("user_memory", {}),
            "conversation_summary": tenant_state.get("conversation_summary", ""),
            "last_topics": tenant_state.get("last_topics", []),
        }

        result = self.graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": f"{context.tenant_id}_{context.user_id}_{context.session_id}",
                    "checkpoint_ns": f"tenant_{context.tenant_id}",
                }
            },
        )

        if result:
            tenant_state.update(
                {
                    "messages": result.get("messages", tenant_state["messages"]),
                    "metadata": result.get("metadata", tenant_state.get("metadata", {})),
                    "user_memory": result.get("user_memory", tenant_state.get("user_memory", {})),
                    "conversation_summary": result.get(
                        "conversation_summary", tenant_state.get("conversation_summary", "")
                    ),
                    "last_topics": result.get("last_topics", tenant_state.get("last_topics", [])),
                    "context": context,
                }
            )
            global_storage.update_session(context, tenant_state)

        ai_content = None
        if tenant_state.get("messages"):
            ai_messages = [m for m in tenant_state["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                ai_content = ai_messages[-1].content

        return ai_content or "(No AI reply generated)"
