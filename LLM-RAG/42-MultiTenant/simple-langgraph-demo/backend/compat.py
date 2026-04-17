#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages

    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    HAS_LANGGRAPH = False
    END = "__end__"

    @dataclass
    class BaseMessage:
        content: str
        type: str = "base"

    class HumanMessage(BaseMessage):
        def __init__(self, content: str):
            super().__init__(content=content, type="human")

    class AIMessage(BaseMessage):
        def __init__(self, content: str):
            super().__init__(content=content, type="ai")

    def add_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
        return list(existing) + list(new)

    class MemorySaver:
        pass

    class _CompiledGraph:
        def __init__(self, nodes: Dict[str, Callable], edges: Dict[str, str], entry_point: str):
            self.nodes = nodes
            self.edges = edges
            self.entry_point = entry_point

        def invoke(self, state: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
            current = self.entry_point
            result = state
            while current != END:
                result = self.nodes[current](result)
                current = self.edges.get(current, END)
            return result

    class StateGraph:
        def __init__(self, state_type: Any):
            self.state_type = state_type
            self.nodes: Dict[str, Callable] = {}
            self.edges: Dict[str, str] = {}
            self.entry_point: str | None = None

        def add_node(self, name: str, fn: Callable) -> None:
            self.nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self.entry_point = name

        def add_edge(self, source: str, target: str) -> None:
            self.edges[source] = target

        def compile(self, checkpointer: Any | None = None) -> _CompiledGraph:
            if self.entry_point is None:
                raise RuntimeError("Entry point is required")
            return _CompiledGraph(self.nodes, self.edges, self.entry_point)
