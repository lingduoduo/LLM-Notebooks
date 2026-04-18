# langgraph.py
"""
LangGraph agent built on the same concrete runtime as the standard flow.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from agent_runtime import (
    AgentDecision,
    decide_next_action,
    discover_tool_catalog,
    execute_tool_request,
    format_agent_answer,
    serialize_decision,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_langgraph_primitives() -> tuple[Any, Any]:
    """
    Import LangGraph even though this file name shadows the package name.
    """
    current_dir = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            path
            for path in sys.path
            if path not in ("", ".") and Path(path).resolve() != current_dir
        ]
        graph_module = importlib.import_module("langgraph.graph")
        return graph_module.END, graph_module.StateGraph
    finally:
        sys.path = original_sys_path


# -------------------------
# State Definition
# -------------------------
class AgentState(TypedDict):
    """Concrete graph state shared across explicit nodes."""

    user_query: str
    tools: list[dict[str, Any]]
    decision: AgentDecision | None
    tool_result: dict[str, Any] | None
    final_answer: str | None


# -------------------------
# Graph Builder
# -------------------------
def build_graph() -> Any:
    """Build a concrete LangGraph workflow around the shared agent runtime."""
    logger.info("Building LangGraph agent")
    end_sentinel, state_graph_cls = _load_langgraph_primitives()

    def discover_node(_: AgentState) -> dict[str, Any]:
        tools = discover_tool_catalog()
        logger.info("LangGraph discovered %s tools", len(tools))
        return {"tools": tools}

    def plan_node(state: AgentState) -> dict[str, Any]:
        decision = decide_next_action(state["user_query"], state["tools"])
        logger.info(
            "LangGraph planner selected %s",
            decision.tool_request.name if decision.tool_request else "direct_answer",
        )
        return {"decision": decision}

    def execute_node(state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None or decision.tool_request is None:
            return {"tool_result": None}

        result = execute_tool_request(decision.tool_request)
        logger.info("LangGraph executed tool %s", decision.tool_request.name)
        return {"tool_result": result}

    def respond_node(state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None:
            raise ValueError("Planner node must run before respond node")

        final_answer = format_agent_answer(
            user_query=state["user_query"],
            decision=decision,
            tool_result=state.get("tool_result"),
        )
        return {"final_answer": final_answer}

    def route_after_plan(state: AgentState) -> str:
        decision = state["decision"]
        if decision and decision.tool_request:
            return "execute"
        return "respond"

    graph = state_graph_cls(AgentState)
    graph.add_node("discover", discover_node)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)
    graph.set_entry_point("discover")
    graph.add_edge("discover", "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"execute": "execute", "respond": "respond"},
    )
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", end_sentinel)

    logger.info("Graph construction completed")
    return graph.compile()


# -------------------------
# Main Execution
# -------------------------
async def main() -> None:
    """
    Main execution function for the LangGraph agent.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting LangGraph agent")
    
    try:
        app = build_graph()
        logger.info("Graph built successfully")

        user_query = (
            "Search docs about MCP, "
            "then summarize the difference between MCP and basic tool calling."
        )
        logger.info(f"Processing query: {user_query}")

        result = await app.ainvoke(
            {
                "user_query": user_query,
                "tools": [],
                "decision": None,
                "tool_result": None,
                "final_answer": None,
            }
        )

        print("\n" + "=" * 60)
        print("LANGGRAPH AGENT RESULT")
        print("=" * 60)
        decision = result["decision"]
        tool_name = decision.tool_request.name if decision and decision.tool_request else "none"
        print(f"\n[Selected Tool]\n  {tool_name}")
        print(f"\n[Decision]\n  {json.dumps(serialize_decision(decision), ensure_ascii=False)}")
        if result["tool_result"] is not None:
            print(f"\n[Tool Result]\n  {result['tool_result']}")
        print(f"\n[Final Answer]\n  {result['final_answer']}")

        logger.info("Agent execution completed successfully")
    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
