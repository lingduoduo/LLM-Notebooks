# agent_langgraph.py
"""
LangGraph agent built on BaseAgent.

Each graph node publishes typed AgentThought events so callers can consume a
live stream via agent.stream() or get the final AgentResult via agent.invoke().
"""
from __future__ import annotations

import importlib
import logging
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from agent_runtime import (
    AgentDecision,
    decide_next_action,
    discover_tool_catalog,
    execute_tool_request,
    format_agent_answer,
)
from base_agent import (
    AgentConfig,
    AgentThought,
    BaseAgent,
    QueueEvent,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_langgraph_primitives() -> tuple[Any, Any]:
    """Import LangGraph even though this file name shadows the package name."""
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
    """Concrete graph state shared across all nodes."""

    user_query: str
    task_id: str
    history: list[Any]
    iteration_count: int
    terminated: bool   # set by plan_node to skip respond_node on max-iterations
    tools: list[dict[str, Any]]
    decision: AgentDecision | None
    tool_result: dict[str, Any] | None
    final_answer: str | None


# -------------------------
# Agent Implementation
# -------------------------
class MCPLangGraphAgent(BaseAgent):
    """MCP agent implemented as an explicit LangGraph StateGraph.

    Nodes publish AgentThought events to the queue manager so that stream()
    and invoke() (both inherited from BaseAgent) work without any extra wiring.
    """

    def _build_agent(self) -> Any:
        logger.info("Building LangGraph agent")
        end_sentinel, state_graph_cls = _load_langgraph_primitives()
        qm = self._queue_manager  # captured by node closures

        MAX_ITER_RESPONSE = (
            "I've reached the maximum number of reasoning steps. "
            "Please rephrase your question or increase max_iterations."
        )

        def discover_node(state: AgentState) -> dict[str, Any]:
            t0 = time.perf_counter()
            tools = discover_tool_catalog()
            latency = time.perf_counter() - t0
            logger.info("Discovered %s tools", len(tools))
            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.AGENT_THOUGHT,
                    thought=f"Discovered {len(tools)} tools from MCP server.",
                    latency=latency,
                ),
            )
            return {"tools": tools}

        def plan_node(state: AgentState) -> dict[str, Any]:
            # Guard: publish a fallback answer and end the graph if iteration budget is exhausted.
            if state["iteration_count"] >= self.agent_config.max_iterations:
                logger.warning("Max iterations (%s) reached", self.agent_config.max_iterations)
                qm.publish(
                    state["task_id"],
                    AgentThought(
                        id=uuid.uuid4(),
                        task_id=state["task_id"],
                        event=QueueEvent.AGENT_MESSAGE,
                        thought=MAX_ITER_RESPONSE,
                        answer=MAX_ITER_RESPONSE,
                    ),
                )
                qm.publish(
                    state["task_id"],
                    AgentThought(
                        id=uuid.uuid4(),
                        task_id=state["task_id"],
                        event=QueueEvent.AGENT_END,
                    ),
                )
                return {
                    "decision": AgentDecision(tool_request=None, answer=MAX_ITER_RESPONSE),
                    "terminated": True,
                }

            t0 = time.perf_counter()
            decision = decide_next_action(state["user_query"], state["tools"])
            latency = time.perf_counter() - t0
            tool_name = (
                decision.tool_request.name if decision.tool_request else "direct_answer"
            )
            logger.info("Planner selected: %s", tool_name)
            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.AGENT_THOUGHT,
                    thought=f"Planner selected action: {tool_name}.",
                    latency=latency,
                ),
            )
            return {
                "decision": decision,
                "iteration_count": state["iteration_count"] + 1,
                "terminated": False,
            }

        def execute_node(state: AgentState) -> dict[str, Any]:
            decision = state["decision"]
            if decision is None or decision.tool_request is None:
                return {"tool_result": None}

            tool_name = decision.tool_request.name
            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.TOOL_CALL,
                    thought=f"Calling tool: {tool_name}",
                    observation=str(decision.tool_request.arguments),
                ),
            )

            t0 = time.perf_counter()
            try:
                result = execute_tool_request(decision.tool_request)
            except Exception as exc:
                logger.error("Tool %s failed: %s", tool_name, exc)
                # Publish a non-terminal error observation so respond_node can still run.
                qm.publish(
                    state["task_id"],
                    AgentThought(
                        id=uuid.uuid4(),
                        task_id=state["task_id"],
                        event=QueueEvent.TOOL_RESULT,
                        thought=f"Tool {tool_name} failed.",
                        observation=str(exc),
                    ),
                )
                return {"tool_result": None}
            latency = time.perf_counter() - t0

            logger.info("Executed tool %s", tool_name)
            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.TOOL_RESULT,
                    thought=f"Tool {tool_name} returned a result.",
                    observation=str(result),
                    latency=latency,
                ),
            )
            return {"tool_result": result}

        def respond_node(state: AgentState) -> dict[str, Any]:
            decision = state["decision"]
            if decision is None:
                raise ValueError("plan_node must run before respond_node")

            t0 = time.perf_counter()
            final_answer = format_agent_answer(
                user_query=state["user_query"],
                decision=decision,
                tool_result=state.get("tool_result"),
            )
            latency = time.perf_counter() - t0

            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.AGENT_MESSAGE,
                    answer=final_answer,
                    latency=latency,
                ),
            )
            qm.publish(
                state["task_id"],
                AgentThought(
                    id=uuid.uuid4(),
                    task_id=state["task_id"],
                    event=QueueEvent.AGENT_END,
                ),
            )
            return {"final_answer": final_answer}

        def route_after_plan(state: AgentState) -> str:
            if state.get("terminated", False):
                return "end"
            decision = state["decision"]
            return "execute" if decision and decision.tool_request else "respond"

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
            {"execute": "execute", "respond": "respond", "end": end_sentinel},
        )
        graph.add_edge("execute", "respond")
        graph.add_edge("respond", end_sentinel)

        logger.info("Graph construction completed")
        return graph.compile()


# -------------------------
# Main Execution
# -------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting LangGraph agent")

    agent = MCPLangGraphAgent(agent_config=AgentConfig(max_iterations=10))

    user_query = (
        "Search docs about MCP, "
        "then summarize the difference between MCP and basic tool calling."
    )
    logger.info("Processing query: %s", user_query)

    print("\n" + "=" * 60)
    print("LANGGRAPH AGENT — STREAMING")
    print("=" * 60)

    result = agent.invoke({
        "user_query": user_query,
        "tools": [],
        "decision": None,
        "tool_result": None,
        "final_answer": None,
        "terminated": False,
    })

    print(f"\n[Status]  {result.status.value}")
    print(f"[Latency] {result.latency:.3f}s")

    print("\n[Thoughts]")
    for thought in result.agent_thoughts:
        if thought.thought:
            print(f"  [{thought.event.value}] {thought.thought}")

    print(f"\n[Answer]\n  {result.answer or '(no answer)'}")

    if result.error:
        print(f"\n[Error]\n  {result.error}")

    logger.info("Agent execution completed")


if __name__ == "__main__":
    main()
