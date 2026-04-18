"""
Shared agent runtime used by both the simple and LangGraph flows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent_client import call_tool, discover_tools_cached

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolRequest:
    """Concrete tool call selected by an agent planner."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentDecision:
    """Planner output for a user query."""

    tool_request: ToolRequest | None
    answer: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Complete result of a single agent turn."""

    tools: list[dict[str, Any]]
    decision: AgentDecision
    tool_result: dict[str, Any] | None
    final_answer: str


def discover_tool_catalog(force_refresh: bool = False) -> list[dict[str, Any]]:
    """Load the available tool catalog from the MCP server."""
    return discover_tools_cached(force_refresh=force_refresh)


def build_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build a concrete tool-selection prompt for an LLM-style planner."""
    tools_json = json.dumps(tools, indent=2)
    return f"""You are an AI agent with access to the following tools:

{tools_json}

When you decide to use a tool, respond with ONLY valid JSON (no additional text):
{{
  "tool_name": "tool_name_here",
  "arguments": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}

If no tool is needed or you can answer directly, respond with:
{{
  "tool_name": null,
  "answer": "Your direct answer here"
}}

Always respond with valid JSON only."""


def decide_next_action(
    user_query: str,
    tools: list[dict[str, Any]],
) -> AgentDecision:
    """
    Make a concrete next-step decision for the current query.

    This keeps the example deterministic while exposing the same shape an LLM
    planner would return.
    """
    system_prompt = build_system_prompt(tools)
    logger.info("Planning next action for query: %s", user_query)
    logger.debug("Planner prompt size: %s characters", len(system_prompt))

    query = user_query.lower().strip()

    if any(word in query for word in ["weather", "temperature", "climate", "hot", "cold"]):
        return AgentDecision(
            tool_request=ToolRequest(
                name="get_weather",
                arguments={"city": "Boston", "unit": "celsius"},
            )
        )

    if any(word in query for word in ["search", "mcp", "document", "find", "look"]):
        return AgentDecision(
            tool_request=ToolRequest(
                name="search_docs",
                arguments={"query": "MCP"},
            )
        )

    if any(word in query for word in ["add", "sum", "total", "calculate", "math"]):
        return AgentDecision(
            tool_request=ToolRequest(
                name="add_numbers",
                arguments={"a": 10, "b": 20},
            )
        )

    return AgentDecision(
        tool_request=None,
        answer="I can answer directly without needing external tools.",
    )


def execute_tool_request(tool_request: ToolRequest) -> dict[str, Any]:
    """Execute a concrete tool request against the MCP server."""
    return call_tool(tool_request.name, tool_request.arguments)


def format_agent_answer(
    user_query: str,
    decision: AgentDecision,
    tool_result: dict[str, Any] | None = None,
) -> str:
    """Turn a decision and optional tool result into a user-facing answer."""
    if decision.tool_request is None:
        return decision.answer or "No response"

    if tool_result is None:
        return f"Selected tool `{decision.tool_request.name}` but no result was produced."

    payload = tool_result.get("result", tool_result)
    tool_name = decision.tool_request.name

    if tool_name == "search_docs":
        results = payload.get("results", [])
        if results:
            return "Search results:\n- " + "\n- ".join(results)
        return f"No documentation matched the query for: {user_query}"

    if tool_name == "get_weather":
        return (
            f"The weather in {payload.get('city', 'unknown')} is "
            f"{payload.get('temperature', 'N/A')} {payload.get('unit', '')} "
            f"and {payload.get('condition', 'unknown')}."
        )

    if tool_name == "add_numbers":
        result = payload.get("result")
        return f"The result is {result}." if result is not None else json.dumps(payload, indent=2)

    return json.dumps(payload, indent=2, ensure_ascii=False)


def serialize_decision(decision: AgentDecision) -> dict[str, Any]:
    """Convert a decision into a JSON-friendly structure."""
    return {
        "tool_name": decision.tool_request.name if decision.tool_request else None,
        "arguments": decision.tool_request.arguments if decision.tool_request else {},
        "answer": decision.answer,
    }


def run_agent_query(user_query: str) -> AgentRunResult:
    """Run one complete query through the shared planner and executor."""
    tools = discover_tool_catalog()
    decision = decide_next_action(user_query, tools)
    tool_result = (
        execute_tool_request(decision.tool_request)
        if decision.tool_request is not None
        else None
    )
    final_answer = format_agent_answer(user_query, decision, tool_result)
    return AgentRunResult(
        tools=tools,
        decision=decision,
        tool_result=tool_result,
        final_answer=final_answer,
    )
