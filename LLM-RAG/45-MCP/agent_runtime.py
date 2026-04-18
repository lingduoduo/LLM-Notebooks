"""
Shared agent runtime used by both the simple and LangGraph flows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from agent_client import call_tool, choose_tool, discover_tools_cached

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
    tools_json = _serialize_tools_for_prompt(tuple(_tool_signature(tool) for tool in tools))
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


def _tool_signature(tool: dict[str, Any]) -> tuple[str, str, str]:
    """Create a stable, hashable representation of a tool definition."""
    return (
        str(tool.get("name", "")),
        str(tool.get("description", "")),
        json.dumps(tool.get("input_schema", {}), sort_keys=True),
    )


@lru_cache(maxsize=8)
def _serialize_tools_for_prompt(tool_signatures: tuple[tuple[str, str, str], ...]) -> str:
    """Cache prompt serialization for repeated runs against the same tool catalog."""
    tools = [
        {
            "name": name,
            "description": description,
            "input_schema": json.loads(input_schema),
        }
        for name, description, input_schema in tool_signatures
    ]
    return json.dumps(tools, indent=2)


def decide_next_action(
    user_query: str,
    tools: list[dict[str, Any]],
    tool_history: list[dict[str, Any]] | None = None,
) -> AgentDecision:
    """Select the next tool to call, or signal readiness to respond.

    tool_history holds the results of all tool calls made so far in this turn.
    When it is non-empty the planner returns a no-tool decision so the graph
    routes to the respond node — enabling multi-step tool chaining: each
    iteration the planner re-evaluates whether another tool is needed.
    A real LLM planner would reason over tool_history before deciding; the
    deterministic version here terminates after the first successful result.
    """
    system_prompt = build_system_prompt(tools)
    logger.info("Planning next action for query: %s", user_query)
    logger.debug("Planner prompt size: %s characters", len(system_prompt))

    if tool_history:
        logger.info(
            "Planner: %d tool result(s) accumulated — routing to respond", len(tool_history)
        )
        return AgentDecision(tool_request=None, answer=None)

    selected_tool = choose_tool(user_query, tools)
    if selected_tool:
        return AgentDecision(
            tool_request=ToolRequest(
                name=selected_tool["name"],
                arguments=selected_tool["arguments"],
            )
        )

    return AgentDecision(
        tool_request=None,
        answer=(
            "I do not have a sufficiently reliable tool for this request. "
            "Please connect an MCP tool that can verify the answer."
        ),
    )


def execute_tool_request(
    tool_request: ToolRequest,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a concrete tool request against the MCP server."""
    if tools is not None:
        available_tools = {tool.get("name") for tool in tools}
        if tool_request.name not in available_tools:
            raise ValueError(f"Tool is not available in discovered MCP catalog: {tool_request.name}")
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
    # Dispatch on display_hint embedded by the tool; fall back to tool name so
    # tools that predate the hint system still work (What/How decoupling:
    # the formatter does not need to know which tool was called).
    hint = payload.get("display_hint") or decision.tool_request.name

    if hint in ("search_results", "search_docs"):
        results = payload.get("results", [])
        if results:
            return "Search results:\n- " + "\n- ".join(results)
        return f"No documentation matched the query for: {user_query}"

    if hint in ("weather", "get_weather"):
        return (
            f"The weather in {payload.get('city', 'unknown')} is "
            f"{payload.get('temperature', 'N/A')} {payload.get('unit', '')} "
            f"and {payload.get('condition', 'unknown')}."
        )

    if hint in ("arithmetic", "add_numbers"):
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
        execute_tool_request(decision.tool_request, tools)
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
