# llm_agent_flow.py
"""
LLM-based Agent Flow - An agent that uses an LLM to decide which tools to call.
This module demonstrates tool integration with language model decision-making.
"""
from __future__ import annotations

import json
import logging

from agent_runtime import run_agent_query, serialize_decision

logger = logging.getLogger(__name__)

# -------------------------
# Agent Execution
# -------------------------
def run_llm_agent(user_query: str) -> None:
    print("\n" + "=" * 60)
    print("LLM-BASED AGENT")
    print("=" * 60)
    
    # Step 1: Discover tools
    print("\n[Step 1] Discovering tools...")
    try:
        run_result = run_agent_query(user_query)
        print(f"Found {len(run_result.tools)} tools")
        for tool in run_result.tools:
            print(f"  - {tool['name']}: {tool['description']}")
    except Exception as e:
        print(f"ERROR: Agent run failed: {e}")
        return
    
    # Step 2: Build system prompt and query LLM
    print("\n[Step 2] Planning next action...")
    print(
        "Planner Output: "
        + json.dumps(
            serialize_decision(run_result.decision),
            indent=2,
            ensure_ascii=False,
        )
    )

    # Step 3: Execute tool or return direct answer
    if run_result.decision.tool_request is None:
        print("\n[Step 3] LLM Direct Answer:")
        print(f"  {run_result.final_answer}")
        return

    print(f"\n[Step 3] Executing tool: {run_result.decision.tool_request.name}...")
    print(f"Tool Result: {json.dumps(run_result.tool_result, indent=2, ensure_ascii=False)}")
    print("\nFinal Answer:")
    print(f"  {run_result.final_answer}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_llm_agent("Find information about MCP")
