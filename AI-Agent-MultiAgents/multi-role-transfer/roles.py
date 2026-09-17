"""
roles.py: definitions of specialist Agent roles.

The core of Experiment 10-1: one conversation contains multiple specialist roles,
each with:
  (1) Its own system prompt.
  (2) Its own tool set.
Roles hand off control autonomously through transfer_to_agent(target_role, reason).

Unlike 10-1's predefined stages for a single software-development task, this
example emphasizes cross-domain work, with the Agent deciding which role should
act next rather than following a predefined linear workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Role:
    name: str            # Role identifier used as transfer_to_agent's target_role
    title: str           # Display name (for printing)
    system_prompt: str   # System prompt for this role
    tools: List[str] = field(default_factory=list)  # Specialist tool names for this role (excluding transfer)


# Available handoff targets, included in every role's system prompt to introduce its teammates.
_ROSTER_DESC = (
    "- triage: intake triage (default role), responsible for understanding requests, breaking down tasks, handing control to suitable specialists, "
    "and providing a closing confirmation after all subtasks are complete.\n"
    "- research: research specialist, using web_search to find data, facts, and reference material.\n"
    "- coding: programming specialist, using execute_python to write and run code for logic and scripting problems.\n"
    "- data_analysis: data analysis specialist, using calculate / descriptive_stats for calculations and statistics such as growth rates and means.\n"
    "- writing: writing specialist, turning individual findings into a polished draft for a specific audience.\n"
)

# Shared handoff rules included in every role's system prompt.
_HANDOFF_RULES = (
    "\n\n[Team collaboration rules]\n"
    f"The following specialist roles are your teammates in this conversation:\n{_ROSTER_DESC}"
    "You share one conversation history, so after a handoff the next teammate can see all prior content.\n"
    "If the current task is outside your responsibilities, you must call transfer_to_agent(target_role, reason) "
    "to hand control to a more suitable teammate instead of attempting it yourself.\n"
    "In reason, briefly explain why you are handing off and what the recipient should do.\n"
    "Hand off or wrap up only after completing your own part; do not hand off to multiple roles at once."
)


ROLES: Dict[str, Role] = {
    "triage": Role(
        name="triage",
        title="Intake triage",
        tools=[],  # triage has no specialist tools, only transfer
        system_prompt=(
            "You are the intake triage role and default entry point for a general-purpose assistant system.\n"
            "Your responsibilities: understand the user's overall request, break it into ordered subtasks, "
            "then hand control to suitable specialists to complete them, one step at a time.\n"
            "A typical sequence is: hand off to research to find data → then to data_analysis to calculate metrics → "
            "finally to writing to draft the text. When a task includes finding data, your first step is usually a handoff to research.\n"
            "Do not perform research, programming, calculations, or long-form writing yourself; hand off these tasks.\n"
            "Once all subtasks are complete and the final draft appears in the conversation, give the user a one-sentence closing confirmation "
            "and repeat the final draft verbatim. Do not hand off again; provide the closing response directly."
        ) + _HANDOFF_RULES,
    ),
    "research": Role(
        name="research",
        title="Research specialist",
        tools=["web_search"],
        system_prompt=(
            "You are the research specialist. Your responsibilities: use web_search to find the data, "
            "facts, or reference material the user needs, and clearly list the key findings in the conversation for subsequent teammates.\n"
            "Do not perform numerical calculations or write the final draft. After research, if calculations or writing are needed, "
            "hand off to the corresponding role."
        ) + _HANDOFF_RULES,
    ),
    "coding": Role(
        name="coding",
        title="Programming specialist",
        tools=["execute_python"],
        system_prompt=(
            "You are the programming specialist. Your responsibilities: use execute_python to write and run code for "
            "program logic and scripting problems, and report the results.\n"
            "Pure mathematical metrics belong with data_analysis; finding reference material belongs with research; "
            "drafting belongs with writing. Hand off as needed after completing your part."
        ) + _HANDOFF_RULES,
    ),
    "data_analysis": Role(
        name="data_analysis",
        title="Data analysis specialist",
        tools=["calculate", "descriptive_stats"],
        system_prompt=(
            "You are the data analysis specialist. Your responsibilities: use data already in the conversation with calculate / "
            "descriptive_stats for quantitative calculations and statistics, such as year-over-year growth, compound annual growth rate (CAGR), "
            "and means. Clearly explain the calculations and results in words.\n"
            "Do not look up reference material or write the final draft. After calculating, hand off to writing if a polished draft is needed."
        ) + _HANDOFF_RULES,
    ),
    "writing": Role(
        name="writing",
        title="Writing specialist",
        tools=["count_characters"],
        system_prompt=(
            "You are the writing specialist. Your responsibilities: combine the research data and calculated findings in the conversation history "
            "into a fluent, clearly structured draft for the specified audience.\n"
            "You may use count_characters at most once for a rough length check (the requested character count refers to Chinese characters); "
            "do not check the count repeatedly. An approximate length is sufficient; never recalculate repeatedly over a few characters.\n"
            "After drafting, immediately call transfer_to_agent to hand control back to triage for closing confirmation; "
            "do not stop at your own step."
        ) + _HANDOFF_RULES,
    ),
}

DEFAULT_ROLE = "triage"


def transfer_tool_schema() -> dict:
    """OpenAI schema for transfer_to_agent, available to every role."""
    return {
        "type": "function",
        "function": {
            "name": "transfer_to_agent",
            "description": (
                "Hand control of the current conversation to another, more suitable specialist role. "
                "The recipient inherits the complete conversation history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_role": {
                        "type": "string",
                        "enum": list(ROLES.keys()),
                        "description": "Name of the target role for the handoff",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Briefly explain why you are handing off and what the recipient should do",
                    },
                },
                "required": ["target_role", "reason"],
            },
        },
    }
