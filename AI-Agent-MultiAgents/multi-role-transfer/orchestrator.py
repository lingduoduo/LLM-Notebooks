"""
orchestrator.py: orchestrator for handoffs between specialist roles.

Core mechanism (Experiment 10-1):
- Maintain a shared conversation history of user, assistant, and tool messages.
- Prepend the current role's system prompt for each model call and expose only
  that role's tools plus transfer_to_agent.
- The model can:
    1) Call its own specialist tools through ordinary function calling.
    2) Call transfer_to_agent to hand control to another role. The orchestrator
       replaces the system prompt and tool set while preserving history, so the
       new role inherits the complete conversation as shared context.
- Continue until a role produces a final response without tool calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from openai import OpenAI

from roles import ROLES, DEFAULT_ROLE, transfer_tool_schema
from tools import TOOL_SCHEMAS, TOOL_IMPLEMENTATIONS


# ---- Terminal colors (no third-party dependencies)----
class C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"


@dataclass
class Handoff:
    from_role: str
    to_role: str
    reason: str


class MultiRoleOrchestrator:
    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-5.6-luna",
        max_steps: int = 20,
        max_output_tokens: Optional[int] = None,
        verbose: bool = True,
        start_role: str = DEFAULT_ROLE,
        provider_receipt_sink: Optional[Callable[[dict], None]] = None,
        tool_receipt_sink: Optional[Callable[[dict], None]] = None,
    ):
        if start_role not in ROLES:
            raise ValueError(f"Unknown starting role {start_role!r}; available: {list(ROLES.keys())}")
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose

        self.history: List[dict] = []          # Shared conversation history (excluding system messages)
        self.current_role: str = start_role    # Role currently in control (configurable starting role)
        self.handoffs: List[Handoff] = []      # Handoff chain
        self._tool_call_counts: Dict[str, int] = {}  # Counts of identical tool calls to prevent infinite loops
        # Activity records: (role, kind, detail), kind in {"tool", "transfer", "final"},
        # used to summarize each role's work after the run.
        self.activity: List[tuple] = []
        self.api_calls: List[dict] = []
        self.steps_used: int = 0
        self.terminated_by_limit: bool = False
        self.provider_receipt_sink = provider_receipt_sink
        self.tool_receipt_sink = tool_receipt_sink

    # -------------------------------------------------------------- Tool assembly
    def _tools_for_current_role(self) -> List[dict]:
        """Tools visible to the current role = specialist tools + transfer_to_agent."""
        role = ROLES[self.current_role]
        schemas = [TOOL_SCHEMAS[name] for name in role.tools]
        schemas.append(transfer_tool_schema())  # Every role can hand off control
        return schemas

    def _messages_for_api(self) -> List[dict]:
        """Prepend the current role's system prompt to the shared history."""
        system_msg = {"role": "system", "content": ROLES[self.current_role].system_prompt}
        return [system_msg] + self.history

    # -------------------------------------------------------------- Logging
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _log_role_banner(self):
        role = ROLES[self.current_role]
        self._log(
            f"\n{C.BOLD}{C.CYAN}┌── Current role: {role.title} ({role.name}){C.RESET}"
            f"{C.DIM}  Tools: {role.tools + ['transfer_to_agent']}{C.RESET}"
        )

    # -------------------------------------------------------------- Single step
    def _run_one_llm_turn(self) -> Optional[str]:
        """
        Run one model call and process its tool calls.
        Returns:
          - None: continue the loop after a tool call or handoff.
          - str: final response with no tool calls; the workflow is complete.
        """
        self._log_role_banner()

        kwargs = dict(
            model=self.model,
            messages=self._messages_for_api(),
            tools=self._tools_for_current_role(),
            temperature=0,
        )
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Reasoning models (such as gpt-5.x) accept only the default temperature;
            # remove this parameter and retry once, as in book-translation / voice-werewolf.
            if "temperature" not in str(e).lower():
                raise
            kwargs.pop("temperature", None)
            response = self.client.chat.completions.create(**kwargs)
        if self.provider_receipt_sink:
            self.provider_receipt_sink({
                "kind": "chat_completion",
                "role": self.current_role,
                "request": kwargs,
                "response": response.model_dump(mode="json"),
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "duration_seconds": round(time.monotonic() - started, 3),
            })
        msg = response.choices[0].message
        usage = getattr(response, "usage", None)
        self.api_calls.append({
            "role": self.current_role,
            "response_id": getattr(response, "id", None),
            "history_messages_visible": len(self.history),
            "tools_visible": [
                tool["function"]["name"] for tool in self._tools_for_current_role()
            ],
            "usage": usage.model_dump(mode="json") if usage is not None else None,
        })

        # No tool calls => final response
        if not msg.tool_calls:
            content = msg.content or ""
            self.history.append({"role": "assistant", "content": content})
            self.activity.append((self.current_role, "final", ""))
            self._log(f"{C.GREEN}└── [{self.current_role}] Final response:{C.RESET}\n{content}")
            return content

        # With tool calls, first append the assistant message (including tool_calls) to history
        self.history.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        pending_transfer: Optional[Handoff] = None

        # Process each tool call and append a tool message for it, as required by OpenAI
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == "transfer_to_agent":
                target = args.get("target_role", "")
                reason = args.get("reason", "")
                if pending_transfer is not None:
                    # Only one handoff per turn can be applied.  Earlier versions
                    # still answered "Handed off to ..." for every extra call and
                    # then silently kept only the last one, so the model was told
                    # about handoffs that never happened.
                    result = (
                        f"Handoff failed: you are already handing off to "
                        f"{pending_transfer.to_role} in this turn and cannot hand off to "
                        f"{target!r} as well. Hand off to one role at a time; the recipient "
                        "can pass control onwards."
                    )
                    self._log(
                        f"{C.RED}└── transfer refused: {pending_transfer.to_role} handoff "
                        f"already pending this turn{C.RESET}"
                    )
                elif isinstance(target, str) and target == self.current_role:
                    # Reject self-handoffs: ask the model to use its own tools or choose another role
                    result = (
                        f"Handoff failed: you are already the {target} role and cannot hand off to yourself. "
                        "Use your own tools to complete the current part, or hand off to another role."
                    )
                    self._log(f"{C.RED}└── transfer rejected: cannot hand off to yourself ({target}){C.RESET}")
                elif isinstance(target, str) and target in ROLES:
                    pending_transfer = Handoff(self.current_role, target, reason)
                    self.activity.append((self.current_role, "transfer", target))
                    result = f"Handed off to {target}. The recipient will inherit the complete conversation history and continue."
                    self._log(
                        f"{C.MAGENTA}└── ⇢ transfer_to_agent: "
                        f"{self.current_role} → {target}{C.RESET}\n"
                        f"    {C.YELLOW}reason:{C.RESET} {reason}"
                    )
                else:
                    result = f"Handoff failed: unknown role {target!r}. Available: {list(ROLES.keys())}"
                    self._log(f"{C.RED}└── transfer failed: unknown role {target!r}{C.RESET}")
            else:
                impl = TOOL_IMPLEMENTATIONS.get(name)
                if impl is None:
                    result = f"Tool {name} does not exist."
                else:
                    try:
                        if name == "web_search" and self.tool_receipt_sink:
                            result = impl(**args, receipt_sink=self.tool_receipt_sink)
                        else:
                            result = impl(**args)
                    except (TypeError, ValueError, RuntimeError) as exc:
                        # The model may supply incorrect or missing arguments (e.g. {"q": ...} instead of {"query": ...})
                        # or values that cannot be converted; return the error as a tool result so it can correct itself
                        # without crashing the entire handoff workflow.
                        result = f"Tool {name} call failed: {exc}. Check argument names and values, then retry."
                    self.activity.append((self.current_role, "tool", name))
                # Prevent infinite loops: give corrective feedback for repeated (role, tool, arguments) calls
                sig = f"{self.current_role}:{name}:{tc.function.arguments}"
                self._tool_call_counts[sig] = self._tool_call_counts.get(sig, 0) + 1
                if self._tool_call_counts[sig] >= 3:
                    result += (
                        "\n[System reminder] You have repeated the exact same call multiple times. Stop repeating it; "
                        "provide the final text directly, or call transfer_to_agent to hand off to the next role."
                    )
                self._log(
                    f"{C.BLUE}└── 🔧 Calling tool {name}{C.RESET} "
                    f"{C.DIM}args={args}{C.RESET}\n"
                    f"    {C.DIM}→ {result[:300]}{C.RESET}"
                )

            self.history.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

        # After processing all tool calls in this turn, apply any handoff while preserving history
        if pending_transfer is not None:
            self.handoffs.append(pending_transfer)
            self.current_role = pending_transfer.to_role

        return None  # Continue the loop

    # -------------------------------------------------------------- Main loop
    def run(self, user_message: str) -> str:
        """Process a user message through the complete multi-role handoff workflow and return the final response."""
        self.history.append({"role": "user", "content": user_message})
        self._log(f"{C.BOLD}👤 User:{C.RESET} {user_message}")

        final_answer = ""
        for step in range(self.max_steps):
            self.steps_used = step + 1
            result = self._run_one_llm_turn()
            if result is not None:
                final_answer = result
                break
        else:
            self.terminated_by_limit = True
            final_answer = "(Maximum step limit reached; workflow terminated)"
            self._log(f"{C.RED}{final_answer}{C.RESET}")

        return final_answer

    # -------------------------------------------------------------- Summary
    def handoff_chain_str(self) -> str:
        """Return a readable handoff chain, e.g. triage → research → data_analysis → writing → triage."""
        if not self.handoffs:
            return DEFAULT_ROLE + " (no handoffs)"
        chain = [self.handoffs[0].from_role]
        for h in self.handoffs:
            chain.append(h.to_role)
        return " → ".join(chain)

    def role_work_summary(self) -> str:
        """
        Summarize each role's work in order of first appearance, listing the
        specialist tools each role actually called and who produced the final response.
        This demonstrates specialist roles taking turns on a shared history.
        """
        order: List[str] = []
        tools_by_role: Dict[str, List[str]] = {}
        final_role: Optional[str] = None
        for role, kind, detail in self.activity:
            if role not in order:
                order.append(role)
                tools_by_role[role] = []
            if kind == "tool" and detail not in tools_by_role[role]:
                tools_by_role[role].append(detail)
            elif kind == "final":
                final_role = role
        if not order:
            return "(No role activity recorded)"
        width = max(len(r) for r in order)
        lines: List[str] = []
        for role in order:
            used = tools_by_role[role]
            desc = ", ".join(used) if used else "(Routing/handoffs only; no specialist tools used)"
            if role == final_role:
                desc += "  ⇒ Produced final response"
            lines.append(f"  {role.ljust(width)} : {desc}")
        return "\n".join(lines)
