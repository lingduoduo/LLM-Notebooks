"""
demo.py -- Experiment 10-1 demo entry point: multi-role transitions / transfer_to_agent

Minimal run (one command for the default composite task):
    python demo.py

Other common options:
    python demo.py --list-roles              # Offline: print the role roster and exit (no API key required)
    python demo.py --scenario coding         # Choose another built-in scenario (routes to the coding role)
    python demo.py --task "..."              # Custom task
    python demo.py --role research            # Set the initial role (default: triage)
    python demo.py --interactive             # Interactive conversation (roles and shared history persist across turns)
    python demo.py --model gpt-5.6-luna --max-steps 30

Demonstrates a composite task requiring multiple cross-domain transitions, with an expected
    triage → research → data_analysis → writing
autonomous handoff chain. Each current role decides when to call transfer_to_agent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from roles import ROLES, DEFAULT_ROLE
from orchestrator import MultiRoleOrchestrator, C


# Load .env if available (optional dependency; exported shell variables also work)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Built-in scenarios deliberately span domains to elicit multiple autonomous handoffs.
# Keys are --scenario values; each value is (task text, one-sentence description).
# ---------------------------------------------------------------------------
COMPOSITE_TASK = (
    "I am preparing material for investors. Please:\n"
    "1) Retrieve new energy vehicle sales in China for 2021, 2022, and 2023;\n"
    "2) Calculate the compound annual growth rate (CAGR) over these years;\n"
    "3) Summarize the data and growth conclusion for investors in English, using at most 360 characters."
)

SCENARIOS: dict[str, tuple[str, str]] = {
    "cagr": (
        COMPOSITE_TASK,
        "Default scenario spanning research, calculation, and writing: retrieve sales -> calculate CAGR -> write an investor summary; "
        "expected chain: triage -> research -> data_analysis -> writing.",
    ),
    "solar": (
        "Retrieve newly installed photovoltaic capacity in China for 2021, 2022, and 2023, "
        "calculate the compound annual growth rate (CAGR), and write a one-sentence conclusion for readers.",
        "The same chain with another dataset (research -> data_analysis -> writing), testing the mechanism rather than a memorized answer.",
    ),
    "coding": (
        "Write a Python script to calculate the first 20 Fibonacci numbers and their sum. "
        "Run the script, then explain the result to nontechnical readers in one sentence.",
        "Route to coding to actually execute code using execute_python, then finish with writing/triage.",
    ),
}
DEFAULT_SCENARIO = "cagr"


def print_roster():
    """Print the roster showing five roles with distinct system prompts and tool sets."""
    print(f"{C.BOLD}=== Role roster ({len(ROLES)} specialist roles) ==={C.RESET}")
    for name, role in ROLES.items():
        default_tag = " (default entry point)" if name == DEFAULT_ROLE else ""
        tools = role.tools + ["transfer_to_agent"]
        first_line = role.system_prompt.strip().splitlines()[0]
        print(
            f"{C.CYAN}• {name}{C.RESET} — {role.title}{default_tag}\n"
            f"    Tools: {tools}\n"
            f"    System prompt (first sentence): {first_line}"
        )
    print()


def print_scenarios():
    """Print built-in scenarios for --help / --list-roles."""
    print(f"{C.BOLD}=== Built-in scenarios (--scenario) ==={C.RESET}")
    for key, (_task, desc) in SCENARIOS.items():
        default_tag = " (default)" if key == DEFAULT_SCENARIO else ""
        print(f"{C.CYAN}• {key}{C.RESET}{default_tag} — {desc}")
    print()


def parse_args() -> argparse.Namespace:
    """Optional command-line arguments; without arguments, run the original default composite task."""
    parser = argparse.ArgumentParser(
        prog="demo.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Experiment 10-1 demo: multi-role transitions / transfer_to_agent.\n"
            "Five specialist roles autonomously hand off using transfer_to_agent over shared conversation history, "
            "creating a handoff chain such as triage -> research -> data_analysis -> writing."
        ),
        epilog=(
            "Examples:\n"
            "  python demo.py                     # Run the default scenario (NEV CAGR investor summary)\n"
            "  python demo.py --list-roles        # Offline: list roles and scenarios without API calls\n"
            "  python demo.py --scenario coding   # Choose the scenario that routes to coding\n"
            "  python demo.py --task 'Help me...'    # Custom task\n"
            "  python demo.py --role research     # Start with the research role\n"
            "  python demo.py --interactive       # Interactive conversation with persistent roles and shared history\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default=DEFAULT_SCENARIO,
        help=f"Choose a built-in scenario (default: {DEFAULT_SCENARIO}); --task overrides this. Options: {list(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Custom task text overriding --scenario; otherwise use the selected built-in scenario.",
    )
    parser.add_argument(
        "--role",
        "--starting-role",
        dest="role",
        choices=list(ROLES.keys()),
        default=DEFAULT_ROLE,
        help=f"Set the initial role (default: {DEFAULT_ROLE}). Options: {list(ROLES.keys())}",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: reuse the orchestrator, preserving roles and shared history (Ctrl-C or exit to quit).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENAI_MODEL (defaults to the environment value, or gpt-5.6-luna if unset).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Hard limit on LLM turns per user message to prevent infinite loops (default: 20).",
    )
    parser.add_argument(
        "--list-roles",
        action="store_true",
        help="Print the role roster and built-in scenarios offline, then exit; no API key required (self-check).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the complete, redacted, machine-readable experiment trace and acceptance results.",
    )
    return parser.parse_args()


def print_run_summary(orch: MultiRoleOrchestrator, final: str):
    """Print the handoff chain, role contributions, and final result for a run."""
    print(f"\n{C.BOLD}================ Run summary ================{C.RESET}")
    print(f"{C.MAGENTA}Autonomous handoff chain:{C.RESET} {orch.handoff_chain_str()}")
    print(f"{C.MAGENTA}Handoff count:{C.RESET} {len(orch.handoffs)}")
    for i, h in enumerate(orch.handoffs, 1):
        print(f"  {i}. {h.from_role} → {h.to_role}  |  reason: {h.reason}")
    print(f"\n{C.MAGENTA}Role contributions (tool usage and final-response author):{C.RESET}")
    print(orch.role_work_summary())
    print(f"\n{C.GREEN}Final result:{C.RESET}\n{final}")


def save_evidence(
    path: Path,
    orch: MultiRoleOrchestrator,
    final: str,
    *,
    model: str,
    base_url: str,
    task: str,
) -> dict:
    """Persist direct receipts and fail-closed manuscript acceptance gates."""
    tools_by_role: dict[str, list[str]] = {}
    for role, kind, detail in orch.activity:
        if kind == "tool":
            tools_by_role.setdefault(role, []).append(detail)
    tool_contents = [
        str(message.get("content", ""))
        for message in orch.history
        if message.get("role") == "tool"
    ]
    real_search = any(
        '"provider": "tavily"' in content and '"url":' in content
        for content in tool_contents
    )
    counted_drafts: list[str] = []
    for message in orch.history:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") != "count_characters":
                continue
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments.get("text"), str):
                counted_drafts.append(arguments["text"])
    final_draft = counted_drafts[-1] if counted_drafts else ""
    chain = [orch.handoffs[0].from_role] + [h.to_role for h in orch.handoffs] if orch.handoffs else []
    required_roles_in_order = all(
        role in chain and chain.index(role) < chain.index(next_role)
        for role, next_role in zip(
            ["triage", "research", "data_analysis"],
            ["research", "data_analysis", "writing"],
        )
    )
    final_chars = len(final)
    gates = {
        "real_web_search_with_urls": real_search,
        "triage_research_analysis_writing_order": required_roles_in_order,
        "research_used_web_search": "web_search" in tools_by_role.get("research", []),
        "data_analysis_used_calculate": "calculate" in tools_by_role.get("data_analysis", []),
        "writing_checked_length": "count_characters" in tools_by_role.get("writing", []),
        "final_not_step_limit": not orch.terminated_by_limit,
        "final_nonempty": bool(final.strip()),
        "investor_summary_within_120_characters": bool(final_draft) and len(final_draft) <= 120,
        "shared_history_visible_after_handoffs": all(
            later["history_messages_visible"] >= earlier["history_messages_visible"]
            for earlier, later in zip(orch.api_calls, orch.api_calls[1:])
        ),
    }
    payload = {
        "schema_version": "1.0",
        "experiment": "10-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": {
            "model": model,
            "base_url": base_url,
            "search": "Tavily",
            "credentials_redacted": True,
        },
        "task": task,
        "handoffs": [vars(h) for h in orch.handoffs],
        "handoff_chain": chain,
        "activity": [
            {"role": role, "kind": kind, "detail": detail}
            for role, kind, detail in orch.activity
        ],
        "api_calls": orch.api_calls,
        "steps_used": orch.steps_used,
        "history": orch.history,
        "final_answer": final,
        "final_character_count": final_chars,
        "counted_investor_summary": final_draft,
        "counted_investor_summary_characters": len(final_draft),
        "acceptance_gates": gates,
        "status": "complete" if all(gates.values()) else "incomplete",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMachine-readable evidence: {path}  status={payload['status']}")
    return payload


def run_interactive(orch: MultiRoleOrchestrator):
    """Interactive conversation: reuse the orchestrator, preserving shared history and the current role."""
    print(
        f"{C.BOLD}=== Interactive mode ==={C.RESET}\n"
        f"{C.DIM}Enter a request; type exit / quit or press Ctrl-C to stop. "
        f"Roles and conversation history persist across turns (shared context).{C.RESET}"
    )
    turn = 0
    while True:
        try:
            user_message = input(f"\n{C.BOLD}👤 You (current role: {orch.current_role})> {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExited interactive mode.")
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit", "q"}:
            print("Exited interactive mode.")
            break
        turn += 1
        final = orch.run(user_message)
        print_run_summary(orch, final)


def main():
    args = parse_args()

    # ---- Offline self-check: no API key required ----
    if args.list_roles:
        print_roster()
        print_scenarios()
        return

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Set it and retry.", file=sys.stderr)
        print("(Tip: run `python demo.py --list-roles` to list roles and scenarios without an API key.)",
              file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print_roster()

    orch = MultiRoleOrchestrator(
        client=client,
        model=model,
        max_steps=args.max_steps,
        verbose=True,
        start_role=args.role,
    )

    if args.interactive:
        print(f"{C.BOLD}=== Model={model}, initial role={args.role} ==={C.RESET}")
        run_interactive(orch)
        return

    # ---- Scripted mode: run one composite task end to end ----
    task = args.task if args.task is not None else SCENARIOS[args.scenario][0]
    scenario_tag = "Custom task" if args.task is not None else f"Scenario {args.scenario}"
    print(f"{C.BOLD}=== Starting ({scenario_tag}, model={model}, initial role={args.role}) ==={C.RESET}")

    final = orch.run(task)
    print_run_summary(orch, final)
    if args.output:
        save_evidence(
            args.output,
            orch,
            final,
            model=model,
            base_url=base_url,
            task=task,
        )


if __name__ == "__main__":
    main()
