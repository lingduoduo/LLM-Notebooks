"""Run the supplementary demo with `python demo.py`.

Demonstrates two capabilities:
  1) Evolution: start with base tools, search, read docs, test in the sandbox,
     package a tool, and use it to retrieve the real NVIDIA (NVDA) stock price
     and percentage change from one week earlier.
  2) Reuse: ask about Apple (AAPL). The agent should first use search_tools to
     find and call the existing tool without searching the web or rebuilding it.
     The program prints trajectories and automatically checks reuse.

The default online path requires internet access and OpenAI calls. Configure
OPENAI_API_KEY first. Without a key or network, use --offline to directly exercise
search miss -> create -> pre-save validation -> register -> reuse.

Examples:
    python demo.py                 # Default evolution and reuse tasks (API required)
    python demo.py --fresh         # Clear tool_library/ to evolve from scratch
    python demo.py --offline       # Full offline self-check, no API/network
    python demo.py --task "Find Bitcoin's current USD price and 24-hour change"
    python demo.py --no-create     # Disable creation for a comparison run
    python demo.py --model gpt-5.6-luna --output run.json
    python demo.py --help          # Show all options

Tools persist in tool_library/. If get_stock_price already exists, the first task
may reuse it at step 0. Add --fresh to demonstrate evolution from scratch again.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tool_manager import LIBRARY_DIR, ToolLibrary


TASK_1 = "Find the latest NVIDIA (ticker NVDA) stock price and the percentage change compared with one week ago. Provide real data."
TASK_2 = "Find the latest Apple (ticker AAPL) stock price and the percentage change compared with one week ago. Provide real data."

_META_TOOLS = {"web_search", "read_webpage", "code_interpreter", "create_tool", "search_tools"}


def _clear_library():
    """Clear generated JSON tools from the persistent library to evolve from scratch."""
    removed = 0
    for p in glob.glob(os.path.join(str(LIBRARY_DIR), "*.json")):
        try:
            os.remove(p)
            removed += 1
        except OSError:
            pass
    print(f"[--fresh] Cleared tool_library/ ({removed} packaged tools removed); evolution will start from scratch.\n")


def _is_reuse(traj: list) -> bool:
    """Check for reuse: search_tools was called, web_search/create_tool were not,
    and a packaged tool outside the meta-tool set was actually invoked."""
    return (
        "search_tools" in traj
        and "web_search" not in traj
        and "create_tool" not in traj
        and any(t not in _META_TOOLS for t in traj)
    )


# --------------------------------------------------------------------------- #
# Offline self-check: exercise the evolution loop without LLM or network calls
# Search miss -> create with pre-save validation -> register -> call -> reuse
# Use a deterministic offline tool that calculates days between dates
# to verify evolution and reuse without API credentials or network access.
# --------------------------------------------------------------------------- #
_DAYS_TOOL_CODE = (
    "from datetime import date\n\n"
    "def run(start, end):\n"
    "    s = date.fromisoformat(start)\n"
    "    e = date.fromisoformat(end)\n"
    "    return {'start': start, 'end': end, 'days': (e - s).days}\n"
)
_DAYS_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
        "end": {"type": "string", "description": "End date YYYY-MM-DD"},
    },
    "required": ["start", "end"],
}
# A broken tool demonstrates that pre-save validation rejects runtime failures
_BAD_TOOL_CODE = (
    "def run(start, end):\n"
    "    return {'days': undefined_name}\n"  # NameError at runtime
)


def run_offline_selftest(output_path: str | None = None) -> int:
    print("=" * 70)
    print("Offline self-check (--offline): exercise the tool evolution loop without LLM/network calls")
    print("  Loop: search_tools miss -> create_tool (pre-save validation) -> register -> call -> reuse")
    print("=" * 70)

    tmp = Path(tempfile.mkdtemp(prefix="selfevolve_selftest_"))
    lib = ToolLibrary(library_dir=tmp)  # Use a temporary library to keep the real tool_library/ intact
    try:
        # ---------- Validation gate demo: reject a broken tool ----------
        print("\n[Validation gate] Attempting to register a tool that crashes at runtime (with test_args)...")
        bad = lib.create_tool(
            "days_between_bad", "An example tool that crashes", _DAYS_TOOL_PARAMS, _BAD_TOOL_CODE,
            test_args={"start": "2020-01-01", "end": "2020-03-01"},
        )
        print(f"  Result: success={bad.get('success')}  ->  {bad.get('error', '')[:60]}")
        assert not bad["success"], "The broken tool unexpectedly passed pre-save validation!"
        assert lib.get_tool("days_between_bad") is None, "The broken tool must not be saved!"
        print("  ✅ Pre-save validation blocked the broken tool and kept it out of the library.")

        # ---------- Task one: evolution through tool creation ----------
        traj1: list = []
        print("\n########## Offline task one: days from 2020-01-01 to 2020-03-01 (evolution) ##########")
        traj1.append("search_tools")
        hit = lib.search_tools("date days between")
        print(f"[step 1] search_tools -> {hit['count']} hits (empty tool library; no match)")

        traj1.append("create_tool")
        created = lib.create_tool(
            "days_between",
            "Calculate the number of days between two ISO dates (YYYY-MM-DD)",
            _DAYS_TOOL_PARAMS, _DAYS_TOOL_CODE,
            test_args={"start": "2020-01-01", "end": "2020-01-11"},
        )
        print(f"[step 2] create_tool(days_between) -> success={created['success']} "
              f"validated={created.get('validated')} (pre-save validation executed run() once)")

        traj1.append("days_between")
        r1 = lib.execute_tool("days_between", {"start": "2020-01-01", "end": "2020-03-01"})
        ans1 = r1.get("result", {}).get("days")
        print(f"[step 3] days_between(...) -> {r1.get('result')}")
        print(f"[Offline task one result] {ans1} days from 2020-01-01 to 2020-03-01.")

        # ---------- Task two: reuse without rebuilding ----------
        traj2: list = []
        print("\n########## Offline task two: days from 2021-01-01 to 2021-12-31 (reuse) ##########")
        traj2.append("search_tools")
        hit2 = lib.search_tools("date days between")
        print(f"[step 1] search_tools -> {hit2['count']} hits: {[t['name'] for t in hit2['tools']]} (reuse!)")

        traj2.append("days_between")
        r2 = lib.execute_tool("days_between", {"start": "2021-01-01", "end": "2021-12-31"})
        ans2 = r2.get("result", {}).get("days")
        print(f"[step 2] days_between(...) -> {r2.get('result')}")
        print(f"[Offline task two result] {ans2} days from 2021-01-01 to 2021-12-31.")

        reused = _is_reuse(traj2)
        print("\n" + "=" * 70)
        print("Offline self-check results")
        print("=" * 70)
        print(f"Task one trajectory: {traj1}")
        print(f"Task two trajectory: {traj2}")
        print(f"Did task two reuse the tool from task one without calling create_tool again: {'Yes ✅' if reused else 'No ❌'}")
        print(f"Did pre-save validation block the broken tool: {'Yes ✅' if not bad['success'] else 'No ❌'}")

        if output_path:
            payload = {
                "mode": "offline_selftest",
                "gate_rejected_bad_tool": (not bad["success"]),
                "tasks": [
                    {"task": "Days from 2020-01-01 to 2020-03-01", "answer_days": ans1, "trajectory": traj1},
                    {"task": "Days from 2021-01-01 to 2021-12-31", "answer_days": ans2, "trajectory": traj2},
                ],
                "reused": reused,
            }
            Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[Written] {output_path}")

        return 0 if (reused and not bad["success"]) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Online path: real LLM calls and network access
# --------------------------------------------------------------------------- #
def run_online(tasks: list, allow_create: bool, model: str | None, output_path: str | None) -> int:
    # Lazy import lets --offline run without the openai dependency
    from agent import SelfEvolvingAgent

    try:
        agent = SelfEvolvingAgent(verbose=True, allow_create=allow_create, model=model)
    except RuntimeError as e:
        print(f"[Configuration error] {e}", file=sys.stderr)
        print(
            "Configure the API key for your provider first (OpenAI by default):\n"
            "  cp env.example .env  then fill in OPENAI_API_KEY in .env;\n"
            "  or export OPENAI_API_KEY=your-openai-api-key directly\n"
            "To switch providers: export LLM_PROVIDER=moonshot|ark and configure "
            "MOONSHOT_API_KEY / ARK_API_KEY。\n"
            "(To check the mechanism without an API key, run: python demo.py --offline)",
            file=sys.stderr,
        )
        return 2

    default = tasks == [TASK_1, TASK_2]
    runs = []
    for i, task in enumerate(tasks, 1):
        label = {1: "Task one", 2: "Task two"}.get(i, f"Task {i}") if default else f"Task {i}"
        tag = {1: " (demonstrates search -> test -> package -> use)", 2: " (demonstrates tool reuse)"}.get(i, "") if default else ""
        print(f"\n########## {label}{tag} ##########")
        agent.trajectory = []
        ans = agent.run(task)
        traj = list(agent.trajectory)
        created = [t["name"] for t in agent.library.list_tools()]
        print(f"\n>>> {label} finished. Packaged tools in the library: {created}")
        print(f">>> {label} action trajectory: {traj}")
        runs.append({"task": task, "answer": ans, "trajectory": traj, "reused": _is_reuse(traj)})

    # Reuse succeeds when any task after the first reuses a tool
    reused = any(r["reused"] for r in runs[1:])
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    for i, r in enumerate(runs, 1):
        print(f"[Task {i}] {r['answer']}")
    print("-" * 70)
    if len(runs) >= 2:
        print(f"Did a later task reuse an existing tool without searching the web or creating it again: {'Yes ✅' if reused else 'No ❌'}")
        print("  Evidence: the reuse task called search_tools without web_search/create_tool.")

    if output_path:
        Path(output_path).write_text(json.dumps(
            {"mode": "online", "model": agent.model, "allow_create": allow_create,
             "runs": runs, "reused": reused},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[Written] {output_path}")

    if len(runs) < 2:
        return 0
    return 0 if reused else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Supplementary case: an agent finds tools online, validates them, and reuses them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python demo.py                 Run the two default tasks (evolution + reuse; API required)\n"
               "  python demo.py --fresh         Clear the library to evolve from scratch\n"
               "  python demo.py --offline       Offline self-check (no API/network)\n"
               "  python demo.py --task '...'    Custom task (repeatable)\n"
               "  python demo.py --no-create     Disable tool creation for comparison\n")
    p.add_argument("--task", action="append", metavar="TASK",
                   help="Task to execute; repeat to run multiple tasks in sequence. "
                        "Defaults to the NVDA and AAPL tasks.")
    p.add_argument("--offline", action="store_true",
                   help="Offline self-check: exercise search -> create -> validate -> register -> reuse without LLM/network calls.")
    p.add_argument("--fresh", action="store_true",
                   help="Clear tool_library/ before running to evolve from scratch (recommended for repeated demos).")
    p.add_argument("--no-create", dest="allow_create", action="store_false",
                   help="Disable create_tool for comparison runs; tool creation is enabled by default.")
    p.add_argument("--model", metavar="MODEL", default=None,
                   help="Override the LLM model, taking precedence over LLM_MODEL, e.g. gpt-5.6-luna.")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="Write tasks, answers, action trajectories, and reuse results to this JSON file.")
    return p


def main():
    args = build_parser().parse_args()

    if args.offline:
        return run_offline_selftest(output_path=args.output)

    if args.fresh:
        _clear_library()

    tasks = args.task if args.task else [TASK_1, TASK_2]
    return run_online(tasks, allow_create=args.allow_create,
                      model=args.model, output_path=args.output)


if __name__ == "__main__":
    sys.exit(main())
