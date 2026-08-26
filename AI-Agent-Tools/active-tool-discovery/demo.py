"""Active tool discovery: retrieval prefilter vs full injection.
Run the same cross-domain tasks against a catalog of 126 tools using three
strategies, then compare accuracy, injected tokens, and latency:

- full_injection: inject all 126 schemas at once (the book's control).
- retrieval_prefilter: one-shot semantic retrieval on the initial query.
- active_discovery: base tools plus discover_tools, loaded on demand.

Usage (see --help):
    python demo.py                         # all tasks and strategies; requires a key
    python demo.py --offline               # local embeddings + mock model; no key
    python demo.py --tasks finance+news,crypto+news
    python demo.py --strategies full,discovery --tool-set-size 30
    python demo.py --query "Check NVIDIA stock and find related news" --offline
    python demo.py --offline --output results/offline.json
"""

import argparse
import json
import os
import sys
import time

from tools_library import TASKS, grade, select_tools


def _to_openrouter_model(model: str) -> str:
    """Map common model names to OpenRouter names when OpenAI is unavailable."""
    if not model:
        return "openai/gpt-5.6-luna"
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return "openai/" + model
    if model.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    return "openai/gpt-5.6-luna"


# Strategy registry: key -> (display name, requires index)
STRATEGIES = {
    "full": ("Full injection", False),
    "prefilter": ("Retrieval prefilter", True),
    "discovery": ("Active discovery", True),
}
STRATEGY_ORDER = ["full", "prefilter", "discovery"]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_strategies(value: str):
    strategies = [item.strip() for item in value.split(",") if item.strip()]
    if not strategies:
        raise ValueError("at least one strategy is required")
    bad = [item for item in strategies if item not in STRATEGIES]
    if bad:
        raise ValueError(f"unknown strategies: {bad}; choices: {list(STRATEGIES)}")
    return sorted(strategies, key=STRATEGY_ORDER.index)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="demo.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Active tool discovery vs retrieval prefilter vs full injection.\n"
                    "Compare accuracy, injected tokens, and latency across tasks over a 126-tool catalog.",
        epilog="Examples:\n"
               "  python demo.py --offline                 # offline mechanism check; no key\n"
               "  python demo.py --strategies full,discovery --tasks finance+news\n"
               "  python demo.py --query 'Check NVIDIA stock and related news' --offline\n")
    ap.add_argument("--query", metavar="TEXT",
                    help="Run one natural-language request instead of the built-in task set.")
    ap.add_argument("--tasks", metavar="IDS",
                    help="Comma-separated built-in task IDs; defaults to all eight. "
                         "Quote IDs containing parentheses, such as 'opinion(inducement)'.")
    ap.add_argument("--strategies", metavar="LIST", default="full,prefilter,discovery",
                    help="Comma-separated strategies: full/prefilter/discovery; defaults to all three.")
    ap.add_argument("--tool-set-size", type=_positive_int, default=None, metavar="N",
                    help="Limit the catalog to N tools while retaining base, generic, and task tools.")
    ap.add_argument("--top-k", type=_positive_int, default=4, metavar="K",
                    help="Candidates returned by each discover_tools call (default: 4).")
    ap.add_argument("--prefilter-n", type=_positive_int, default=10, metavar="N",
                    help="Candidates injected by one-shot retrieval prefiltering (default: 10).")
    ap.add_argument("--model", default=os.getenv("MODEL", "gpt-5.6-luna"), metavar="NAME",
                    help="Chat model (MODEL or gpt-5.6-luna by default); ignored offline.")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                    metavar="NAME", help="Embedding model (default: text-embedding-3-small); ignored offline.")
    ap.add_argument("--max-steps", type=_positive_int, default=10, metavar="N",
                    help="Maximum ReAct steps per task (default: 10).")
    ap.add_argument("--offline", action="store_true",
                    help="Offline mechanism check using local hash embeddings and a scripted mock; no API key.")
    ap.add_argument("--output", metavar="PATH",
                    help="Write structured per-task, per-strategy results to this JSON file.")
    return ap


def _fmt_grade(g):
    tag = "✅ Exact match" if g["precise"] else ("⚠️ Complete with wrong pick" if g["correct"] else "❌ Failed")
    detail = f"{g['filled_slots']}/{g['total_slots']} capability slots filled"
    extra = ""
    if g["missed_slots"]:
        extra += f" | missing: {[s[0] for s in g['missed_slots']]}"
    if g["used_generic_substitute"]:
        extra += f" | generic tools misused: {g['used_generic_substitute']}"
    return f"{tag} ({detail}{extra})"


def _make_task_from_query(query: str):
    """Wrap a one-off --query as a scored task, inferring slots by keyword."""
    from offline_backend import match_intents
    slots = [[tool] for tool, _ in match_intents(query)]
    if not slots:
        raise ValueError("could not infer any capability for the ad-hoc query")
    return {"id": "adhoc", "prompt": query, "required_slots": slots}


def run_strategy(key, client, model, prompt, index, tools, args):
    """Run one strategy and return (result_dict, latency_s)."""
    from agent import (run_active_discovery, run_full_injection,
                       run_retrieval_prefilter)
    t0 = time.perf_counter()
    if key == "full":
        res = run_full_injection(client, model, prompt, tools=tools, max_steps=args.max_steps)
    elif key == "prefilter":
        res = run_retrieval_prefilter(client, model, prompt, index,
                                      top_n=args.prefilter_n, tools=tools, max_steps=args.max_steps)
    else:
        res = run_active_discovery(client, model, prompt, index,
                                   top_k=args.top_k, tools=tools, max_steps=args.max_steps)
    return res, time.perf_counter() - t0


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        strategies = _parse_strategies(args.strategies)
    except ValueError as exc:
        parser.error(str(exc))

    # ---- Tasks ----
    if args.query:
        try:
            tasks = [_make_task_from_query(args.query)]
        except ValueError as exc:
            parser.error(str(exc))
    else:
        tasks = TASKS
        if args.tasks:
            want = set(args.tasks.split(","))
            tasks = [t for t in TASKS if t["id"] in want]
        if not tasks:
            print(f"No matching task IDs: {args.tasks}")
            sys.exit(2)

    tools = select_tools(args.tool_set_size, tasks)
    need_index = any(STRATEGIES[s][1] for s in strategies)

    # ---- Backend (online OpenAI / offline mock) ----
    if args.offline:
        from offline_backend import LocalEmbedder, MockChatClient
        from discovery import ToolIndex
        client = MockChatClient()
        model = "mock-offline"
        embedder = LocalEmbedder()
        print("=" * 92)
        print("Offline mechanism check: local hash embeddings + scripted mock model (no API key).")
        print("  · Tokens and latency are measured; accuracy reflects heuristic routing only.")
        print("  · Compare token use and the structural misses of one-shot prefiltering.")
        print("=" * 92)
    else:
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError:
            print("Missing openai/python-dotenv. Run pip install -r requirements.txt or use --offline.")
            sys.exit(1)
        load_dotenv()
        from discovery import OpenAIEmbedder, ToolIndex
        if os.getenv("OPENAI_API_KEY"):
            # Direct OpenAI: both chat and embeddings use OpenAI.
            client = OpenAI()
            model = args.model
            embedder = OpenAIEmbedder(client, model=args.embed_model)
        elif os.getenv("OPENROUTER_API_KEY"):
            # OpenRouter proxies chat completions but has no embeddings API, so
            # use OpenRouter for chat and local hash embeddings for retrieval.
            from offline_backend import LocalEmbedder
            client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1")
            model = _to_openrouter_model(args.model)
            embedder = LocalEmbedder()
            print("OPENAI_API_KEY not found; using the OpenRouter fallback:")
            print(f"  · Chat model: {model} (live call)")
            print("  · Retrieval: local hash embeddings (OpenRouter has no embeddings API).")
        else:
            print("Set OPENAI_API_KEY or OPENROUTER_API_KEY (see env.example), or use --offline.")
            sys.exit(1)

    index = ToolIndex(embedder, tools=tools) if need_index else None

    print(f"Model: {model}  |  Embeddings: {embedder.name}  |  Catalog: {len(tools)} tools  "
          f"|  Tasks: {len(tasks)}  |  Strategies: {[STRATEGIES[s][0] for s in strategies]}\n")

    # ---- Run each task ----
    records = []           # Each: {task, strategy, result, grade, latency}
    for task in tasks:
        print("=" * 92)
        print(f"Task [{task['id']}]: {task['prompt']}")
        print("-" * 92)
        for key in strategies:
            res, latency = run_strategy(key, client, model, task["prompt"], index, tools, args)
            g = grade(
                task, res["called"], finished=res["finished"],
                successful_tools=res["successful"],
            )
            records.append({"task": task["id"], "strategy": key, "result": res,
                            "grade": g, "latency_s": round(latency, 3)})
            cname = STRATEGIES[key][0]
            print(f"[{cname}] introduced {res['injected_tokens']:>6} schema tokens "
                  f"({res['num_tools_exposed']} tools exposed)  latency {latency:5.2f}s")
            if key == "prefilter":
                print(f"           Prefiltered: {res['prefiltered']}")
            if key == "discovery":
                for line in res["trace"]:
                    if line.startswith("[discover_tools]"):
                        print(f"           {line}")
                print(f"           Discovered and loaded: {res['discovered']}")
            print(f"           Call trace: {res['called']}")
            print(f"           Grade: {_fmt_grade(g)}")
        print()

    _print_summary(tasks, strategies, records)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        payload = {"model": model, "embedder": embedder.name, "tool_set_size": len(tools),
                   "offline": args.offline, "strategies": strategies,
                   "records": records}
        json.dump(payload, open(args.output, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\nStructured results written to: {args.output}")


def _print_summary(tasks, strategies, records):
    n = len(tasks)
    print("=" * 92)
    print("Summary (exact match = all capability slots filled with no generic fallback misuse)")
    print("=" * 92)
    header = (f"{'Strategy':<20}{'Exact':>10}{'Complete':>10}"
              f"{'Avg schema tok':>16}{'Total schema tok':>18}{'Avg latency(s)':>16}")
    print(header)
    print("-" * 92)
    for key in strategies:
        rs = [r for r in records if r["strategy"] == key]
        precise = sum(int(r["grade"]["precise"]) for r in rs)
        correct = sum(int(r["grade"]["correct"]) for r in rs)
        tok = [r["result"]["injected_tokens"] for r in rs]
        lat = [r["latency_s"] for r in rs]
        avg_tok = sum(tok) / len(tok) if tok else 0
        avg_lat = sum(lat) / len(lat) if lat else 0
        print(f"{STRATEGIES[key][0]:<12}{f'{precise}/{n}':>10}{f'{correct}/{n}':>10}"
              f"{avg_tok:>16.0f}{sum(tok):>18}{avg_lat:>12.3f}")
    print("-" * 92)
    if "full" in strategies and "discovery" in strategies:
        ft = sum(r["result"]["injected_tokens"] for r in records if r["strategy"] == "full")
        at = sum(r["result"]["injected_tokens"] for r in records if r["strategy"] == "discovery")
        if at:
            print(f"Schema tokens introduced: full {ft} vs active discovery {at}; "
                  f"about {ft/at:.1f}x fewer per task.")


if __name__ == "__main__":
    main()
