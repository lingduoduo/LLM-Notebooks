"""End-to-end agent cost-analysis demo and CLI.

Live mode calls a model and records actual usage. Offline mode loads a captured
trace and recalculates costs with configurable prices. Both modes report a
per-step breakdown and a complete 2x2 comparison of KV caching and compression.
"""

import argparse
import json
import os
import sys

import config
from config import PRICING_PRESETS, Pricing


DEFAULT_TRACE = os.path.join(os.path.dirname(__file__), "sample_trace.json")
SCENARIO_KEYS = ["naive", "kv", "compress", "both"]


def _pct(saved: float, base: float) -> str:
    if base == 0:
        return "0.0%"
    return f"{saved / base * 100:.1f}%"


def build_pricing(args) -> Pricing:
    """Build pricing from the model preset and optional price overrides."""
    base = PRICING_PRESETS.get(args.model)
    if base is None:
        base = config.default_pricing()
    return Pricing(
        input_per_m=args.price_input if args.price_input is not None else base.input_per_m,
        cached_per_m=args.price_cached if args.price_cached is not None else base.cached_per_m,
        output_per_m=args.price_output if args.price_output is not None else base.output_per_m,
    )


def resolve_scenarios(arg: str):
    """Parse --scenario into an ordered, deduplicated list of keys."""
    if arg == "all":
        return list(SCENARIO_KEYS)
    if arg == "ab":
        return ["naive", "both"]
    keys, seen = [], set()
    for k in arg.split(","):
        k = k.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


# ---------------------------------------------------------------------------
# Collect data from a live model or an offline trace.
# ---------------------------------------------------------------------------
def collect_live(keys, pricing, warmup: bool):
    import agent

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set. Export one "
              "of them, or use --offline to recalculate without an API key.",
              file=sys.stderr)
        sys.exit(1)

    # Build the client and resolve the model name (possibly an OpenRouter ID).
    client, resolved = config.make_client_and_model(config.MODEL)
    if resolved != config.MODEL:
        print(f">>> Using OpenRouter fallback: {config.MODEL} -> {resolved}")
        config.MODEL = resolved
        agent.MODEL = resolved
        try:
            agent._encoder.cache_clear()
        except Exception:
            pass
    tracers = []
    for k in keys:
        name, kv, compress = agent.SCENARIOS[k]
        # Warm the stable prefix before measuring a cache-enabled scenario.
        if kv and warmup:
            print(f">>> Warming the stable prefix for [{name}]...")
            agent.run_scenario(client, kv, compress, name=name, pricing=pricing)
        print(f">>> Running [{name}] {'(live measurement)' if kv else ''}...")
        tr = agent.run_scenario(client, kv, compress, name=name, pricing=pricing)
        tracers.append((k, tr))
    return tracers


def collect_offline(keys, pricing, trace_path):
    from tracer import Tracer

    if not os.path.exists(trace_path):
        print(f"Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)
    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_key = {s.get("key", s.get("name")): s for s in data.get("scenarios", [])}
    print(f"Offline mode: reading {trace_path}")
    print(f"  Trace model = {data.get('model', '?')}; {len(by_key)} captured scenarios. "
          "Token counts are observed; costs use the current pricing.")

    tracers = []
    for k in keys:
        sc = by_key.get(k)
        if sc is None:
            print(f"  [skip] Scenario '{k}' is not present in the trace; capture it with --save-trace.",
                  file=sys.stderr)
            continue
        spans = sc.get("spans")
        if not spans:
            print(f"  [skip] Scenario '{k}' has no span data; capture it with --save-trace.", file=sys.stderr)
            continue
        tr = Tracer.from_records(spans, name=sc.get("name", k), pricing=pricing)
        tracers.append((k, tr))
    if not tracers:
        print("The trace contains none of the selected scenarios.", file=sys.stderr)
        sys.exit(1)
    return tracers


# ---------------------------------------------------------------------------
# A/B comparison table.
# ---------------------------------------------------------------------------
def print_ab_table(tracers):
    print("\n\n===== A/B cost comparison (same eight-turn refund task) =====")
    header = (f"{'Scenario':<46} {'Input tok':>10} {'Cached':>10} {'Cache %':>8} "
              f"{'Output':>8} {'Cost ($)':>12} {'vs base':>10}")
    print(header)
    print("-" * len(header))

    base_cost = tracers[0][1].total_cost()
    for _, tr in tracers:
        pin = tr.total_prompt_tokens()
        cac = tr.total_cached_tokens()
        rate = f"{(cac / pin * 100):.1f}%" if pin else "0.0%"
        cost = tr.total_cost()
        vs = "baseline" if abs(cost - base_cost) < 1e-12 else f"-{_pct(base_cost - cost, base_cost)}"
        print(f"{tr.name:<46} {pin:>10} {cac:>10} {rate:>8} "
              f"{tr.total_completion_tokens():>8} {cost:>12.6f} {vs:>10}")
    print("-" * len(header))

    # Highlight the first (baseline) and last (usually fully optimized) scenarios.
    base_k, base = tracers[0]
    best_k, best = tracers[-1]
    if base_k != best_k:
        tok_a = base.total_prompt_tokens() + base.total_completion_tokens()
        tok_b = best.total_prompt_tokens() + best.total_completion_tokens()
        cost_a, cost_b = base.total_cost(), best.total_cost()
        print(f"\nKey comparison: {base.name}  ->  {best.name}")
        print(f"  Total tokens: A={tok_a} -> B={tok_b}; saved {tok_a - tok_b} ({_pct(tok_a - tok_b, tok_a)})")
        print(f"  Cached tokens: A={base.total_cached_tokens()} -> B={best.total_cached_tokens()}")
        print(f"  Total cost: A=${cost_a:.6f} -> B=${cost_b:.6f}; saved ${cost_a - cost_b:.6f} ({_pct(cost_a - cost_b, cost_a)})")
        if cost_b > 0:
            print(f"  Cost ratio: A costs {cost_a / cost_b:.2f}x B")

    print("\nConclusion: a stable long prefix enables cached pricing, while context")
    print("compression limits history growth; together they reduce end-to-end cost.")


def dump_output(path, tracers, pricing, model):
    out = {
        "model": model,
        "pricing": {"input": pricing.input_per_m, "cached": pricing.cached_per_m,
                    "output": pricing.output_per_m},
        "scenarios": [],
    }
    import agent
    for k, tr in tracers:
        name = agent.SCENARIOS[k][0] if k in agent.SCENARIOS else tr.name
        out["scenarios"].append({
            "key": k, "name": name,
            "total_cost": tr.total_cost(),
            "component_costs": tr.component_costs(),
            "cost_distribution": tr.cost_distribution(),
            "spans": tr.to_records(),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote results to {path}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="demo.py",
        description="End-to-end cost analysis for a customer-refund agent, including a complete 2x2 comparison of KV caching and context compression.",
        epilog="Examples:\n"
               "  python demo.py                       # live: naive and optimized\n"
               "  python demo.py --scenario all        # live: complete 2x2\n"
               "  python demo.py --offline             # recalculate the bundled trace\n"
               "  python demo.py --offline --model gpt-4o   # use another price preset\n"
               "  python demo.py --live --save-trace out.json  # capture live usage\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="Live mode (default): call a real model; requires API credentials.")
    mode.add_argument("--offline", action="store_true",
                      help="Offline mode: recalculate captured token usage without calling a model.")
    p.add_argument("--trace", metavar="FILE", default=DEFAULT_TRACE,
                   help=f"Trace file for offline mode (default: {os.path.basename(DEFAULT_TRACE)}).")
    p.add_argument("--save-trace", metavar="FILE", default=None,
                   help="Save live token usage as a trace for later offline use.")
    p.add_argument("--scenario", metavar="NAME", default="ab",
                   help="Scenarios: ab (default), all, or a comma-separated subset of naive,kv,compress,both.")
    p.add_argument("--model", metavar="NAME", default=config.MODEL,
                   help=f"Model name used for the default price preset ({', '.join(PRICING_PRESETS)}); default: {config.MODEL}.")
    p.add_argument("--price-input", type=float, default=None,
                   help="Override input price in USD per million tokens.")
    p.add_argument("--price-cached", type=float, default=None,
                   help="Override cached-input price in USD per million tokens.")
    p.add_argument("--price-output", type=float, default=None,
                   help="Override output price in USD per million tokens.")
    p.add_argument("--no-warmup", action="store_true",
                   help="Disable stable-prefix warmup for live cache-enabled scenarios.")
    p.add_argument("--output", metavar="FILE", default=None,
                   help="Write cost components, distributions, and per-step usage to JSON.")
    return p


def main():
    args = build_parser().parse_args()

    # Configure the selected model for the agent and tokenizer.
    config.MODEL = args.model
    try:
        import agent
        agent.MODEL = args.model
        agent._encoder.cache_clear()
    except Exception:
        pass

    pricing = build_pricing(args)
    keys = resolve_scenarios(args.scenario)
    bad = [k for k in keys if k not in SCENARIO_KEYS]
    if bad:
        print(f"Unknown scenarios {bad}; choose from {SCENARIO_KEYS}, all, or ab.", file=sys.stderr)
        sys.exit(2)

    print(f"Model: {args.model}")
    print(f"Prices per million tokens: input ${pricing.input_per_m} / cached input ${pricing.cached_per_m} "
          f"/ output ${pricing.output_per_m}")

    if args.offline:
        tracers = collect_offline(keys, pricing, args.trace)
    else:
        print("Note: prompt caching is automatic for eligible repeated prefixes;")
        print("cached input appears in usage.prompt_tokens_details.cached_tokens.")
        tracers = collect_live(keys, pricing, warmup=not args.no_warmup)
        if args.save_trace:
            dump_output(args.save_trace, tracers, pricing, args.model)

    # Per-scenario cost breakdown.
    for _, tr in tracers:
        tr.print_breakdown(title=f"{tr.name} (single-task end-to-end breakdown)")

    # A/B comparison.
    if len(tracers) >= 2:
        print_ab_table(tracers)

    if args.output:
        dump_output(args.output, tracers, pricing, args.model)


if __name__ == "__main__":
    main()
