"""CLI for the DecisionRAG prototype: tag a room, evaluate, or run the Nolan demo."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache" / "jev"
DEFAULT_OUTPUT = HERE / "out" / "report.json"
DEMO_CORPUS = HERE / "examples" / "nolan_corpus.json"
DEMO_QUERY = "Who is the main character in Christopher Nolan's film from the year 2000?"
DEMO_QUESTION = "Is the information needed to answer the user's question in Candidate {n}?"
NO_KEY = "No Jev API key: set TYPESAFE_API_KEY or OPENROUTER_API_KEY (or put it in .env), or pass --fake"
FLAT_MAX_PER_REQUEST = 10_000  # flat mode judges every unit in one shared context

from jev import JevError  # noqa: E402
from judge import FakeJudge  # noqa: E402
from navigate import NavigationConfig, navigate  # noqa: E402
from tree import Node  # noqa: E402


def load_env(path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ without overriding existing variables."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def make_judge_factory(fake: bool):
    """Return (factory, transport_or_None); factory(**judge_kwargs) builds a fresh judge.
    None if key missing."""
    if fake:
        return (lambda **kw: FakeJudge()), None
    from jev import PROVIDERS, CachedTransport, HttpTransport, JevJudge
    provider = next((p for p in PROVIDERS if os.environ.get(p.env_key)), None)
    if provider is None:
        return None, None
    http = HttpTransport(os.environ[provider.env_key], endpoint=provider.endpoint)
    transport = CachedTransport(http, CACHE_DIR / provider.name)

    def factory(**kw):
        return JevJudge(transport, model=provider.model, provider=provider.name, **kw)

    return factory, transport


def judge_kwargs(mode: str) -> dict:
    return {"max_per_request": FLAT_MAX_PER_REQUEST} if mode == "flat" else {}


def print_usage(judge) -> None:
    if hasattr(judge, "cost_usd"):
        label = "estimated cost" if judge.cost_estimated else "nominal cost"
        models = ", ".join(sorted(judge.models_seen)) or "unknown"
        print(f"provider: {judge.provider or 'unknown'}  model: {judge.model} (resolved: {models})")
        print(f"{label}: ${judge.cost_usd:.4f} (cached responses included; real spend = "
              f"cache misses)  requests: {judge.requests}")


def print_trace(trace: list[dict]) -> None:
    for i, st in enumerate(trace, 1):
        if "choice" in st:
            print(f"stage {i}: choice over {st['choice']}")
            print("  P(mul): " + ", ".join(f"{k}={v:.3f}" for k, v in st["p"].items()))
        else:
            print(f"stage {i}: judged {len(st['judged'])} candidates")
            print("  P(yes): " + ", ".join(f"{k}={v:.3f}" for k, v in st["p"].items()))
            print(f"  kept:   {st['kept']}")
            print(f"  pruned: {st['pruned']}")
        print(f"  ({st['requests']} requests, {st['latency_s']:.2f}s)")


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def cmd_tag(args, factory) -> int:
    from taxonomy_corpus import load_tree, tag_room, taxonomy_config
    _, root = load_tree()
    config = taxonomy_config(use_choice=not args.no_choice)
    judge = factory(**judge_kwargs(args.mode))
    res = tag_room(root, judge, config, args.subject, args.tags,
                   self_declared=args.declared, mode=args.mode)
    print("id  p_bin  p_mul  source")
    for r in res.ranked:
        print(f"{r.id}  {fmt(r.p_bin)}  {fmt(r.p_mul)}  {r.source}")
    print_trace(res.trace)
    print_usage(judge)
    return 0


def load_demo_tree(path: Path = DEMO_CORPUS) -> Node:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = tuple(
        Node(d["id"], d["title"],
             tuple(Node(p["id"], p["text"]) for p in d["passages"]))
        for d in data["documents"])
    return Node("__root__", "corpus", docs)


def cmd_demo(args, factory) -> int:
    root = load_demo_tree()
    config = NavigationConfig(k=2, tag_threshold=0.0, r=5, m=5, binary_question=DEMO_QUESTION,
                              choice_question="Which candidate contains the information needed "
                                              "to answer the user's question?")
    print(f"Query: {DEMO_QUERY}\n")
    judge = factory()
    res = navigate(root, DEMO_QUERY, judge, config)
    print_trace(res.trace)
    print("\nfinal passage ranking:")
    texts = {n.id: n.description for d in root.children for n in d.children}
    for i, r in enumerate(res.ranked, 1):
        print(f"{i}. {r.id}  p_bin={fmt(r.p_bin)}  p_mul={fmt(r.p_mul)}  {texts[r.id]}")
    print_usage(judge)
    return 0


def cmd_eval(args, factory) -> int:
    from evaluate import run_baselines, run_decisionrag, write_report
    from taxonomy_corpus import taxonomy_config
    if args.mode == "all":
        plan = [("navigate", True, None), ("navigate", False, "decisionrag-navigate-no-choice"),
                ("flat", True, None), ("separate", True, None)]
    else:
        plan = [(args.mode, not args.no_choice, None)]
    runs = []
    try:
        for mode, choice, name in plan:
            runs.append(run_decisionrag(factory(**judge_kwargs(mode)),
                                        taxonomy_config(use_choice=choice), mode=mode, name=name))
    except JevError:
        write_report(runs, args.output)  # keep the runs that finished
        print(f"wrote partial report ({len(runs)} runs) to {args.output}", file=sys.stderr)
        raise
    if args.mode == "all":
        runs += run_baselines()
    write_report(runs, args.output)
    for r in runs:
        c = r["report"]["concepts"]
        kind = "est. cost" if r.get("cost_estimated") else "cost"
        print(f"{r['name']}: micro-F1={fmt(c['micro_f1'])} {kind}=${r['cost_usd']:.4f} "
              f"requests={r['requests']}")
    print(f"wrote {args.output} and {Path(args.output).with_suffix('.md')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--fake", action="store_true", help="use the offline FakeJudge")

    t = sub.add_parser("tag", help="tag one room")
    t.add_argument("subject")
    t.add_argument("--tags", nargs="*", default=[])
    t.add_argument("--declared", nargs="*", default=[])
    t.add_argument("--mode", choices=["navigate", "flat", "separate"], default="navigate")
    t.add_argument("--no-choice", action="store_true")
    common(t)
    t.set_defaults(func=cmd_tag)

    d = sub.add_parser("demo", help="Nolan running example on a document/passage tree")
    common(d)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("eval", help="evaluate on the labeled test rooms")
    e.add_argument("--mode", choices=["navigate", "flat", "separate", "all"], default="navigate")
    e.add_argument("--no-choice", action="store_true")
    e.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common(e)
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None, env_paths: list[Path] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if env_paths is None:
        env_paths = [HERE.parents[2] / ".env", HERE / ".env"]  # repo root, then local
    for path in env_paths:
        load_env(path)
    factory, transport = make_judge_factory(args.fake)
    if factory is None:
        print(NO_KEY, file=sys.stderr)
        return 2
    try:
        code = args.func(args, factory)
    except JevError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 1
    if transport is not None:
        print(f"cache: {transport.hits} hits, {transport.misses} misses ({transport.cache_dir})")
    return code


if __name__ == "__main__":
    sys.exit(main())
