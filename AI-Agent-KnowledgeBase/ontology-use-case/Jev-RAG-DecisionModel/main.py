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

FAKE_TREE_PATH = HERE / "out" / "keyword_tree.fake.json"
TREE_MISSING = "keyword tree not found at {path}; run: python main.py build-tree"
TREE_STALE = "keyword tree is stale (keywords.csv changed); run: python main.py build-tree"

from jev import JevError  # noqa: E402
from keyword_corpus import KEYWORD_TREE_PATH, KEYWORDS_CSV  # noqa: E402
from judge import FakeJudge  # noqa: E402
from navigate import NavigationConfig, navigate  # noqa: E402
from tree import Node  # noqa: E402
from tree_builder import DEFAULT_LLM_MODEL  # noqa: E402


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


def check_keyword_tree(path):
    """Load the built keyword tree; (meta, root), or None if missing. Warns on stale."""
    from keyword_corpus import is_stale, load_keyword_tree
    try:
        meta, root = load_keyword_tree(path)
    except FileNotFoundError:
        return None
    if is_stale(meta, KEYWORDS_CSV):
        print(TREE_STALE, file=sys.stderr)
    return meta, root


def cmd_tag_keywords(args, factory) -> int:
    from keyword_corpus import keyword_config
    from taxonomy_corpus import tag_room
    loaded = check_keyword_tree(KEYWORD_TREE_PATH)
    if loaded is None:
        print(TREE_MISSING.format(path=KEYWORD_TREE_PATH), file=sys.stderr)
        return 2
    _, root = loaded
    from tree import walk
    leaves = {n.id: n for n in walk(root) if n.is_leaf}
    judge = factory()  # default 50 per request: 415 keywords in one request is too big
    res = tag_room(root, judge, keyword_config(use_choice=not args.no_choice), args.subject,
                   args.tags, self_declared=args.declared, mode=args.mode)
    print("keyword [concept]  p_bin  p_mul  source")
    for r in res.ranked:
        leaf = leaves.get(r.id)
        label = f"{leaf.description} [{leaf.payload.get('concept_id', '')}]" if leaf else r.id
        print(f"{label}  {fmt(r.p_bin)}  {fmt(r.p_mul)}  {r.source}")
    print_trace(res.trace)
    print_usage(judge)
    return 0


def cmd_tag(args, factory) -> int:
    if args.tree == "keywords":
        return cmd_tag_keywords(args, factory)
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


def keyword_tree_runs(factory):
    """Yield the keyword-tree runs one at a time, so callers keep finished ones on error."""
    from evaluate import run_keyword_tree
    from keyword_corpus import keyword_config
    if check_keyword_tree(KEYWORD_TREE_PATH) is None:
        print(f"skipping keyword-tree runs: {KEYWORD_TREE_PATH} not found")
        return
    config = keyword_config()
    yield run_keyword_tree(factory(), config, mode="navigate",
                           name="decisionrag-keyword-tree-navigate", tree_path=KEYWORD_TREE_PATH)
    # flat keeps the default 50 units per request (no FLAT_MAX_PER_REQUEST override)
    yield run_keyword_tree(factory(), config, mode="flat",
                           name="decisionrag-keyword-flat", tree_path=KEYWORD_TREE_PATH)


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
        if args.mode == "all":
            for run in keyword_tree_runs(factory):
                runs.append(run)
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


def _fake_embed(keywords):
    return [[float(len(k)), float(ord(k[0]))] for k in keywords]


def _fake_kmeans(vectors, k, seed):
    return [i % k for i in range(len(vectors))]


def _fake_describe(keywords):
    return "Keywords about " + keywords[0]


def _no_describer(keywords):
    from tree_builder import DescriberUnavailable
    raise DescriberUnavailable("no OPENAI_API_KEY")


def _leaf_count(node: dict) -> int:
    return 1 if "keyword" in node else sum(_leaf_count(c) for c in node["children"])


def cmd_build_tree(args, factory=None) -> int:
    import csv
    import datetime
    import hashlib
    import io
    import tree_builder as tb
    describer = None
    if args.fake:
        embed, kmeans, describe = _fake_embed, _fake_kmeans, _fake_describe
        embed_model, llm_model, status = "fake", "none", "fake"
        kmeans_backend, n_init = "fake", None
    else:
        embed, kmeans = tb.E5Embedder(), tb.sklearn_kmeans
        embed_model = embed.model
        kmeans_backend, n_init = "sklearn.KMeans", tb.KMEANS_N_INIT
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            describer = describe = tb.OpenAIDescriber(key, args.llm_model)
            llm_model, status = args.llm_model, "ok"
        else:
            print("OPENAI_API_KEY not set: using keyword-list descriptions for every node")
            describe, llm_model, status = _no_describer, "none", "no_key"
    out = Path(args.out) if args.out else (FAKE_TREE_PATH if args.fake else KEYWORD_TREE_PATH)
    raw = Path(KEYWORDS_CSV).read_bytes()  # hash the raw bytes; parse with csv's own newlines
    rows = tb.load_keyword_rows(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    tree = tb.build_keyword_tree(rows, embed, kmeans, describe, clusters=args.clusters,
                                 leaf_size=args.leaf_size,
                                 csv_sha256=hashlib.sha256(raw).hexdigest())
    meta = tree["meta"]
    c = meta["counts"]
    llm_described = c["clusters"] + c["subclusters"] - c["fallback_descriptions"]
    if describer is not None:
        if describer.disabled:
            status = "rejected" if describer.disabled_code in (401, 403) else "unavailable"
        if llm_described == 0:
            llm_model = "none"  # no description came from the LLM
    meta.update(embed_model=embed_model, llm_model=llm_model, describer_status=status,
                kmeans_backend=kmeans_backend, kmeans_n_init=n_init,
                describe_prompt=tb.DESCRIBE_PROMPT,
                built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    tb.write_tree(tree, out)
    print(f"{c['keywords']} keywords -> {c['clusters']} clusters, {c['subclusters']} sub-clusters")
    print("level-1 sizes: " + ", ".join(str(_leaf_count(n)) for n in tree["tree"]["children"]))
    print(f"fallback descriptions: {c['fallback_descriptions']}")
    if describer is not None and llm_described == 0:
        if status == "rejected":
            print("OpenAI key rejected (401/403): all descriptions are keyword-list fallbacks")
        elif status == "unavailable":
            print(f"OpenAI describer unavailable (HTTP {describer.disabled_code}): "
                  "all descriptions are keyword-list fallbacks")
        else:
            print("OpenAI describer failed for every node: all descriptions are keyword-list "
                  "fallbacks")
    elif describer is not None:
        print(f"OpenAI tokens: {describer.prompt_tokens} prompt, "
              f"{describer.completion_tokens} completion")
    print(f"wrote {out}")
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
    t.add_argument("--tree", choices=["taxonomy", "keywords"], default="taxonomy")
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

    b = sub.add_parser("build-tree", help="build the keyword tree from keywords.csv (no Jev key)")
    b.add_argument("--clusters", type=int, default=20)
    b.add_argument("--leaf-size", type=int, default=5)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    common(b)
    b.set_defaults(func=cmd_build_tree)
    return p


def main(argv: list[str] | None = None, env_paths: list[Path] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if env_paths is None:
        env_paths = [HERE.parents[2] / ".env", HERE / ".env"]  # repo root, then local
    for path in env_paths:
        load_env(path)
    if args.command == "build-tree":
        return args.func(args)  # offline build: no Jev key needed
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
