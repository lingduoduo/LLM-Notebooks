"""Standalone command line: run, build, query, and report."""

import argparse
import json
import os
from pathlib import Path
import sys

from .data import (
    load_dataset,
    normalize_observations,
    synthetic_dataset,
    walkthrough_dataset,
)
from .experiment import ANSWER_PROMPT, run_experiment
from .memory import Memory, RetrievalConfig
from .providers import (
    APIEncoder,
    ChatCompletion,
    JevDecisions,
    LexicalEncoder,
    LocalDecisions,
    OfflineCompletion,
)
from .report import render_report


def _providers(args):
    if args.mode == "offline":
        if args.embedding == "api" or args.decisions == "jev":
            raise ValueError(
                "Offline mode requires lexical embeddings and heuristic decisions"
            )
        return LexicalEncoder(), LocalDecisions(), OfflineCompletion(), None, {}
    # Choose each key with its own endpoint; do not rewrite global OPENAI_* settings.
    deepseek = bool(os.getenv("DEEPSEEK_API_KEY")) and not os.getenv("ANSWER_API_KEY")
    answer_key = os.getenv("ANSWER_API_KEY") or (
        os.getenv("DEEPSEEK_API_KEY") if deepseek else os.getenv("OPENAI_API_KEY")
    )
    answer_url = (
        args.answer_base_url
        or os.getenv("ANSWER_BASE_URL")
        or (
            "https://api.deepseek.com/v1"
            if deepseek
            else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    )
    answer_model = (
        args.answer_model
        or os.getenv("ANSWER_MODEL")
        or ("deepseek-chat" if deepseek else "gpt-4o-mini")
    )
    needs_answer = getattr(args, "command", "run") != "build"
    answer = (
        ChatCompletion(answer_model, answer_url, answer_key) if needs_answer else None
    )
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL") or answer_model
    judge = (
        ChatCompletion(
            judge_model,
            args.judge_base_url or os.getenv("JUDGE_BASE_URL") or answer_url,
            os.getenv("JUDGE_API_KEY") or answer_key,
        )
        if needs_answer and getattr(args, "command", "run") == "run"
        else None
    )
    if args.embedding == "lexical":
        encoder = LexicalEncoder()
    else:
        silicon = bool(os.getenv("SILICONFLOW_API_KEY")) and not os.getenv(
            "EMBEDDING_API_KEY"
        )
        embedding_key = os.getenv("EMBEDDING_API_KEY") or (
            os.getenv("SILICONFLOW_API_KEY") if silicon else os.getenv("OPENAI_API_KEY")
        )
        embedding_url = (
            args.embedding_base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or (
                "https://api.siliconflow.cn/v1"
                if silicon
                else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
        )
        embedding_model = (
            args.embedding_model
            or os.getenv("EMBEDDING_MODEL")
            or ("Qwen/Qwen3-Embedding-0.6B" if silicon else "text-embedding-3-small")
        )
        encoder = APIEncoder(embedding_model, embedding_url, embedding_key)
    decisions = (
        LocalDecisions()
        if args.decisions == "heuristic"
        else JevDecisions(
            model=os.getenv("JEV_MODEL", "jev-latest"),
            base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai"),
        )
    )
    return (
        encoder,
        decisions,
        answer,
        judge,
        {
            "answer_model": answer_model,
            "judge_model": judge_model,
            "embedding_model": getattr(encoder, "model", encoder.fingerprint),
            "decision_model": decisions.model,
        },
    )


def _dataset(args):
    if args.data:
        return [load_dataset(args.data, args.sample, args.max_questions)]
    datasets = []
    if args.experiment in ("walkthrough", "all"):
        datasets.append(walkthrough_dataset())
    if args.experiment in ("scaling", "all"):
        datasets.extend(synthetic_dataset(n) for n in args.scales)
    if args.max_questions:
        for data in datasets:
            data["questions"] = data["questions"][: args.max_questions]
    return datasets


def _parser():
    parser = argparse.ArgumentParser(
        description="Standalone conversational memory pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="Build, compare four arms, judge, and render reports"
    )
    run.add_argument(
        "--experiment", choices=("walkthrough", "scaling", "all"), default="walkthrough"
    )
    run.add_argument("--scales", type=int, nargs="+", default=[48, 96, 192, 384])
    run.add_argument(
        "--data", type=Path, help="Normalized JSON dataset or native LoCoMo JSON"
    )
    run.add_argument("--sample", type=int, default=0)
    run.add_argument("--max-questions", type=int)
    run.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts"
    )
    build = commands.add_parser(
        "build", help="Build and persist memory from JSON observations"
    )
    build.add_argument("--data", type=Path, required=True)
    build.add_argument("--store", type=Path, required=True)
    query = commands.add_parser(
        "query", help="Load memory, retrieve evidence, and answer"
    )
    query.add_argument("--store", type=Path, required=True)
    query.add_argument("question")
    report = commands.add_parser(
        "report", help="Regenerate an SVG report from a run summary"
    )
    report.add_argument("run_directory", type=Path)
    for command in (run, build, query):
        command.add_argument("--mode", choices=("offline", "live"), default="offline")
        command.add_argument("--embedding", choices=("lexical", "api"))
        command.add_argument("--decisions", choices=("heuristic", "jev"))
        for field in (
            "answer-model",
            "answer-base-url",
            "judge-model",
            "judge-base-url",
            "embedding-model",
            "embedding-base-url",
        ):
            command.add_argument("--" + field)
    for command in (run, build):
        command.add_argument("--max-nodes", type=int, default=15)
        command.add_argument("--max-depth", type=int, default=3)
        command.add_argument("--graph-budget", type=int, default=20)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    decisions = None
    try:
        if args.command == "report":
            print(render_report(args.run_directory))
            return 0
        if args.command == "run":
            if args.max_questions is not None and args.max_questions < 1:
                raise ValueError("max-questions must be positive")
            datasets = _dataset(
                args
            )  # Validate all datasets before constructing live providers.
        if args.command == "build":
            raw = json.loads(args.data.read_text(encoding="utf-8"))
            if isinstance(raw, list) and (
                not raw
                or isinstance(raw[0], str)
                or isinstance(raw[0], dict)
                and "conversation" not in raw[0]
            ):
                observations = normalize_observations(raw)
            elif isinstance(raw, dict) and "observations" in raw:
                observations = normalize_observations(raw["observations"])
            else:
                observations = load_dataset(args.data)["observations"]
            if not observations:
                raise ValueError("No observations to build")
            if (args.store / "memory.json").exists():
                raise ValueError(
                    "Store already exists; choose a fresh --store directory"
                )
        if hasattr(args, "max_nodes"):
            config = RetrievalConfig(
                anchor_count=min(5, args.max_nodes),
                max_nodes=args.max_nodes,
                max_depth=args.max_depth,
                graph_budget=args.graph_budget,
            )
        encoder, decisions, answer, judge, manifest = _providers(args)
        if args.command == "run":
            failures = False
            for dataset in datasets:
                run = run_experiment(
                    dataset,
                    args.output,
                    mode=args.mode,
                    answerer=answer,
                    judge=judge,
                    encoder=encoder,
                    decisions=decisions,
                    config=config,
                    manifest=manifest,
                )
                summary = json.loads((run / "summary.json").read_text())
                failures = failures or any(
                    v["errors"] for v in summary["arms"].values()
                )
                print(
                    json.dumps(
                        {
                            "dataset": dataset["name"],
                            "run": str(run),
                            "mode": args.mode,
                            "scores": {
                                a: v["mean_score"] for a, v in summary["arms"].items()
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            return 1 if failures else 0
        if args.command == "build":
            memory = Memory(encoder, decisions, config, args.store / "decisions.jsonl")
            result = memory.build(observations)
            memory.save(args.store)
            print(json.dumps({"store": str(args.store), **result}))
            return 0
        memory = Memory.load(
            args.store,
            encoder=encoder,
            decisions=decisions,
            audit_path=args.store / "queries.jsonl",
        )
        result = memory.retrieve(args.question)
        completion = answer.complete(
            ANSWER_PROMPT.format(context=result["context"], question=args.question)
        )
        print(
            json.dumps(
                {
                    "answer": completion.text,
                    "usage": completion.usage,
                    "mode": args.mode,
                    **result,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        # Provider adapters sanitize exceptions before they reach here.
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if decisions is not None and hasattr(decisions, "close"):
            decisions.close()
