"""Unified Agentic RAG demo — all five patterns collaborating in one crew.

Patterns combined:
  1. Hierarchical task planning   — manager LLM orchestrates all agents dynamically
  2. Dynamic tool selection       — ToolIndex pre-selects top-k tools before crew build
  3. Corrective RAG               — retrieve → evaluate → rewrite → re-retrieve
  4. Object-level tool indexing   — 9 tools embedded; only relevant subset given to agents
  5. Heterogeneous data fusion    — structured, stream, and graph specialists feed a coordinator

All agents share a single LLM and knowledge source (SharedResources), created once.

Usage:
  python unified_agent.py
  python unified_agent.py "How does Agentic RAG compare to standard RAG?"
  python unified_agent.py --top-k 4 --quiet
"""

from __future__ import annotations

from agentRAG_utils import (
    SharedResources,
    build_unified_crew,
    load_environment,
    parse_indexed_args,
    print_result,
)


def main() -> None:
    args = parse_indexed_args(
        description="Run the unified Agentic RAG demo (all five patterns).",
        top_k_help=(
            "Number of tools selected by the object index for the retriever "
            "agent (default: 3)."
        ),
    )
    load_environment()

    # One LLM + one knowledge source, shared across all agents in the crew.
    resources = SharedResources.create()

    crew, explain = build_unified_crew(
        question=args.question,
        resources=resources,
        verbose=not args.quiet,
        top_k=args.top_k,
    )

    print(f"\n{explain}\n")

    result = crew.kickoff(inputs={"question": args.question})
    print_result("Unified Agentic RAG Result", result)


if __name__ == "__main__":
    main()
