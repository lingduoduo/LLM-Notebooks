"""Object-Level Indexing Agentic RAG demo.

Mirrors the LlamaIndex ObjectIndex pattern:
  - All tool descriptions are embedded into an in-memory vector index.
  - The top-k most relevant tools for the question are retrieved semantically.
  - Only those tools are given to the agent, shrinking context and reducing errors.

Usage:
  python object_indexed_agent.py "What is Agentic RAG?"
  python object_indexed_agent.py --top-k 2 "What SQL data is available?"
  python object_indexed_agent.py --quiet "Summarise the key concepts."
"""

from __future__ import annotations

from agentRAG_utils import (
    build_object_indexed_crew,
    load_environment,
    parse_indexed_args,
    print_result,
)


def main() -> None:
    args = parse_indexed_args(
        description="Run the Object-Level Indexing Agentic RAG demo.",
        top_k_help="Number of tools to retrieve from the index (default: 3).",
    )
    load_environment()

    crew, explain = build_object_indexed_crew(
        question=args.question,
        verbose=not args.quiet,
        top_k=args.top_k,
    )

    print(f"\n{explain}\n")

    result = crew.kickoff(inputs={"question": args.question})
    print_result("Object-Indexed RAG Result", result)


if __name__ == "__main__":
    main()
