"""Hierarchical Agentic RAG demo — Top Agent / Tool Agents / Coordinator."""

from __future__ import annotations

from agentRAG_utils import (
    build_hierarchical_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the hierarchical Agentic RAG demo.")
    load_environment()

    crew = build_hierarchical_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Hierarchical RAG Result", result)


if __name__ == "__main__":
    main()
