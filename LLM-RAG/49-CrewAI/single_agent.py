"""Single-agent Agentic RAG demo with CrewAI knowledge sources."""

from __future__ import annotations

from agentRAG_utils import (
    build_single_agent_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the single-agent Agentic RAG demo.")
    load_environment()

    crew = build_single_agent_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Single-Agent RAG Result", result)


if __name__ == "__main__":
    main()
