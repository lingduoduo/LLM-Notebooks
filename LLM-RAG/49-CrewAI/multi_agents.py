"""Multi-agent Agentic RAG demo with researcher, analyst, and summarizer roles."""

from __future__ import annotations

from agentRAG_utils import (
    build_multi_agent_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the multi-agent Agentic RAG demo.")
    load_environment()

    crew = build_multi_agent_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Multi-Agent RAG Result", result)


if __name__ == "__main__":
    main()
