"""Dynamic Tool Selection Agentic RAG demo.

The agent picks from five tools at runtime based on question semantics:
  vector_rag_tool   — precise fact retrieval
  summary_rag_tool  — macro-level overviews
  text_to_sql_tool  — structured / tabular queries
  web_search_tool   — real-time information
  custom_api_tool   — external system actions
"""

from __future__ import annotations

from agentRAG_utils import (
    build_dynamic_tool_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the dynamic tool selection Agentic RAG demo.")
    load_environment()

    crew = build_dynamic_tool_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Dynamic Tool Selection RAG Result", result)


if __name__ == "__main__":
    main()
