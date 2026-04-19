"""Corrective RAG demo — self-correction feedback loop.

Five-step pipeline:
  1. Initial Retrieval       — pull candidate facts from the knowledge base
  2. Relevance Evaluation    — score quality and identify gaps
  3. Query Rewriting         — refine the query to target missing information
  4. Supplemental Retrieval  — re-retrieve with the improved query
  5. Synthesis               — merge both passes into a final answer
"""

from __future__ import annotations

from agentRAG_utils import (
    build_corrective_rag_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the Corrective RAG self-correction demo.")
    load_environment()

    crew = build_corrective_rag_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Corrective RAG Result", result)


if __name__ == "__main__":
    main()
