"""Heterogeneous Data Fusion Agentic RAG demo.

Four specialist agents each own one data-modality tool:
  DocumentAgent      — unstructured docs (PDF, Word, web pages)
  StructuredAgent    — relational tables and Excel files
  StreamAgent        — real-time APIs and message queues
  GraphAgent         — knowledge graph entity relationships

A Fusion Coordinator synthesises all four outputs into a unified answer,
cross-referencing findings and resolving conflicts across modalities.
"""

from __future__ import annotations

from agentRAG_utils import (
    build_heterogeneous_fusion_crew,
    load_environment,
    parse_args,
    print_result,
)


def main() -> None:
    args = parse_args("Run the Heterogeneous Data Fusion Agentic RAG demo.")
    load_environment()

    crew = build_heterogeneous_fusion_crew(verbose=not args.quiet)
    result = crew.kickoff(inputs={"question": args.question})
    print_result("Heterogeneous Fusion RAG Result", result)


if __name__ == "__main__":
    main()
