"""
Main entry point for the structured index tool: build / query RAPTOR and
GraphRAG indexes, or run the offline comparison demo.

Note: **building** a RAPTOR or GraphRAG index calls an LLM (entity extraction,
recursive summarization), so build / query need OPENAI_API_KEY plus the heavy
dependencies (umap, sentence-transformers, ...). If you only want to see what
problem structured indexing solves that flat retrieval cannot, run the `demo`
subcommand, which needs no API key.
"""

import argparse
import asyncio
from pathlib import Path
import json
import logging
import sys

# loguru is the project's logger, but it is a hard dependency only for the
# build/query paths. Importing it at module level meant a missing loguru broke
# even `--help` and the dependency-free `demo`, contradicting the lazy-import
# design below. Fall back to the stdlib logger so the light paths still run.
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("structured-index")
    logger.warning("loguru is not installed; using the standard library logger. "
                   "Install it with: pip install loguru")


async def build_indexes(file_path: Path, index_type: str = "both",
                        output: str = None):
    """Build RAPTOR and/or GraphRAG indexes from a document."""
    # Import heavy dependencies lazily so --help and demo still work when
    # umap and friends are not installed
    from config import get_raptor_config, get_graphrag_config
    from raptor_indexer import RaptorIndexer
    from graphrag_indexer import GraphRAGIndexer
    from document_processor import DocumentProcessor

    logger.info(f"Building {index_type} index(es) from {file_path}")

    # Process document
    processor = DocumentProcessor()
    text = await processor.process_file(file_path)
    logger.info(f"Processed document: {len(text)} characters")

    all_stats = {}

    # Build RAPTOR index
    if index_type in ["raptor", "both"]:
        logger.info("Building RAPTOR tree index...")
        raptor_config = get_raptor_config()
        raptor = RaptorIndexer(raptor_config)
        raptor.build_index(text)
        raptor.save_index()
        stats = raptor.get_tree_statistics()
        all_stats["raptor"] = stats
        logger.info(f"RAPTOR index built: {stats}")

    # Build GraphRAG index
    if index_type in ["graphrag", "both"]:
        logger.info("Building GraphRAG knowledge graph...")
        graphrag_config = get_graphrag_config()
        graphrag = GraphRAGIndexer(graphrag_config)
        graphrag.build_knowledge_graph(text)
        graphrag.detect_communities()
        graphrag.hierarchical_summarization()
        graphrag.save_index()
        stats = graphrag.get_graph_statistics()
        all_stats["graphrag"] = stats
        logger.info(f"GraphRAG index built: {stats}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Index statistics written to {output}")

    logger.info("Indexing complete!")


async def query_indexes(query: str, index_type: str = "both", top_k: int = 5,
                        multi_hop: int = 0):
    """Query RAPTOR and/or GraphRAG indexes."""
    from config import get_raptor_config, get_graphrag_config
    from raptor_indexer import RaptorIndexer
    from graphrag_indexer import GraphRAGIndexer

    results = {}

    # Query RAPTOR
    if index_type in ["raptor", "both"]:
        try:
            raptor_config = get_raptor_config()
            raptor = RaptorIndexer(raptor_config)
            raptor.load_index()
            raptor_results = raptor.search(query, top_k)
            results["raptor"] = raptor_results
            logger.info(f"RAPTOR returned {len(raptor_results)} results")
        except Exception as e:
            logger.error(f"Error querying RAPTOR: {e}")

    # Query GraphRAG
    if index_type in ["graphrag", "both"]:
        try:
            graphrag_config = get_graphrag_config()
            graphrag = GraphRAGIndexer(graphrag_config)
            graphrag.load_index()
            graphrag_results = graphrag.search(query, top_k)
            results["graphrag"] = graphrag_results
            logger.info(f"GraphRAG returned {len(graphrag_results)} results")

            # Multi-hop relational retrieval: start from the best recalled
            # entity and walk the relationship edges
            if multi_hop > 0 and graphrag_results:
                start = next((r.get("name") for r in graphrag_results
                              if r.get("type") == "entity"), None)
                if start:
                    paths = graphrag.multi_hop_search(start, max_hops=multi_hop)
                    results["graphrag_multi_hop"] = paths
                    logger.info(f"GraphRAG multi-hop from '{start}' "
                                f"returned {len(paths)} paths")
        except Exception as e:
            logger.error(f"Error querying GraphRAG: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Structured index tool: build and query RAPTOR (hierarchical tree) "
                    "and GraphRAG (entity relationship graph) indexes in one framework, "
                    "corresponding to experiment 3-8 in the book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build a structured index from a document (needs OPENAI_API_KEY)")
    build_parser.add_argument("file", type=str,
                              help="Path to the document to index (.pdf/.txt/.md/.html)")
    build_parser.add_argument("--type", choices=["raptor", "graphrag", "both"],
                              default="both", help="Which index type to build (default: both)")
    build_parser.add_argument("--output", type=str, default=None,
                              help="Write index statistics to this JSON file")

    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Query a built index (needs OPENAI_API_KEY and an existing index)")
    query_parser.add_argument("query", type=str, help="Search query")
    query_parser.add_argument("--type", choices=["raptor", "graphrag", "both"],
                              default="both", help="Which index type to query (default: both)")
    query_parser.add_argument("--top-k", type=int, default=5,
                              help="Number of results to return (default: 5)")
    query_parser.add_argument("--multi-hop", type=int, default=0, metavar="N",
                              help="Additionally walk N hops of relationships in GraphRAG "
                                   "(0 disables it)")
    query_parser.add_argument("--output", type=str, default=None,
                              help="Write query results to this JSON file")

    # Demo command (offline, no API needed)
    demo_parser = subparsers.add_parser(
        "demo", help="Offline comparison demo: structured index vs flat retrieval "
                     "(no API key needed)")
    demo_parser.add_argument("--query", type=str, default=None,
                             help="Custom query; defaults to the three built-in "
                                  "comparison queries")
    demo_parser.add_argument("--top-k", type=int, default=3,
                             help="How many flat-retrieval results to show (default: 3)")
    demo_parser.add_argument("--output", type=str, default=None,
                             help="Write demo results to this JSON file")

    # Server command
    subparsers.add_parser("serve", help="Start the HTTP API service")

    args = parser.parse_args()

    if args.command == "build":
        asyncio.run(build_indexes(Path(args.file), args.type, args.output))
    elif args.command == "query":
        results = asyncio.run(query_indexes(args.query, args.type, args.top_k,
                                            args.multi_hop))

        # Display results
        for index_type, index_results in results.items():
            print(f"\n{index_type.upper()} Results:")
            print("-" * 50)
            if index_type == "graphrag_multi_hop":
                for i, r in enumerate(index_results, 1):
                    chain = r["path"][0]["source"]
                    for step in r["path"]:
                        chain += f" --{step['relation']}--> {step['target']}"
                    print(f"\n{i}. [{r['hops']} hop(s)] {chain}")
                continue
            for i, result in enumerate(index_results, 1):
                print(f"\n{i}. Score: {result.get('score', 'N/A'):.3f}")
                if 'summary' in result:
                    print(f"   Summary: {result['summary'][:200]}...")
                elif 'description' in result:
                    print(f"   Description: {result['description'][:200]}...")
                if 'level' in result:
                    print(f"   Level: {result['level']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\nQuery results written to {args.output}")
    elif args.command == "demo":
        from structured_vs_flat_demo import run_demo
        run_demo(top_k=args.top_k, custom_query=args.query, output=args.output)
    elif args.command == "serve":
        from api_service import run_server
        run_server()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
