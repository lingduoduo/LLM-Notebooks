# Neo4j KG Demo

This folder now includes a very simple, reliable Neo4j knowledge graph demo for the `LLM-RAG-KG` project.

## Recommended Starting Point

Open [`simple_neo4j_kg_demo.ipynb`](/Users/linghuang/Git/LLM-Notebooks/LLM-RAG-KG/Neo4j/simple_neo4j_kg_demo.ipynb).

The notebook does four things:

1. Connects to Neo4j using environment variables.
2. Loads a tiny demo knowledge graph about `LLM`, `RAG`, `Knowledge Graph`, `Neo4j`, and related concepts.
3. Runs a few plain Cypher queries to show the graph is useful.
4. Optionally uses [`text2cypher.py`](/Users/linghuang/Git/LLM-Notebooks/LLM-RAG-KG/Neo4j/text2cypher.py) to translate a natural-language question into Cypher.

## Requirements

Set these environment variables before running the notebook:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
```

Optional for the final text-to-Cypher section:

```bash
export OPENAI_API_KEY="your-openai-key"
```

Install the Python packages used by the notebook and helpers:

```bash
pip install neo4j openai tiktoken
```

If you want schema inspection through `apoc.meta.data()`, make sure the Neo4j instance has APOC available.

## Why This Demo

Some of the older files in this folder are experiments or are still tied to a movie-domain example. This notebook avoids that complexity and focuses on a clean, repeatable Neo4j KG walkthrough.

## Related Files

- [`utils.py`](/Users/linghuang/Git/LLM-Notebooks/LLM-RAG-KG/Neo4j/utils.py): connection helpers and OpenAI helpers
- [`schema_utils.py`](/Users/linghuang/Git/LLM-Notebooks/LLM-RAG-KG/Neo4j/schema_utils.py): Neo4j schema inspection
- [`text2cypher.py`](/Users/linghuang/Git/LLM-Notebooks/LLM-RAG-KG/Neo4j/text2cypher.py): optional natural-language to Cypher generation

