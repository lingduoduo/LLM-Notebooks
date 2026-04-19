"""Five dynamic tools for Agentic RAG: vector RAG, summary RAG, text-to-SQL, web search, custom API.

Each tool has a distinct use-case described in its docstring so the agent's LLM can
decide which tool(s) to invoke based solely on the question semantics.
"""

from __future__ import annotations

from crewai.tools import tool


@tool("Vector RAG Engine")
def vector_rag_tool(query: str) -> str:
    """Precise, detail-level fact retrieval from a vector knowledge base.
    Use this when the question needs specific facts, definitions, examples, or citations.
    Not suitable for broad overviews or real-time data.
    """
    # Replace with a real vector store query (e.g., ChromaDB, Pinecone, Weaviate).
    return (
        f"[Vector RAG] Detail retrieval for '{query}':\n"
        "- Agentic RAG uses AI agents to dynamically decide retrieval strategies "
        "rather than relying on a fixed top-k pipeline.\n"
        "- Source: https://developer.volcengine.com/articles/7373874019113631754\n"
        "- Source: https://docs.crewai.com/en/concepts/knowledge"
    )


@tool("Summary RAG Engine")
def summary_rag_tool(topic: str) -> str:
    """High-level, macro-level overview of a topic from the knowledge base.
    Use this when the question needs broad conceptual understanding rather than specific facts.
    Not suitable for precise citations or detail-level queries.
    """
    # Replace with a summarisation-focused retrieval pipeline (e.g., map-reduce over chunks).
    return (
        f"[Summary RAG] High-level overview of '{topic}':\n"
        "Agentic RAG extends traditional RAG by introducing agent-driven decision-making. "
        "Agents actively plan retrieval steps, select tools, and iterate until the task "
        "is complete — enabling multi-step reasoning and complex task automation."
    )


@tool("Text-to-SQL Tool")
def text_to_sql_tool(natural_language_query: str) -> str:
    """Convert a natural language question into a SQL query and return structured results.
    Use this when the question involves structured data — tables, counts, filters, aggregations.
    Not suitable for unstructured document retrieval.
    """
    # Replace with a real Text-to-SQL engine (e.g., Vanna, LangChain SQL agent, sqlglot).
    keyword = natural_language_query.split()[0] if natural_language_query else "topic"
    sql = (
        f"SELECT id, title, summary FROM knowledge_base\n"
        f"WHERE topic LIKE '%{keyword}%'\n"
        f"ORDER BY relevance_score DESC\n"
        f"LIMIT 5;"
    )
    return (
        f"[Text-to-SQL] Generated query:\n{sql}\n\n"
        "Result: (stub — connect a real database to execute this query)"
    )


@tool("Web Search Tool")
def web_search_tool(search_query: str) -> str:
    """Search the web for real-time or up-to-date information not present in the knowledge base.
    Use this when the question requires current data, recent events, or live statistics.
    Not suitable when the knowledge base already contains sufficient information.
    """
    # Replace with SerperDevTool, DuckDuckGoSearchTool, or BraveSearchTool from crewai_tools.
    return (
        f"[Web Search] Live results for '{search_query}':\n"
        "(stub — integrate SerperDevTool or a real search API for live results)\n"
        "Tip: set SERPER_API_KEY and use crewai_tools.SerperDevTool() as a drop-in replacement."
    )


@tool("Custom API Tool")
def custom_api_tool(action: str) -> str:
    """Trigger a custom business action via an external API.
    Examples: send a notification, update a CRM record, post a message, or write to a database.
    Use this when the task requires acting on an external system, not just retrieving information.
    """
    # Replace with real HTTP calls: requests.post(endpoint, json={...}, headers={...})
    return (
        f"[Custom API] Action triggered: '{action}'\n"
        "(stub — wire up your API endpoint, authentication, and payload schema here)"
    )


ALL_TOOLS = [
    vector_rag_tool,
    summary_rag_tool,
    text_to_sql_tool,
    web_search_tool,
    custom_api_tool,
]

# ---------------------------------------------------------------------------
# Heterogeneous data-modality tools (one per modality for fusion demo)
# ---------------------------------------------------------------------------


@tool("Document Modality Tool")
def document_tool(query: str) -> str:
    """Retrieve content from unstructured document sources: PDF, Word, and web pages.
    Use this when the question requires reading narrative text, technical articles,
    reports, or any document-format knowledge source.
    Not suitable for tabular data, real-time feeds, or graph relationships.
    """
    # Replace with CrewDoclingSource queries, LangChain document loaders,
    # or LlamaIndex SimpleDirectoryReader for real document ingestion.
    return (
        f"[Document] Retrieved from documents for '{query}':\n"
        "- Agentic RAG decouples retrieval logic from the LLM, giving agents "
        "the ability to plan multi-step retrieval across heterogeneous sources.\n"
        "- Source: https://developer.volcengine.com/articles/7373874019113631754\n"
        "- Source: https://docs.crewai.com/en/concepts/knowledge"
    )


@tool("Structured Data Modality Tool")
def structured_data_tool(query: str) -> str:
    """Query structured data sources: relational database tables and Excel files.
    Use this when the question involves counts, aggregations, filters over rows,
    or any tabular / spreadsheet content.
    Not suitable for free-text documents, live streams, or graph traversal.
    """
    # Replace with a real Text-to-SQL engine (Vanna, LangChain SQL agent)
    # or a pandas/openpyxl pipeline for Excel files.
    keyword = query.split()[0] if query else "topic"
    sql = (
        f"SELECT source, count(*) AS mentions\n"
        f"FROM knowledge_records\n"
        f"WHERE content LIKE '%{keyword}%'\n"
        f"GROUP BY source ORDER BY mentions DESC LIMIT 10;"
    )
    return (
        f"[Structured Data] Query for '{query}':\n"
        f"Generated SQL:\n{sql}\n"
        "Result: (stub — connect a real database or load an Excel file to execute)"
    )


@tool("Real-Time Stream Modality Tool")
def stream_tool(topic: str) -> str:
    """Fetch real-time or near-real-time data from live APIs and message queues.
    Use this when the question requires up-to-date information that may not be
    present in static knowledge sources (e.g., current events, live metrics).
    Not suitable for historical documents or graph-structured data.
    """
    # Replace with requests/httpx calls to a REST API, or a Kafka/Pulsar consumer
    # for message queue data (e.g., confluent_kafka.Consumer).
    return (
        f"[Real-Time Stream] Live data for '{topic}':\n"
        "(stub — wire up an HTTP endpoint, WebSocket, or Kafka consumer here)\n"
        "Example: response = httpx.get('https://api.example.com/stream', params={'q': topic})"
    )


@tool("Knowledge Graph Modality Tool")
def graph_tool(entity: str) -> str:
    """Traverse a knowledge graph to find entity relationships and linked concepts.
    Use this when the question asks how concepts relate to each other, requires
    multi-hop entity linking, or needs ontological / taxonomic reasoning.
    Not suitable for plain document retrieval or numerical aggregations.
    """
    # Replace with a graph DB query (Neo4j Cypher, RDFLib SPARQL, NetworkX traversal).
    cypher = (
        f"MATCH (e:Entity {{name: '{entity}'}})-[r]->(related)\n"
        f"RETURN e, type(r) AS relation, related LIMIT 10;"
    )
    return (
        f"[Knowledge Graph] Entity relationships for '{entity}':\n"
        f"Generated Cypher:\n{cypher}\n"
        "Result: (stub — connect Neo4j or an RDF store to execute)\n"
        "Known relations: AgenticRAG → uses → LLMAgent; "
        "LLMAgent → selects → RetrievalTool; RetrievalTool → queries → KnowledgeBase"
    )


HETEROGENEOUS_TOOLS = [document_tool, structured_data_tool, stream_tool, graph_tool]

# All tools combined — used by the unified crew's ToolIndex
ALL_TOOLS_UNIFIED = ALL_TOOLS + HETEROGENEOUS_TOOLS
