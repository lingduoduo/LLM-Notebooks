"""Shared utilities for the Agentic RAG CrewAI demos."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from crewai import Agent, Crew, LLM, Process, Task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from dotenv import load_dotenv

import openai

from tool_definitions import ALL_TOOLS, ALL_TOOLS_UNIFIED, HETEROGENEOUS_TOOLS
from tool_index import ToolIndex


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_QUESTION = "What is Agentic RAG? Please be sure to provide sources."
DEFAULT_KNOWLEDGE_URLS = [
    "https://developer.volcengine.com/articles/7373874019113631754",
    "https://docs.crewai.com/en/concepts/knowledge",
]


class SharedResources:
    """A single LLM instance and knowledge source shared across all crew builders.

    Create once, pass everywhere — avoids redundant API-client and index construction.

    Usage:
        resources = SharedResources.create()
        crew1 = build_single_agent_crew(resources=resources)
        crew2 = build_multi_agent_crew(resources=resources)
    """

    def __init__(self, llm: LLM, knowledge_source: CrewDoclingSource) -> None:
        self.llm = llm
        self.knowledge_source = knowledge_source

    @classmethod
    def create(
        cls,
        model: str | None = None,
        sources: Sequence[str] | None = None,
        api_key: str | None = None,
    ) -> SharedResources:
        return cls(
            llm=create_llm(model=model, api_key=api_key),
            knowledge_source=create_content_source(sources),
        )


def load_environment() -> None:
    load_dotenv(override=True)


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or add it to a local .env file."
        )
    return api_key


def create_content_source(sources: Sequence[str] | None = None) -> CrewDoclingSource:
    return CrewDoclingSource(file_paths=list(sources or DEFAULT_KNOWLEDGE_URLS))


def create_llm(
    model: str | None = None,
    temperature: float = 0,
    api_key: str | None = None,
) -> LLM:
    return LLM(
        model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        api_key=api_key or get_api_key(),
    )


def _build_tool_index(
    question: str,
    tools: list,
    top_k: int,
) -> tuple[list, str]:
    """Embed *tools*, retrieve top-k for *question*, return (tools, score_table)."""
    client = openai.OpenAI(api_key=get_api_key())
    index = ToolIndex(tools=tools, top_k=top_k)
    index.build(client)
    return index.retrieve_and_explain(question, client)


def build_corrective_rag_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    """Corrective RAG crew implementing a 5-step self-correction feedback loop.

    Step 1 — Initial Retrieval:   Retriever pulls candidate facts from the knowledge base.
    Step 2 — Relevance Evaluation: Evaluator scores quality and identifies gaps.
    Step 3 — Query Rewriting:      Rewriter refines the query to address identified gaps.
    Step 4 — Supplemental Retrieval: Retriever re-fetches using the improved query.
    Step 5 — Synthesis:            Synthesizer merges both retrievals into a final answer.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm

    retriever = Agent(
        role="Knowledge Retriever",
        goal="Retrieve as many relevant facts and sources as possible for the given question.",
        backstory=(
            "You conduct thorough, source-attributed retrieval from the knowledge base. "
            "Your output is always used by a downstream evaluator, so completeness matters."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    evaluator = Agent(
        role="Relevance Evaluator",
        goal=(
            "Score the quality of retrieved content and explicitly list any gaps, "
            "ambiguities, or missing sub-topics."
        ),
        backstory=(
            "You are a critical reviewer. You assess whether retrieved content fully "
            "addresses the question, assign a relevance score (0–10), and output a "
            "structured gap report that the query rewriter will act on."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    rewriter = Agent(
        role="Query Rewriter",
        goal="Produce an improved retrieval query that fills the gaps identified by the evaluator.",
        backstory=(
            "You specialise in query reformulation. Given the original question and a gap "
            "report, you craft a refined query that targets the missing information."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    supplemental_retriever = Agent(
        role="Supplemental Knowledge Retriever",
        goal="Retrieve additional facts using the refined query to fill gaps from the first retrieval.",
        backstory=(
            "You perform a focused second-pass retrieval, strictly targeting the gaps "
            "identified in the evaluation step. Avoid repeating facts already retrieved."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    synthesizer = Agent(
        role="Answer Synthesizer",
        goal="Merge both retrieval passes into a single accurate, source-grounded final answer.",
        backstory=(
            "You reconcile and combine results from multiple retrieval rounds. "
            "You resolve any conflicts, eliminate redundancy, and produce a clear answer "
            "that cites sources from both retrieval passes."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    # Step 1
    retrieve_task = Task(
        description=(
            "Retrieve all relevant facts from the knowledge base for: {question}\n"
            "Include source references for every claim."
        ),
        expected_output="A list of facts with source references relevant to the question.",
        agent=retriever,
    )
    # Step 2
    evaluate_task = Task(
        description=(
            "Evaluate the retrieval results against the original question: {question}\n"
            "Output:\n"
            "  - Relevance score (0–10)\n"
            "  - List of well-covered aspects\n"
            "  - List of gaps or missing sub-topics that need further retrieval"
        ),
        expected_output=(
            "Structured evaluation: relevance score, covered aspects, and a gap list."
        ),
        agent=evaluator,
        context=[retrieve_task],
    )
    # Step 3
    rewrite_task = Task(
        description=(
            "Using the gap list from the evaluation, rewrite or expand the original "
            "query to target the missing information: {question}"
        ),
        expected_output="A refined retrieval query that specifically addresses the identified gaps.",
        agent=rewriter,
        context=[evaluate_task],
    )
    # Step 4
    supplemental_task = Task(
        description=(
            "Use the refined query to retrieve supplemental facts from the knowledge base. "
            "Focus only on filling the gaps — do not repeat already-retrieved facts."
        ),
        expected_output="Supplemental facts with source references covering the identified gaps.",
        agent=supplemental_retriever,
        context=[rewrite_task, evaluate_task],
    )
    # Step 5
    synthesize_task = Task(
        description=(
            "Merge the initial and supplemental retrieval results into a final answer "
            "for: {question}\n"
            "Resolve conflicts, remove duplicates, and cite sources from both passes."
        ),
        expected_output=(
            "A complete, source-grounded answer that draws on both retrieval passes, "
            "with a brief note on what the self-correction step added."
        ),
        agent=synthesizer,
        context=[retrieve_task, supplemental_task],
    )

    return Crew(
        agents=[retriever, evaluator, rewriter, supplemental_retriever, synthesizer],
        tasks=[retrieve_task, evaluate_task, rewrite_task, supplemental_task, synthesize_task],
        verbose=verbose,
        process=Process.sequential,
        knowledge_sources=[resources.knowledge_source],
    )


def build_object_indexed_crew(
    question: str,
    resources: SharedResources | None = None,
    verbose: bool = True,
    top_k: int = 3,
) -> tuple[Crew, str]:
    """Object-Level Indexing crew.

    Mirrors the LlamaIndex ObjectIndex pattern inside a pure CrewAI project:
      1. Embed all tool descriptions into an in-memory vector index.
      2. Retrieve the top-k tools most similar to *question*.
      3. Give the agent only those tools — not the full list.

    Returns (crew, explain_string) so the caller can print the retrieval scores.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm
    retrieved_tools, explain = _build_tool_index(question, ALL_TOOLS, top_k)

    agent = Agent(
        role="Object-Indexed Research Agent",
        goal=(
            "Answer the question using only the pre-selected tools provided. "
            "Always state which tool you used and why it was the best fit."
        ),
        backstory=(
            f"A semantic tool index selected {len(retrieved_tools)} of "
            f"{len(ALL_TOOLS)} available tools as most relevant for this question. "
            "You work efficiently because your toolbox is already focused."
        ),
        tools=retrieved_tools,
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    task = Task(
        description=(
            "Answer the following question using the tools available to you: {question}\n"
            "State which tool you chose and explain why it outranked the alternatives."
        ),
        expected_output=(
            "Tool selection rationale followed by a clear, source-grounded answer."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=verbose,
        process=Process.sequential,
        knowledge_sources=[resources.knowledge_source],
    )
    return crew, explain


def build_dynamic_tool_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    """Dynamic Tool Selection crew.

    A single agent is equipped with five tools covering the full retrieval spectrum:
      - VectorRAG      → precise fact retrieval
      - SummaryRAG     → macro-level overviews
      - Text-to-SQL    → structured / tabular data
      - Web Search     → real-time information
      - Custom API     → business action triggers

    The agent's LLM reads each tool's description and selects the best fit for the
    question — no hard-coded routing logic required.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm
    agent = Agent(
        role="Dynamic Research Agent",
        goal=(
            "Answer any question by selecting the most appropriate tool(s) from "
            "the toolbox. Always explain which tool you chose and why."
        ),
        backstory=(
            "You are a versatile analyst equipped with multiple retrieval tools. "
            "You match the tool to the question: vector search for details, "
            "summary search for overviews, SQL for structured data, web search "
            "for current events, and API calls for system actions."
        ),
        tools=ALL_TOOLS,
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    task = Task(
        description=(
            "Answer the following question using the most appropriate tool(s): {question}\n"
            "State which tool you selected, why you chose it over the alternatives, "
            "and provide a source-grounded answer."
        ),
        expected_output=(
            "Tool selection rationale followed by a clear, source-grounded answer."
        ),
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=verbose,
        process=Process.sequential,
    )


def build_heterogeneous_fusion_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    """Heterogeneous Data Fusion crew.

    Four specialist agents each own exactly one data-modality tool:
      - DocumentAgent      → unstructured docs (PDF, Word, web)
      - StructuredAgent    → relational tables and Excel files
      - StreamAgent        → real-time APIs and message queues
      - GraphAgent         → knowledge graph entity relationships

    A Fusion Coordinator collects all four outputs and synthesises a unified answer,
    resolving conflicts and cross-referencing findings across modalities.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm

    doc_tool, sql_tool, rt_tool, kg_tool = HETEROGENEOUS_TOOLS

    document_agent = Agent(
        role="Document Specialist",
        goal="Extract relevant facts from unstructured document sources for the question.",
        backstory=(
            "You are an expert at reading PDFs, Word files, and web articles. "
            "You surface key claims, definitions, and source references from narrative text."
        ),
        tools=[doc_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    structured_agent = Agent(
        role="Structured Data Specialist",
        goal="Query relational tables and Excel data to extract numerical and categorical facts.",
        backstory=(
            "You translate natural language questions into precise database queries. "
            "You return well-formatted tabular results with column headers and row counts."
        ),
        tools=[sql_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    stream_agent = Agent(
        role="Real-Time Stream Specialist",
        goal="Fetch the latest data from live APIs and message queues relevant to the question.",
        backstory=(
            "You monitor real-time data feeds and report the most current information. "
            "You always note the data freshness so the coordinator can weigh recency."
        ),
        tools=[rt_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    graph_agent = Agent(
        role="Knowledge Graph Specialist",
        goal="Traverse the knowledge graph to identify entity relationships and linked concepts.",
        backstory=(
            "You reason over graph-structured data — finding how entities relate, "
            "tracing multi-hop paths, and surfacing ontological connections."
        ),
        tools=[kg_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    fusion_coordinator = Agent(
        role="Fusion Coordinator",
        goal=(
            "Synthesise findings from all four data modalities into a single coherent answer. "
            "Resolve conflicts, highlight cross-modality corroborations, and cite every source."
        ),
        backstory=(
            "You are the final integration layer. You receive structured outputs from four "
            "specialists — documents, tables, streams, and graphs — and produce a unified "
            "answer that is richer than any single modality could provide alone."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    doc_task = Task(
        description="Retrieve relevant facts from document sources for: {question}",
        expected_output="Key facts and source references extracted from documents.",
        agent=document_agent,
    )
    sql_task = Task(
        description="Query structured data sources (tables/Excel) for: {question}",
        expected_output="Tabular results with column labels, row counts, and data provenance.",
        agent=structured_agent,
    )
    stream_task = Task(
        description="Fetch real-time data from live APIs or streams relevant to: {question}",
        expected_output="Current data snapshot with a freshness timestamp and source URL.",
        agent=stream_agent,
    )
    graph_task = Task(
        description="Traverse the knowledge graph for entities and relationships related to: {question}",
        expected_output="Entity relationship findings with node labels, edge types, and graph path.",
        agent=graph_agent,
    )
    fusion_task = Task(
        description=(
            "Synthesise the outputs from all four data-modality specialists into a final answer "
            "for: {question}\n"
            "Structure your answer as:\n"
            "  1. Cross-modality summary\n"
            "  2. Modality-by-modality highlights\n"
            "  3. Conflicts or gaps (if any)\n"
            "  4. Sources"
        ),
        expected_output=(
            "A unified, source-grounded answer structured across all four data modalities."
        ),
        agent=fusion_coordinator,
        context=[doc_task, sql_task, stream_task, graph_task],
    )

    return Crew(
        agents=[document_agent, structured_agent, stream_agent, graph_agent, fusion_coordinator],
        tasks=[doc_task, sql_task, stream_task, graph_task, fusion_task],
        verbose=verbose,
        process=Process.sequential,
    )


def build_single_agent_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    resources = resources or SharedResources.create()
    llm = resources.llm
    agent = Agent(
        role="Article Comprehension Expert",
        goal="Read and understand the supplied articles, then answer questions with sources.",
        backstory="You are precise, source-aware, and strong at technical article comprehension.",
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    task = Task(
        description="Answer the following question about the articles: {question}",
        expected_output="A concise, source-grounded answer to the question.",
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=verbose,
        process=Process.sequential,
        knowledge_sources=[resources.knowledge_source],
    )


def build_hierarchical_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    """Three-tier hierarchical crew.

    Tier 1 — Top Agent:    manager_llm decomposes the question and delegates.
    Tier 2 — Tool Agents:  KnowledgeRetriever + DomainExpert answer sub-tasks.
    Tier 3 — Coordinator:  aggregates state and produces the final answer.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm

    knowledge_retriever = Agent(
        role="Knowledge Retriever",
        goal="Extract precise facts, definitions, and source references from the knowledge base.",
        backstory=(
            "You are a focused retrieval specialist. Given a sub-question, you scan "
            "the knowledge base and return only verified, source-attributed facts."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    domain_expert = Agent(
        role="Domain Expert",
        goal="Interpret retrieved facts, explain technical concepts, and surface key insights.",
        backstory=(
            "You transform raw retrieved content into structured technical reasoning, "
            "highlighting principles, tradeoffs, and practical implications."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    coordinator = Agent(
        role="Result Coordinator",
        goal="Aggregate sub-task outputs into a coherent, source-grounded final answer.",
        backstory=(
            "You receive outputs from retrieval and analysis sub-tasks, resolve any "
            "conflicts, manage overall state, and produce the final synthesized response."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    # Tasks have no `agent=` — the manager (Top Agent) delegates dynamically.
    retrieve_task = Task(
        description=(
            "Break down the question into retrieval sub-queries and collect relevant "
            "facts from the knowledge base: {question}"
        ),
        expected_output="Structured facts with source references for each sub-query.",
    )
    analyse_task = Task(
        description=(
            "Using the retrieved facts, explain the core concepts, technical principles, "
            "and any tradeoffs relevant to: {question}"
        ),
        expected_output="A technical analysis grounded in the retrieved facts.",
        context=[retrieve_task],
    )
    coordinate_task = Task(
        description=(
            "Aggregate the retrieval and analysis outputs. Resolve conflicts, fill "
            "gaps, and produce a final answer to: {question}"
        ),
        expected_output="A clear, source-grounded final answer with a brief summary of the reasoning path.",
        context=[retrieve_task, analyse_task],
    )

    return Crew(
        agents=[knowledge_retriever, domain_expert, coordinator],
        tasks=[retrieve_task, analyse_task, coordinate_task],
        verbose=verbose,
        process=Process.hierarchical,
        manager_llm=llm,
        knowledge_sources=[resources.knowledge_source],
    )


def build_multi_agent_crew(
    resources: SharedResources | None = None,
    verbose: bool = True,
) -> Crew:
    resources = resources or SharedResources.create()
    llm = resources.llm
    researcher = Agent(
        role="Senior Researcher",
        goal="Collect accurate, relevant information from the knowledge base.",
        backstory=(
            "You identify definitions, key claims, examples, and sources from "
            "technical documents."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    analyst = Agent(
        role="Technical Analyst",
        goal="Analyze relationships, technical principles, and tradeoffs.",
        backstory=(
            "You turn raw research into structured technical reasoning and "
            "identify what matters for practitioners."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    summarizer = Agent(
        role="Content Summary Expert",
        goal="Produce a clear, accurate final answer with sources.",
        backstory=(
            "You synthesize research and analysis into concise, well-structured "
            "answers."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    research_task = Task(
        description=(
            "Research this question from the knowledge base: {question}. "
            "Collect definitions, characteristics, application scenarios, and sources."
        ),
        expected_output="A research report with relevant facts and source references.",
        agent=researcher,
    )
    analysis_task = Task(
        description=(
            "Analyze the research report. Explain core concepts, technical "
            "principles, advantages, and relationships between concepts."
        ),
        expected_output="A technical analysis grounded in the research report.",
        agent=analyst,
        context=[research_task],
    )
    summary_task = Task(
        description=(
            "Use the research and analysis to answer the original question. "
            "Keep the answer clear, accurate, and source-grounded."
        ),
        expected_output="A final answer with explanation and source references.",
        agent=summarizer,
        context=[research_task, analysis_task],
    )

    return Crew(
        agents=[researcher, analyst, summarizer],
        tasks=[research_task, analysis_task, summary_task],
        verbose=verbose,
        process=Process.sequential,
        knowledge_sources=[resources.knowledge_source],
    )


def build_unified_crew(
    question: str,
    resources: SharedResources | None = None,
    verbose: bool = True,
    top_k: int = 3,
) -> tuple[Crew, str]:
    """Unified Agentic RAG crew that combines all five patterns.

    Pattern 1 — Hierarchical:        Process.hierarchical + manager_llm orchestrates agents.
    Pattern 2 — Dynamic Tool Select:  ToolIndex pre-selects top-k tools before crew is built.
    Pattern 3 — Corrective RAG:       Evaluator + Rewriter + Supplemental Retriever in pipeline.
    Pattern 4 — Object-Level Index:   All 9 tools embedded; only relevant subset given to agents.
    Pattern 5 — Heterogeneous Fusion: Four modality specialists feed a Fusion Coordinator.

    All agents share a single LLM and knowledge source via SharedResources.

    Returns (crew, explain_string) where explain_string shows the tool retrieval scores.
    """
    resources = resources or SharedResources.create()
    llm = resources.llm
    ks = resources.knowledge_source

    # ── Pattern 4: Object-Level Indexing ──────────────────────────────────────
    # Embed all 9 tool descriptions; retrieve top-k most relevant for this question.
    selected_tools, explain = _build_tool_index(question, ALL_TOOLS_UNIFIED, top_k)

    # ── Agents ────────────────────────────────────────────────────────────────
    # Pattern 5 (Heterogeneous): three modality specialists with dedicated tools.
    # Pattern 2 (Dynamic Tools): the retriever agent gets the object-indexed tool subset.
    _, sql_tool, rt_tool, kg_tool = HETEROGENEOUS_TOOLS

    knowledge_retriever = Agent(
        role="Knowledge Retriever",
        goal="Retrieve initial facts from the knowledge base using the pre-selected tools.",
        backstory=(
            "You perform the first-pass retrieval. A semantic tool index has already "
            "narrowed your toolbox to the most relevant tools for this question. "
            "Use them to gather as many source-attributed facts as possible."
        ),
        tools=selected_tools,
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    evaluator = Agent(
        role="Relevance Evaluator",
        goal="Score retrieval quality and produce a structured gap report.",
        backstory=(
            "You are a critical quality reviewer. You assess whether the retrieved "
            "content fully answers the question, assign a relevance score (0–10), "
            "and list any missing sub-topics the rewriter must target."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    rewriter = Agent(
        role="Query Rewriter",
        goal="Reformulate the query to target gaps identified by the evaluator.",
        backstory=(
            "You specialise in query reformulation. Given the original question and "
            "a gap report, you produce a refined query that fills the missing pieces."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    structured_specialist = Agent(
        role="Structured Data Specialist",
        goal="Query relational tables and Excel sources for numerical and categorical facts.",
        backstory="You translate natural language into precise queries over structured data.",
        tools=[sql_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    stream_specialist = Agent(
        role="Real-Time Stream Specialist",
        goal="Fetch the latest data from live APIs and message queues.",
        backstory="You surface the most current information and note data freshness.",
        tools=[rt_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    graph_specialist = Agent(
        role="Knowledge Graph Specialist",
        goal="Traverse the knowledge graph for entity relationships and multi-hop links.",
        backstory="You reason over graph-structured data to find how concepts connect.",
        tools=[kg_tool],
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )
    fusion_coordinator = Agent(
        role="Fusion Coordinator",
        goal=(
            "Synthesise findings from all agents into a single coherent, "
            "source-grounded final answer."
        ),
        backstory=(
            "You are the final integration layer. You reconcile outputs from document "
            "retrieval, corrective re-retrieval, and three modality specialists — "
            "resolving conflicts, eliminating redundancy, and citing every source."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    # ── Tasks (no agent= set — Pattern 1: manager delegates) ─────────────────
    retrieve_task = Task(
        description=(
            "Using your pre-selected tools, retrieve all relevant facts for: {question}\n"
            "Include source references for every claim."
        ),
        expected_output="Source-attributed facts relevant to the question.",
    )
    # Pattern 3 — Corrective RAG: evaluate → rewrite → re-retrieve
    evaluate_task = Task(
        description=(
            "Evaluate the retrieval results against: {question}\n"
            "Output a relevance score (0–10), covered aspects, and a gap list."
        ),
        expected_output="Structured evaluation: score, covered aspects, gap list.",
        context=[retrieve_task],
    )
    rewrite_task = Task(
        description="Rewrite the query to address the gaps identified in the evaluation.",
        expected_output="A refined retrieval query targeting the identified gaps.",
        context=[evaluate_task],
    )
    supplemental_task = Task(
        description=(
            "Use the refined query to retrieve supplemental facts that fill the gaps. "
            "Do not repeat already-retrieved facts."
        ),
        expected_output="Supplemental facts with source references covering the gaps.",
        context=[rewrite_task, evaluate_task],
    )
    # Pattern 5 — Heterogeneous: parallel specialist tasks
    structured_task = Task(
        description="Query structured data sources (tables/Excel) for: {question}",
        expected_output="Tabular results with column labels and data provenance.",
    )
    stream_task = Task(
        description="Fetch real-time data from live APIs or streams for: {question}",
        expected_output="Current data snapshot with freshness timestamp and source.",
    )
    graph_task = Task(
        description="Traverse the knowledge graph for entities related to: {question}",
        expected_output="Entity relationship findings with node labels and edge types.",
    )
    fusion_task = Task(
        description=(
            "Synthesise all agent outputs into a final answer for: {question}\n"
            "Structure as:\n"
            "  1. Unified answer\n"
            "  2. What self-correction added (gap → supplemental finding)\n"
            "  3. Cross-modality highlights\n"
            "  4. Sources"
        ),
        expected_output=(
            "A complete, source-grounded answer integrating document retrieval, "
            "corrective re-retrieval, and all three modality specialists."
        ),
        context=[
            retrieve_task,
            supplemental_task,
            structured_task,
            stream_task,
            graph_task,
        ],
    )

    crew = Crew(
        agents=[
            knowledge_retriever,
            evaluator,
            rewriter,
            structured_specialist,
            stream_specialist,
            graph_specialist,
            fusion_coordinator,
        ],
        tasks=[
            retrieve_task,
            evaluate_task,
            rewrite_task,
            supplemental_task,
            structured_task,
            stream_task,
            graph_task,
            fusion_task,
        ],
        verbose=verbose,
        process=Process.hierarchical,  # Pattern 1: manager_llm orchestrates all agents
        manager_llm=llm,               # shared LLM acts as Top Agent
        knowledge_sources=[ks],        # shared knowledge source
    )
    return crew, explain


def print_result(title: str, result) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(result)
    print("=" * 80)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(
    description: str,
    *,
    top_k_help: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "question",
        nargs="?",
        default=os.getenv("AGENTIC_RAG_QUESTION", DEFAULT_QUESTION),
        help="Question to answer from the knowledge base.",
    )
    if top_k_help is not None:
        parser.add_argument(
            "--top-k",
            type=positive_int,
            default=3,
            dest="top_k",
            help=top_k_help,
        )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose CrewAI execution output.",
    )
    return parser.parse_args()


def parse_indexed_args(
    description: str,
    top_k_help: str = "Number of tools to retrieve from the index (default: 3).",
) -> argparse.Namespace:
    return parse_args(description, top_k_help=top_k_help)
