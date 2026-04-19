# 49-CrewAI

Eight Agentic RAG demos using CrewAI. The first seven each demonstrate one pattern in isolation; the eighth (`unified_agent.py`) combines all five patterns into a single collaborating crew.

## Setup

```bash
cd LLM-RAG/49-CrewAI
python -m pip install -e .
```

Set your API key once (or add it to a `.env` file):

```bash
export OPENAI_API_KEY=sk-...
```

## Demos

### 1. Single Agent
```bash
python single_agent.py
```

### 2. Sequential Multi-Agent
```bash
python multi_agents.py
```

### 3. Hierarchical Task Planning
```bash
python hierarchical_agents.py
```

### 4. Dynamic Tool Selection
```bash
python dynamic_tools_agent.py
```

### 5. Corrective RAG (Self-Correction Feedback Loop)
```bash
python corrective_rag_agent.py
```

### 6. Object-Level Tool Indexing
```bash
python object_indexed_agent.py
python object_indexed_agent.py --top-k 2 "What SQL data is available?"
```

### 7. Heterogeneous Data Fusion
```bash
python heterogeneous_fusion_agent.py
```

### 8. Unified — All Patterns Collaborating
All five patterns in one crew. Agents share a single LLM and knowledge source (`SharedResources`).
```bash
python unified_agent.py
python unified_agent.py --top-k 4 "How does Agentic RAG compare to standard RAG?"
python unified_agent.py --quiet
```

All scripts accept an optional positional question and a `--quiet` flag:

```bash
python multi_agents.py "What makes Agentic RAG different from standard RAG?"
python multi_agents.py --quiet
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | yes | — | OpenAI API key. |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Model name passed to CrewAI LLM. |
| `AGENTIC_RAG_QUESTION` | no | (built-in) | Fallback question when none is supplied on the CLI. |

## Files

| File | Pattern | Description |
|------|---------|-------------|
| `single_agent.py` | Single agent | Article comprehension with CrewAI knowledge sources. |
| `multi_agents.py` | Sequential multi-agent | Researcher → Analyst → Summarizer pipeline. |
| `hierarchical_agents.py` | Hierarchical task planning | Manager LLM (Top Agent) dynamically delegates to KnowledgeRetriever, DomainExpert, and Coordinator. |
| `dynamic_tools_agent.py` | Dynamic tool selection | Agent picks from 5 tools at runtime based on question semantics: Vector RAG, Summary RAG, Text-to-SQL, Web Search, Custom API. |
| `corrective_rag_agent.py` | Self-correction feedback loop | 5-step Corrective RAG: retrieve → evaluate → rewrite query → re-retrieve → synthesize. |
| `object_indexed_agent.py` | Object-level tool indexing | Embeds all tool descriptions into a vector index; retrieves top-k tools by cosine similarity before building the crew. Supports `--top-k N`. |
| `heterogeneous_fusion_agent.py` | Heterogeneous data fusion | Four modality-specialist agents (document, structured, stream, graph) each own one tool; a Fusion Coordinator cross-references all four outputs. |
| `tool_definitions.py` | — | Nine `@tool`-decorated functions covering dynamic-selection tools (Vector RAG, Summary RAG, Text-to-SQL, Web Search, Custom API) and heterogeneous-modality tools (document, structured data, real-time stream, knowledge graph). All non-trivial implementations are stubs with clear replacement comments. |
| `tool_index.py` | — | `ToolIndex` class: embeds tool objects via OpenAI `text-embedding-3-small`, retrieves top-k by cosine similarity, and prints a ranked score table via `explain()`. |
| `unified_agent.py` | All five patterns | Single crew combining hierarchical orchestration, object-indexed tool selection, corrective RAG loop, and heterogeneous modality specialists — all sharing one `SharedResources` instance. |
| `agentRAG_utils.py` | — | Shared utilities: `SharedResources` dataclass, LLM creation, knowledge source wiring, all crew builder functions, CLI argument parsing, and result printing. |

The scripts intentionally do not print API keys or store execution transcripts in source files.
