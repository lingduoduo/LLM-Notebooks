# Structured Indexing: RAPTOR & GraphRAG 

### Overview

Two advanced approaches for large technical documents (e.g. Intel® SDM-style manuals):

1. **RAPTOR** — hierarchical tree with recursive abstractive summarization  
2. **GraphRAG** — entities, relations, communities, multi-hop traversal  

### Features

**RAPTOR:** multi-level abstraction; recursive summaries; leaf→root search; GMM clustering; UMAP.

**GraphRAG:** LLM entity/relation extract; community detection; community summaries; multi-strategy search; **`GraphRAGIndexer.multi_hop_search`** for “how is A connected to B” questions flat vector search cannot express.

**HTTP API:** build/query, uploads, async large docs, hybrid search, stats.

### Installation

```bash
cd chapter3/structured-index
pip install -r requirements.txt
cp env.example .env
# API keys and preferences
```

### CLI

Chinese `--help` on all subcommands: `python main.py --help`, `python main.py demo --help`, etc.

```
usage: main.py [-h] {build,query,demo,serve} ...
  build   Build structured indexes (needs OPENAI_API_KEY)
  query   Query existing indexes (needs key + built indexes)
  demo    Offline structured vs flat compare (no API key)
  serve   Start HTTP API
```

#### 0. Offline demo (no API key — recommended first)

Hand-curated small Intel x86 SIMD knowledge base; three query types: multi-hop, cross-node synthesis, multi-level navigation.

```bash
python main.py demo
python main.py demo --query "which register does VADDPS use"
python main.py demo --output demo_result.json
```

Example (multi-hop; flat fails, graph succeeds):

```
[Query 1 | Multi-hop relational reasoning] Before running the ADDPS instruction, which control register bit must the operating system set to 1?
-- Flat retrieval (independent fragments by lexical similarity) --
  1. [control-bit] CR4.OSFXSR  (score=0.569)
  2. [control-bit] CR0.EM  (score=0.223)
  ...
  X Isolated fragments only; it cannot *connect* ADDPS to a particular control bit.
-- Structured graph retrieval (multi-hop walk along relationship edges) --
  ADDPS --belongs_to--> SSE --requires_enabling--> CR4.OSFXSR
  ADDPS --belongs_to--> SSE --requires_cleared--> CR0.EM
  = Answer: CR4.OSFXSR (reachable from ADDPS in 2 hops)
```

> `build` / `query` need real indexes (LLM for entities/summaries) → `OPENAI_API_KEY` (embeddings: local SentenceTransformers). `demo` uses hand-authored structure so readers see the point without keys.

#### 1. Build (needs OPENAI_API_KEY)

```bash
python main.py build path/to/document.pdf
python main.py build path/to/document.pdf --type raptor
python main.py build path/to/document.pdf --type graphrag
python main.py build path/to/document.pdf --output stats.json
```

#### 2. Query

```bash
python main.py query "What are the MOV instruction variants?"
python main.py query "explain SSE instructions" --type raptor --top-k 10
python main.py query "SSE registers" --type graphrag --multi-hop 2
python main.py query "control registers" --output result.json
```

#### 3. Serve

```bash
python main.py serve
# http://localhost:4242
```

### HTTP API examples

```bash
curl -X POST "http://localhost:4242/upload" \
  -F "file=@path/to/intel_manual.pdf" \
  -F "index_type=both"

curl -X POST "http://localhost:4242/build" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf", "index_type": "both", "force_rebuild": false}'

curl -X POST "http://localhost:4242/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are vector instructions?", "index_type": "hybrid", "top_k": 5}'

curl http://localhost:4242/status
curl http://localhost:4242/statistics
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/build` | POST | Build from text/file |
| `/upload` | POST | Upload + build |
| `/query` | POST | Query indexes |
| `/status` | GET | Status |
| `/statistics` | GET | Stats |
| `/indexes` | DELETE | Clear |

### Project structure

```
structured-index/
├── config.py, raptor_indexer.py, graphrag_indexer.py
├── document_processor.py, api_service.py
├── structured_vs_flat_demo.py   # offline demo
├── main.py, requirements.txt, env.example
├── indexes/{raptor,graphrag}/, cache/
```

### How it works

**RAPTOR:** chunk → embed → leaves → GMM cluster → parent summaries → multi-level tree → multi-level search.

**GraphRAG:** entity extract → relations → NetworkX graph → communities → summaries → hierarchical merge → entity/community search (+ multi-hop).

### Advanced params (see `config.py`)

RAPTOR: `chunk_size`, `chunk_overlap`, `tree_depth`, `summarization_length`.  
GraphRAG: `chunk_size`, `max_knowledge_triples`, community algorithm, summarization model.

### Performance / troubleshooting

Large manuals take time; watch API rate limits and memory. Cache speeds re-queries. OOM → smaller chunks; check keys; start with smaller models for tests.

### Integration

Backend for agentic-rag style projects; see related chapter labs.

### References

- [RAPTOR](https://arxiv.org/abs/2401.18059)  
- [GraphRAG](https://github.com/microsoft/graphrag)  
- [Intel SDM](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)  

