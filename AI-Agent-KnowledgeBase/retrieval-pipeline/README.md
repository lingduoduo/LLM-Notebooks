# Hybrid Retrieval Pipeline with Neural Reranking 

### Educational goals

1. **Dense vs sparse**: when each wins and why  
2. **Hybrid search**: combining methods  
3. **Neural reranking**: reorder candidates with transformers  
4. **Parallel processing**: multi-service index/search  
5. **Production-ish patterns**: API design and error handling  

### Architecture

```
┌──────────────────────────────────────────────┐
│            Client Application                 │
└────────────────────┬─────────────────────────┘
                     ▼
┌──────────────────────────────────────────────┐
│         Retrieval Pipeline (Port 4242)        │
│  Document Store (In-Memory)                   │
│  BGE-Reranker-v2 (Local Model)                │
└────────┬──────────────────┬─────────────────┘
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  Dense Service  │  │  Sparse Service │
│   (Port 4240)   │  │   (Port 4241)   │
│   BGE-M3 Model  │  │   BM25 Engine   │
└─────────────────┘  └─────────────────┘
```

### Key concepts

**Dense (BGE-M3)**: semantic / cross-lingual / synonyms; may miss exact codes; costlier.  
**Sparse (BM25)**: exact terms / IDs; no semantics; fast.  
**Fusion (`fusion.py`)**: RRF `score(d)=Σ 1/(k+rank)` with `k=60` (rank-only, scale-free) or weighted sum after min-max normalize to `[0,1]`.  
**Rerank**: BGE-Reranker-v2-M3 (service); `BAAI/bge-reranker-base` in `evaluate.py`.

### Prerequisites

Python 3.8+, macOS M1/M2 (or adjust device), ≥8GB RAM, ~5GB disk for models.

### Installation

```bash
cd AI-Agent-KnowledgeBase/retrieval-pipeline
pip install -r requirements.txt
# First run downloads: BGE-M3 ~2.3GB, BGE-Reranker-v2-M3 ~1.1GB
```

### Running services

```bash
./start_all_services.sh
# Dense 4240, Sparse 4241, Pipeline 4242
```

Or individually:

```bash
# Terminal 1
cd ../dense-embedding && python main.py --port 4240
# Terminal 2
cd ../sparse-embedding && python server.py 4241
# Terminal 3
cd ../retrieval-pipeline && python main.py --port 4242
```

### Testing with services

```bash
python test_client.py   # educational cases
python demo.py          # interactive demo
# API docs: http://localhost:4242/docs
```

### Offline evaluation CLI (`evaluate.py`)

`test_client.py` / `demo.py` need ports 4240–4242. **`evaluate.py` runs the full pipeline in one process — no service startup needed, and fully offline once the models are cached**. Note: the first run still downloads the dense/rerank models from HuggingFace, so initial execution requires network access.

```bash
python evaluate.py --help          # Chinese help
python evaluate.py                 # full stage table (default)
python evaluate.py --no-dense      # BM25 only, no models
python evaluate.py --no-rerank
python evaluate.py --query "XR-7003"
python evaluate.py --embed-model BAAI/bge-m3 --pooling cls
python evaluate.py --output result.json
```

| Stage | Default component | Offline? |
|-------|-------------------|----------|
| chunk | character-window splitter | ✅ pure Python |
| sparse | BM25 (`rank_bm25`) | ✅ no model download |
| dense | `sentence-transformers/all-MiniLM-L6-v2` (~90MB) | ✅ cached HF |
| fuse | RRF + weighted (`fusion.py`) | ✅ pure Python |
| rerank | `BAAI/bge-reranker-base` (~1.1GB first download) | ✅ once cached |

> `--no-dense` needs no ML model. Dense/rerank models download from HuggingFace on first run (network required); after that they run from local cache, and `--offline` forces loading from the local cache only. On Apple Silicon, MPS `NaN` is detected and falls back to CPU.

### Real output (reproduced)

Hard clusters: near-duplicate codes (`XR-7001..`, `HTTP-400..`) break dense; zero-lexical paraphrases break BM25.

```
Stage / Method            Recall@3         MRR      nDCG@3
------------------------------------------------------------------------------
BM25 (sparse)               0.9000      0.8500      0.8631
Dense                       1.0000      0.9000      0.9262
Hybrid-RRF                  1.0000      1.0000      1.0000
Hybrid-Weighted             1.0000      0.9500      0.9631
Hybrid-RRF+Rerank           1.0000      0.9500      0.9631
```

**How to read it:** BM25 nails codes, fails paraphrases; Dense is the mirror; **Hybrid-RRF** reaches perfect 1.00 (headline of Exp. 3-6). Weighted can be less robust (scale alignment). On this toy 17-doc set RRF is already strong; rerank value grows on larger pools / NL queries.

```
$ python evaluate.py --query "XR-7003"
[BM25 (sparse)]
  1. xr_7003        score=  3.2260  Product model XR-7003 is a smartphone available now.
[Dense]
  1. xr_7001        score=  0.5247  Product model XR-7001 ...
  2. xr_7003        score=  0.5195  Product model XR-7003 ...
[Hybrid-RRF]
  1. xr_7003        score=  0.0325  Product model XR-7003 ...
```

### Educational test cases (with services)

1. Semantic (“kitty behavior” / feline) — dense wins  
2. Exact name (“Alexander Humphrey”) — sparse wins  
3. Multilingual / reworded (“artificial intelligence” vs Spanish and paraphrased docs) — dense wins  
4. Codes (“HTTP-403”) — sparse wins  
5. Concepts (“happiness and excitement”) — dense wins  

### API

```bash
POST /index
{"text": "Document content", "doc_id": "optional_id", "metadata": {"category": "example"}}

POST /search
{"query": "search terms", "mode": "hybrid", "top_k": 20, "rerank_top_k": 10}

GET /stats
GET /documents?limit=10&offset=0
```

Response includes dense/sparse rankings, reranked results, rank changes, overlap stats.

### Project structure

```
retrieval-pipeline/
├── config.py, document_store.py, retrieval_client.py
├── reranker.py, fusion.py, retrieval_pipeline.py
├── evaluate.py, main.py, test_client.py, demo.py
├── requirements.txt, start_all_services.sh, stop_all_services.sh
└── README.md
```

### Performance / takeaways

- Latency ballpark: dense 50–100ms, sparse 10–30ms, rerank 100–200ms (20 docs)  
- Memory ~4GB models + docs  
- No single method wins; hybrid usually better; rerank improves relevance  

### Troubleshooting

Ports 4240–4242 free; models downloaded; Python 3.8+. OOM → smaller batches, CPU, FP16. First run slow (downloads).

### Further reading

[BGE-M3](https://arxiv.org/abs/2402.03216) · [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) · [Neural IR](https://arxiv.org/abs/2301.09191)

### License

Educational project for learning purposes.

## Shared service controller

From the parent directory, use `python services.py start --dry-run` to inspect the
canonical ports and commands; `start`, `status`, `stop` and `restart` manage the
services. The existing shell scripts delegate to this controller. It records
process identities under `.runtime/services/` and never kills arbitrary port
occupants. Real startup can download model weights. See the [parent guide](../README.md#retrieval-services)
for separate Python environments and startup/shutdown behavior.

The shared offline test runner is `python ../run_checks.py --suite retrieval-pipeline`.
Live examples remain available as `python test_client.py`, `python test_pipeline.py`
and `python demo.py` after the services are running.
