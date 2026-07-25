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
cd chapter3/retrieval-pipeline
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
cd ../sparse-embedding && python server.py --port 4241
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

---

## Quick Reference

A condensed version of the sections above.

### Educational goals

1. **Dense vs sparse**: where each one wins
2. **Hybrid retrieval**: the routes complement each other
3. **Neural reranking**: reorder candidates with a transformer
4. **Parallel processing**: indexing/searching across several services
5. **Engineering patterns**: API design and error handling

### Architecture

(Same as the English section: Pipeline 4242, Dense 4240, Sparse 4241.)

### Key concepts

**Dense (BGE-M3)**: semantics, cross-lingual matching, synonyms; can miss exact codes; more expensive to compute.
**Sparse (BM25)**: exact terms and IDs; no semantics; fast.
**Fusion (`fusion.py`)**: RRF (`k=60`), or min-max normalization followed by a weighted sum.
**Reranking**: the service uses BGE-Reranker-v2-M3; `evaluate.py` uses `BAAI/bge-reranker-base`.

### Requirements and installation

Python 3.8+, at least 8GB of RAM recommended, and about 5GB of disk for models.

```bash
cd chapter3/retrieval-pipeline
pip install -r requirements.txt
```

### Starting the services

```bash
./start_all_services.sh
```

Or one at a time:

```bash
cd ../dense-embedding && python main.py --port 4240
cd ../sparse-embedding && python server.py --port 4241
cd ../retrieval-pipeline && python main.py --port 4242
```

### Testing against the services

```bash
python test_client.py
python demo.py
# http://localhost:4242/docs
```

### Offline evaluation CLI (`evaluate.py`)

Runs chunk -> embed -> retrieve -> fuse -> rerank in a **single process that can work offline**.

```bash
python evaluate.py --help
python evaluate.py
python evaluate.py --no-dense
python evaluate.py --no-rerank
python evaluate.py --query "XR-7003"
python evaluate.py --embed-model BAAI/bge-m3 --pooling cls
python evaluate.py --output result.json
```

| Stage | Default component | Offline? |
|-------|-------------------|----------|
| chunk | character-window splitting | ✅ pure Python |
| sparse | BM25 | ✅ no model download |
| dense | MiniLM-L6-v2 (~90MB) | ✅ HF cache |
| fuse | RRF + weighted | ✅ pure Python |
| rerank | bge-reranker-base | ✅ cached after first download |

> `--no-dense` needs no ML model at all. On Apple Silicon, if MPS produces `NaN` the code falls back to CPU automatically.

### Reading the actual output

Near-duplicate codes break dense retrieval; paraphrases with zero lexical overlap break BM25. **Hybrid-RRF scoring 1.00 across the board** is the core result of experiment 3-6. Weighted fusion is more sensitive to score scale. On a small corpus RRF is already strong; reranking earns its keep with a larger candidate pool and natural-language queries.

Single-query trace:

```
$ python evaluate.py --query "XR-7003"
[BM25 (sparse)]
  1. xr_7003        ...
[Dense]
  1. xr_7001        ...   # dense ranks a sibling code first
  2. xr_7003        ...
[Hybrid-RRF]
  1. xr_7003        ...   # fusion pushes the exact match back to rank 1
```

### Teaching test cases (services required)

Semantics / exact person names / multilingual / technical codes / concept words -- watch dense or sparse win on each.

### API

```bash
POST /index
{"text": "Document content", "doc_id": "optional_id", "metadata": {"category": "example"}}

POST /search
{"query": "search terms", "mode": "hybrid", "top_k": 20, "rerank_top_k": 10}

GET /stats
GET /documents?limit=10&offset=0
```

The response includes the raw dense/sparse rankings, the reranked results, rank changes and overlap statistics.

### Project layout

```
retrieval-pipeline/
├── config.py, document_store.py, retrieval_client.py
├── reranker.py, fusion.py, retrieval_pipeline.py
├── evaluate.py, main.py, test_client.py, demo.py
├── requirements.txt, start_all_services.sh, stop_all_services.sh
└── README.md
```

### Performance notes

- Latency, order of magnitude: dense 50-100ms, sparse 10-30ms, reranking about 100-200ms (20 documents)
- Roughly 4GB of memory for the models
- There is no single best method; hybrid is usually better, and reranking improves relevance

### Troubleshooting

Check ports 4240-4242 and that the models downloaded. On OOM, reduce the batch size, switch to CPU, or enable FP16.

### Further reading and license

[BGE-M3](https://arxiv.org/abs/2402.03216) · [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) · teaching project.

---

## Notes

- Upstream services: [`../dense-embedding/`](../dense-embedding/) (4240), [`../sparse-embedding/`](../sparse-embedding/) (4241).
