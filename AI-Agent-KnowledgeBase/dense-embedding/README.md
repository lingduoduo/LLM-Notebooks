# Vector Similarity Search Service (Dense Embedding) 

### Overview

Educational HTTP service for vector similarity search using BGE-M3 embeddings with configurable ANNOY or HNSW backends, plus an offline `cli.py` for Experiment 3-4 metrics.

### CLI: dense retrieval & ANN comparison (`cli.py`, Experiment 3-4)

Besides the HTTP service, `cli.py` is **ready-to-run and offline-reproducible**—no need to start the server first:

1. **Semantic power of dense embeddings** — `recall@k / precision@k / MRR` on a small labelled corpus  
2. **ANN backend comparison** (focus of Exp. 3-4) — ANNOY / HNSW from `indexing.py` vs **exact brute-force**, measuring recall, build time, query latency  

#### Usage

```bash
# 1) Single dense query (default "a cat playing"; needs embedding model)
python cli.py -q "model distillation" -k 3

# 2) Retrieval quality: recall@k / precision@k / MRR
python cli.py --eval

# 2') Offline: small cached model (no 2.3GB BGE-M3 download)
python cli.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --eval

# 3) ANN backend compare (synthetic vectors; fully offline, no model)
python cli.py --compare-ann -k 10
python cli.py --compare-ann --backend hnsw --hnsw-ef-search 200 -k 10

# Custom corpus / labels / output
python cli.py --corpus my.json --labels my_labels.json --eval -o result.json
```

`python cli.py --help` has full Chinese flag docs.

#### Common flags

| Flag | Description |
| --- | --- |
| `-q, --query` | Query (default `a cat playing`) |
| `-c, --corpus` | Corpus (`.json` array or `.jsonl`); default built-in sample |
| `-k, --top-k` | Top-k (default 5) |
| `-o, --output` | Write results/metrics JSON |
| `--embedding-model` | Model (default `BAAI/bge-m3`; offline: `sentence-transformers/all-MiniLM-L6-v2`) |
| `--pooling` | `auto` / `mean` / `cls` |
| `--eval` | Evaluate recall@k / precision@k / MRR |
| `--compare-ann` | Compare ANNOY / HNSW (synthetic vectors) |
| `--ann-base / --ann-dim / --ann-queries` | Synthetic base size / dim / queries (default 3000 / 128 / 100) |
| `--annoy-n-trees / --hnsw-M / --hnsw-ef-search` | ANN hyperparameters |

#### Measured results (real runs)

**Dense quality** (12-doc built-in, `all-MiniLM-L6-v2`, offline):

```
Macro average  recall@5=1.000  precision@5=0.320  MRR=1.000  miss rate(1-recall@5)=0.000
```

Query `a cat playing` ranks docs that only say `kitten` / `feline` (no literal “cat”) at ranks 1–2—semantic strength vs BM25 (Exp. 3-5 may miss them).

**ANN compare** (3000 × 128-d unit vectors, 100 queries, top-10); HNSW recall rises with `ef_search`:

| Config | recall@10 | Mean query latency |
| --- | --- | --- |
| HNSW `ef_search=20` | 0.562 | 0.05 ms |
| HNSW `ef_search=200` | 0.991 | 0.25 ms |

> **Environment note**: each backend is health-checked by self-querying. On some macOS/arm64 setups, prebuilt `annoy==1.17.3` is broken (even self-query only returns itself); the tool warns and marks those numbers untrusted. HNSW is unaffected. Full ANNOY vs HNSW: use an environment where Annoy works (e.g. Linux x86_64).

### Service features

- **BGE-M3**: dense embeddings, 100+ languages, long context (up to 8192 tokens)  
- **Dual backends**: ANNOY (tree), HNSW (graph)  
- **Educational logging**: embed, index ops, metrics, vector stats  
- **REST API**: index / delete / search / stats  
- **In-memory** (no persistence)  

### Architecture

```
┌──────────────────┐
│   HTTP Client    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  FastAPI Server  │
└────────┬─────────┘
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌──────────────┐
│ Document │ │  Embedding   │
│  Store   │ │   Service    │
└──────────┘ │  (BGE-M3)    │
             └──────┬───────┘
          ┌─────────┴──────────┐
          ▼                    ▼
    ┌──────────┐        ┌──────────┐
    │  ANNOY   │        │   HNSW   │
    └──────────┘        └──────────┘
```

### Installation

- Python 3.8+, macOS (M1/M2 optimized) or Linux  
- ≥4GB RAM (8GB recommended); optional CUDA GPU  

```bash
cd chapter3/dense-embedding
pip install -r requirements.txt
```

BGE-M3 (~2.3GB) downloads on first use into the HuggingFace cache.

### Starting the service

```bash
python main.py                              # HNSW (default)
python main.py --index-type annoy
python main.py --index-type hnsw --host 0.0.0.0 --port 4242 --debug --show-embeddings
```

Options: `--index-type` (`annoy`|`hnsw`, default `hnsw`), `--host` (default `0.0.0.0`), `--port` (default `4240`), `--debug`, `--show-embeddings`.

Docs: http://localhost:4240/docs · OpenAPI: http://localhost:4240/openapi.json

### API endpoints

**POST `/index`**

```json
{
  "text": "Machine learning is a subset of artificial intelligence.",
  "doc_id": "doc_001",
  "metadata": {"category": "AI", "author": "John Doe"}
}
```

**POST `/search`**

```json
{
  "query": "What is deep learning?",
  "top_k": 5,
  "return_documents": true
}
```

**DELETE `/index`** — body `{"doc_id": "doc_001"}`  
**GET `/stats`** · **GET `/documents?limit=10`**

### Testing

```bash
python test_client.py
python test_client.py --performance
```

```bash
curl -X POST http://localhost:4240/index \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test document about machine learning."}'

curl -X POST http://localhost:4240/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 5}'
```

### Index comparison

**ANNOY**: fast build, low memory, good for static/read-heavy; rebuild for delete; trade accuracy via `n_trees`.  
**HNSW**: high recall, incremental updates, soft delete; more memory, slower build; tune `M` / `ef_*`.

### Configuration (env `VEC_` prefix)

```bash
export VEC_INDEX_TYPE=hnsw
export VEC_MODEL_NAME=BAAI/bge-m3
export VEC_USE_FP16=true
export VEC_MAX_SEQ_LENGTH=512
export VEC_MAX_DOCUMENTS=100000
export VEC_LOG_LEVEL=DEBUG
export VEC_ANNOY_N_TREES=50
export VEC_ANNOY_METRIC=angular
export VEC_HNSW_EF_CONSTRUCTION=200
export VEC_HNSW_M=32
export VEC_HNSW_EF_SEARCH=100
export VEC_HNSW_SPACE=cosine
```

Educational logging: `python main.py --debug --show-embeddings`.

### Memory / optimization notes

- Model ~2.3GB; ~4KB per doc (1024-d float32)  
- ANNOY: raise `n_trees` for accuracy; `angular` for normalized vectors; batch then build  
- HNSW: raise `M` / `ef_construction` / `ef_search` for quality vs cost  
- FP16 faster with slight accuracy trade-off  

### Troubleshooting

OOM → smaller batches, FP16, lower `max_seq_length`, prefer ANNOY. Slow index → lower `ef_construction` / `n_trees`, use GPU. Poor quality → raise `n_trees` / `M` / `ef_search`.

### References

- [BGE-M3 Paper](https://arxiv.org/abs/2402.03216) · [Model](https://huggingface.co/BAAI/bge-m3)  
- [ANNOY](https://github.com/spotify/annoy) · [HNSWlib](https://github.com/nmslib/hnswlib) · [FastAPI](https://fastapi.tiangolo.com/)
