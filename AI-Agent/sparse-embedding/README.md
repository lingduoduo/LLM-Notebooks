# Sparse Vector Search Engine (BM25)

> Companion material for *AI Agents in Depth*, Chapter 3 — **Experiment 3-5**: educational BM25 / inverted-index sparse search with offline CLI evaluation.  

### Overview

An educational sparse vector search engine using an inverted index and BM25. It demonstrates core IR concepts with extensive logging and visualization.

### Features

- **Full BM25 implementation**
- **Advanced tokenization**: numbers, codes, technical terms, mixed case
- **Inverted index** for term lookup
- **HTTP API** (FastAPI)
- **Interactive Web UI** for index/search
- **Educational logging** through index and search
- **Index visualization** APIs
- **In-memory storage** (educational simplicity)

#### Tokenization capabilities

- **Numbers**: `404`, `3.14`, `2.0.1`
- **Codes**: `XK9-2B4-7Q1`, `API_KEY_123`
- **Technical terms**: `C++`, `.NET`, `Node.js`
- **Mixed case**: `JavaScript`, `PyTorch`, `iPhone`
- **Email**: `user@example.com`
- **Hex**: `#FF5733`, `0x1234`
- **Acronyms**: `API`, `HTTP`, `NASA`
- **Alphanumeric**: `Python3`, `ES6`, `HTML5`

### Architecture

1. **TextProcessor**: tokenizer for words, numbers, codes, technical terms, mixed case  
2. **InvertedIndex**: term/document frequencies  
3. **BM25**: ranking  
4. **SparseSearchEngine**: orchestration  
5. **HTTP Server**: FastAPI surface  

BM25 uses TF, IDF, and document-length normalization. Key params: `k1` (default 1.5), `b` (default 0.75).

### Installation

```bash
pip install -r requirements.txt
```

`cli.py` (below) uses only the Python standard library and runs offline with no third-party packages; `server.py` / `demo.py` need FastAPI and the rest of `requirements.txt`.

### CLI tool `cli.py` (Experiment 3-5, recommended entry)

Fully offline CLI: BM25 on a built-in 10-doc corpus, per-term TF/IDF/BM25 contribution logs (as in the book), and labelled recall/precision/MRR.

```bash
python cli.py --help                          # all flags
python cli.py                                 # default demo query "model distillation"
python cli.py -q "model distillation" --explain   # per-term TF/IDF/BM25
python cli.py --eval                          # recall@k / precision@k / MRR
python cli.py -q "cat"                        # synonym failure (kitten/feline miss)
python cli.py --corpus my.json -q "reimbursement" -o out.json
python cli.py --k1 2.0 -b 0.5 -q "..."
python cli.py --method splade -q "..."        # SPLADE (needs downloaded model)
```

| Flag | Description |
| --- | --- |
| `-q, --query` | Query string (default `model distillation`) |
| `-c, --corpus` | Corpus (`.json` array or `.jsonl`); default built-in sample |
| `-m, --method` | `bm25` (default, offline) or `splade` (learned sparse; needs model) |
| `-k, --top-k` | Top-k (default 5) |
| `-o, --output` | Write results/metrics JSON |
| `--eval` | Evaluate recall@k / precision@k / MRR on labels |
| `--labels` | Custom labels `{query: [doc_id,...]}` |
| `--explain` | Per-term TF / IDF / BM25 on hits |
| `--k1` / `-b` | BM25 k1 and b |
| `-v, --verbose` | Engine DEBUG logs |

#### Retrieval quality (`--eval`)

Built-in labels cover exact keywords, error codes, proper names, and synonym-only queries. Real `python cli.py --eval` output (k=5):

```
Query 'model distillation'   recall@5=1.00  precision@5=1.00  RR=1.00
Query 'HTTP 404 error'        recall@5=1.00  precision@5=0.50  RR=1.00
Query 'XK9-2B4-7Q1'           recall@5=1.00  precision@5=1.00  RR=1.00
Query 'BM25 ranking function' recall@5=1.00  precision@5=1.00  RR=1.00
Query 'cat'                   recall@5=0.00  precision@5=0.00  RR=0.00   <- missed (synonym blind spot)

Macro average  recall@5=0.800  precision@5=0.700  MRR=0.800  miss rate(1-recall@5)=0.200
```

BM25 excels on exact keywords, codes, and names (recall=1.0) but misses synonyms—query `cat` does not hit docs that only say `kitten` / `feline`. That gap motivates hybrid search (Experiment 3-6 `retrieval-pipeline`).

#### Learned sparse (`--method splade`)

SPLADE weights terms with a masked LM and can expand semantically related terms. Needs pretrained `naver/splade-cocondenser-ensembledistil` (`torch`, `transformers`). Offline without weights, the command fails fast with a clear message (BM25 path needs no model). Online: `huggingface-cli download naver/splade-cocondenser-ensembledistil` then run.

### Server usage

```bash
python server.py
```

Server: `http://localhost:4241`. Web UI: open that URL. API docs: `http://localhost:4241/docs`.

#### API endpoints

```bash
POST /index
{
  "text": "Your document text here",
  "metadata": {"title": "Document Title", "category": "Category"}
}

POST /search
{
  "query": "your search query",
  "top_k": 10
}

GET /stats
GET /index/structure
GET /document/{doc_id}
DELETE /index
```

#### Demo

```bash
python demo.py
```

Demo: clear index → sample CS docs → stats → index structure → sample queries → document get.

### Educational features

Logging covers tokenization, TF, IDF, per-term BM25, query processing, candidates. `/index/structure` returns inverted map, doc stats, BM25 params, global TF distribution. Search results include matched terms, doc length, TFs, per-term score contributions.

### Project structure

```
sparse-embedding/
├── bm25_engine.py     # Core engine
├── cli.py             # Offline CLI: BM25/SPLADE + metrics
├── server.py          # FastAPI server
├── demo.py            # Demo script
├── requirements.txt
└── README.md
```

### Limitations

In-memory only; basic tokenization (no lemmatization); English stopwords; no phrase queries / synonyms / multi-thread.
