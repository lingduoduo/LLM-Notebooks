# Contextual Retrieval System 

### Overview

Educational implementation of Anthropic’s Contextual Retrieval: prepend chunk-specific context before embedding/indexing to fix the “orphaned chunk” problem.

### Key insight

**Problem:** Traditional RAG loses context when chunking. “The company’s revenue grew by 3%” is meaningless without which company / which period.

**Solution:** Generate short explanatory context per chunk and prepend it before indexing so BM25 and embeddings keep identity signals.

### Core offline experiment (Experiment 3-11)

`compare_retrieval.py` quantifies the claim **fully offline**: the same chunks
are indexed two ways—plain (raw text only) and contextual (prefix + text)—then
compared with `recall@k` on `evaluation/retrieval_eval.json`. The default
`evaluation/contextual_retrieval_corpus.json` is a compact, curated paraphrase
fixture covering all 15 gold chunk IDs. Its metadata links to the official
government publications of the Constitution and Procurators Law. It is an
educational retrieval fixture, not authoritative legal text.

```bash
python compare_retrieval.py
python compare_retrieval.py --per-query
python compare_retrieval.py --query "What are the powers of the President?" --top-k 5
python compare_retrieval.py --mode plain
python compare_retrieval.py --output result.json
python compare_retrieval.py --corpus document_store.json
python compare_retrieval.py --help   # Chinese help
```

Bundled-fixture run (15 chunks, 15 queries, jieba):

```
Retrieval recall: plain chunks  vs.  contextual retrieval (BM25)
====================================================================
Method            recall@1    recall@3    recall@5
----------------------------------------------------
plain                86.7%     100.0%     100.0%
contextual          100.0%     100.0%     100.0%
----------------------------------------------------
gain (pp)          +13.3pp      +0.0pp      +0.0pp
----------------------------------------------------
failure drop          100%           -           -
```

Conclusion: in the deterministic teaching fixture, context prefixes lift
top-1 recall from 86.7% to 100%. The gain is strongest at recall@1;
`--query` shows how the prefix re-ranks the correct section first. Results
from a user-generated `document_store.json` will depend on its documents,
chunking, and generated contexts.

> `--method embedding` / `--method hybrid` need embedding APIs (not offline); the script falls back to BM25 offline results. Full dense + rerank lives in `contextual_tools.py`.  
> Same logic is also in `ContextualChunker.compare_retrieval_methods()`.

### Educational features

1. Watch LLM context generation per chunk  
2. Dual indexing (BM25 + embeddings) benefits from context  
3. Compare with `use_contextual=False`  
4. Metrics and token/cost awareness  

### Architecture

```
Document → Basic Chunking → Context Generation (optional LLM)
  → Enhanced chunks (context+text vs text only)
  → Retrieval pipeline (sparse BM25 + dense embeddings)
  → Hybrid search + reranking
```

### Quick start

```bash
pip install -r requirements.txt
cp env.example .env
# Set OPENAI_API_KEY in .env (LLM_MODEL defaults to gpt-5.6-terra).

# Separate terminal for full e2e with pipeline:
cd ../retrieval-pipeline
python main.py
# http://localhost:4242

# Back in this project directory (or a second terminal):
cd ../contextual-retrieval

# Index with contextual enhancement
python index_local_laws_contextual.py
python index_local_laws_contextual.py --no-contextual

# Queries
python main.py
python main.py --query "What does Article 1 of the Constitution say" --mode agentic
python main.py --query "What does Article 1 of the Constitution say" --mode compare
```

`--model` is an optional OpenAI model override for `main.py`. The indexing
script accepts `--llm-model` as an optional OpenAI model override.

### Context generation process

1. Provide full document (or surrounding context) to the LLM  
2. Show the specific chunk  
3. Ask for 2–3 sentence situating context  

Template sketch:

```
<document>
[Full document or surrounding context]
</document>

Here is the chunk we want to situate:
<chunk>
[Specific chunk text]
</chunk>

Please give a short, succinct context to situate this chunk within the overall document...
```

### References / license

- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)  
- [Constitution of the People's Republic of China](https://english.www.gov.cn/archive/lawregulations/201911/20/content_WS5ed8856ec6d0b3f0e9499913.html)
- [Procurators Law of the People's Republic of China](https://en.moj.gov.cn/2021-06/24/c_635994.htm)
- Educational project for learning purposes. Acknowledgments: Anthropic engineering research.
