# Contextual Retrieval System 

### Overview

Educational implementation of Anthropic’s Contextual Retrieval: prepend chunk-specific context before embedding/indexing to fix the “orphaned chunk” problem.

### Key insight

**Problem:** Traditional RAG loses context when chunking. “The company’s revenue grew by 3%” is meaningless without which company / which period.

**Solution:** Generate short explanatory context per chunk and prepend it before indexing so BM25 and embeddings keep identity signals.

### Core offline experiment (Experiment 3-11)

`compare_retrieval.py` quantifies the claim **fully offline**: same chunks indexed two ways—plain (raw text only) vs contextual (LLM-generated prefix + text)—then compares `recall@k` on `evaluation/retrieval_eval.json` (15 queries + human gold chunks). **No API or retrieval service** (BM25 + jieba).

```bash
python compare_retrieval.py
python compare_retrieval.py --per-query
python compare_retrieval.py --query "What are the powers of the President?" --top-k 5
python compare_retrieval.py --mode plain
python compare_retrieval.py --output result.json
python compare_retrieval.py --help   # Chinese help
```

Real run (22 Constitution / Prosecutor Law chunks, 15 queries, jieba):

```
Retrieval recall: plain chunks  vs.  contextual retrieval (BM25)
====================================================================
Method            recall@1    recall@3    recall@5
----------------------------------------------------
plain                60.0%      86.7%      93.3%
contextual           86.7%      86.7%      93.3%
----------------------------------------------------
gain (pp)          +26.7pp      +0.0pp      +0.0pp
----------------------------------------------------
failure drop           67%          0%          0%
```

Conclusion (matches the book): context prefixes lift top-1 recall (60% → 86.7%; failure rate 1−recall@1 down 67%). Gain is strongest at recall@1; `--query` shows how the prefix re-ranks the correct section first.

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
# MOONSHOT_API_KEY / ARK_API_KEY / OPENAI_API_KEY / etc.

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
- Educational project for learning purposes. Acknowledgments: Anthropic engineering research.

