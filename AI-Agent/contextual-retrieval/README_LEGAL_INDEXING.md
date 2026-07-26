# Contextual Legal Document Indexing

This script implements Anthropic's Contextual Retrieval approach for indexing Chinese legal documents.

## Key Innovation: Contextual Retrieval

Unlike traditional RAG that loses context when chunking, this script:
1. Generates contextual descriptions for each chunk using LLM
2. Prepends context to chunks before indexing
3. Significantly improves retrieval accuracy

## Features

- **Contextual Enhancement**: Uses LLM to generate chunk-specific context
- **Smart Chunking**: Paragraph-aware boundaries (soft: 1024, hard: 2048 chars)
- **Comparison Mode**: Run with/without context for performance comparison
- **Cache Optimization**: Caches context for similar chunks to reduce API costs
- **Detailed Statistics**: Token usage, generation time, and cost estimation

## Prerequisites

1. Set up your LLM API key:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   # LLM_MODEL defaults to gpt-5.6-terra; set it to override the OpenAI model.
   ```

2. Ensure retrieval pipeline is running:
   ```bash
   # Terminal 1: Dense service
   python dense_service.py
   
   # Terminal 2: Sparse service
   python sparse_service.py
   
   # Terminal 3: Main pipeline
   python main.py
   ```

3. The `laws` directory should be linked/present (automatically created as symlink to agentic-rag/laws)

## Usage

### Basic Contextual Indexing
```bash
# Index with contextual enhancement (default)
python index_local_laws_contextual.py
```

### Advanced Options
```bash
# Process limited documents
python index_local_laws_contextual.py --max-docs 10

# Process specific categories
python index_local_laws_contextual.py --categories "constitution" "civil_code"

# Override the OpenAI model
python index_local_laws_contextual.py --llm-model gpt-5.6-terra

# Custom batch size for indexing
python index_local_laws_contextual.py --batch-size 20

# Skip cleanup
python index_local_laws_contextual.py --no-cleanup
```

## Cost Considerations

Context generation requires LLM API calls:
- ~150 tokens per chunk for context generation
- Cache reduces costs for duplicate content

Estimate for 288 legal documents:
- ~3000-5000 chunks total
- ~450K-750K tokens
- Estimate token usage against the selected OpenAI model's current pricing

## Document Store

Maintains `document_store.json` with:
- Document metadata
- Chunk statistics
- Context token usage
- Generation metrics
- Indexing timestamps
