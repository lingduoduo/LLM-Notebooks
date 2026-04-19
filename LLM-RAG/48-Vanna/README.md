# 48-Vanna

RAG as a design pattern for text-to-SQL, demonstrated with [Vanna](https://github.com/vanna-ai/vanna).

The notebook (`48.ipynb`) covers the full theory — RAG paradigm, text2SQL challenges, Vanna's architecture, source code walkthrough, and broader RAG use cases. The script (`product_catalog_rag.py`) is the runnable implementation using a small SQLite product catalog with two tables: `products` and `categories`.

## Run

Notebook:

```bash
cd LLM-RAG/48-Vanna
jupyter notebook 48.ipynb
```

Script:

```bash
cd LLM-RAG/48-Vanna
python -m pip install -e .
OPENAI_API_KEY=... python product_catalog_rag.py
```

Optional environment variables:

- `OPENAI_API_KEY`: required.
- `OPENAI_MODEL`: OpenAI model name. Defaults to `gpt-4o`.
- `PRODUCT_CATALOG_DB_PATH`: SQLite path. Defaults to `product_catalog.sqlite`.
- `VANNA_CHROMA_PATH`: Chroma vector-store path. Defaults to `.chroma-products`.

## Code Structure

`product_catalog_rag.py` has three layers:

**Data layer**
- `create_product_catalog()`: creates the SQLite schema and seeds fixture data.
- `build_vanna()`: initialises a `ProductCatalogVanna` instance (ChromaDB + OpenAI) and connects it to the database.
- `build_product_catalog_rag()`: wires data layer and RAG wrapper together.

**RAG layer — `ProductCatalogRAG`**
- `train()` → `train_product_catalog()`: loads DDL, business documentation, and NL/SQL examples into the Chroma vector store.
- `get_retrieval_context()`: fetches related DDL, docs, and SQL examples for a question.
- `generate_sql()`: retrieves context, prints a prompt preview via `build_sql_prompt()`, then delegates generation to the Vanna engine.
- `run_sql()` / `ask()`: executes the generated SQL and displays the result.

**Prompt layer**
- `build_sql_prompt()`: composes the RAG prompt from retrieved DDL, documentation, and SQL examples.
- `_format_section()`: formats each retrieved item for the prompt.

**Entry points**
- `run_use_cases(rag)`: runs all `USE_CASES` questions through `rag.ask()`.
- `main()`: end-to-end pipeline — setup, train, run.

## Generated Files

The demo creates all data from code; no checked-in binaries are needed.

Keep these out of git (already in `.gitignore`):

- `product_catalog.sqlite`
- `.chroma-products/`
- `chroma.sqlite3`
