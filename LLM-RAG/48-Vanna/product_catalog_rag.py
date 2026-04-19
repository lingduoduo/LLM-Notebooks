"""Product catalog text-to-SQL demo with Vanna RAG.

This script mirrors the notebook use case with two tables, `products` and
`categories`, then trains Vanna with schema, documentation, and example SQL.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent


def resolve_env_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


DB_PATH = resolve_env_path("PRODUCT_CATALOG_DB_PATH", "product_catalog.sqlite")
CHROMA_PATH = resolve_env_path("VANNA_CHROMA_PATH", ".chroma-products")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
USE_CASES = [
    "Which products are in the phone category?",
    "What is the total inventory for active products in each category?",
    "Find products priced below 3000 that are in stock",
    "What is the average active product price by category?",
    "Which active products are out of stock?",
]
CATALOG_DOCUMENTATION = """
The product catalog has two tables: categories and products.
categories.name_zh stores Chinese category names such as 手机, 笔记本电脑, 音频, and 配件.
products.category_id joins to categories.category_id.
Inventory or stock refers to products.stock_quantity.
Price refers to products.price.
When asking for currently listed products, filter with products.is_active = 1.
"""
QUESTION_SQL_EXAMPLES = [
    {
        "question": "Which products are in the phone category?",
        "sql": """
        SELECT p.name, p.sku, p.price, p.stock_quantity
        FROM products AS p
        JOIN categories AS c ON p.category_id = c.category_id
        WHERE c.name_zh = '手机'
          AND p.is_active = 1
        ORDER BY p.price DESC
        """,
    },
    {
        "question": "What is the total inventory for active products in each category?",
        "sql": """
        SELECT c.name_zh AS category_name, SUM(p.stock_quantity) AS total_stock
        FROM categories AS c
        JOIN products AS p ON p.category_id = c.category_id
        WHERE p.is_active = 1
        GROUP BY c.category_id, c.name_zh
        ORDER BY total_stock DESC
        """,
    },
    {
        "question": "Find active products priced below 3000 that are in stock",
        "sql": """
        SELECT p.name, c.name_zh AS category_name, p.price, p.stock_quantity
        FROM products AS p
        JOIN categories AS c ON p.category_id = c.category_id
        WHERE p.price < 3000
          AND p.stock_quantity > 0
          AND p.is_active = 1
        ORDER BY p.price ASC
        """,
    },
]
CATEGORIES = [
    (1, "phone", "手机", "Smartphones and mobile devices"),
    (2, "laptop", "笔记本电脑", "Portable computers for work and gaming"),
    (3, "audio", "音频", "Headphones, speakers, and audio accessories"),
    (4, "accessory", "配件", "Chargers, cases, cables, and adapters"),
]
PRODUCTS = [
    (101, 1, "iPhone 15", "PHN-IP15-128", 5999.0, 18, 1),
    (102, 1, "Galaxy S24", "PHN-GS24-256", 5499.0, 12, 1),
    (103, 1, "Pixel 8", "PHN-PX8-128", 4299.0, 0, 1),
    (201, 2, "ThinkPad X1 Carbon", "LAP-X1C-001", 10999.0, 6, 1),
    (202, 2, "MacBook Air 13", "LAP-MBA13-001", 7999.0, 9, 1),
    (301, 3, "AirPods Pro", "AUD-APP-002", 1899.0, 25, 1),
    (302, 3, "Bluetooth Speaker Mini", "AUD-SPK-MINI", 399.0, 40, 1),
    (401, 4, "USB-C Charger 65W", "ACC-CHG-65W", 199.0, 60, 1),
    (402, 4, "Phone Case Clear", "ACC-CASE-CLR", 79.0, 100, 1),
    (403, 4, "Legacy Cable", "ACC-CBL-OLD", 39.0, 8, 0),
]


class ProductCatalogRAG:
    """Workflow wrapper over a Vanna vector-store/LLM engine."""

    def __init__(self, engine):
        self.engine = engine

    def train(self) -> None:
        train_product_catalog(self.engine)

    def get_retrieval_context(self, question: str) -> dict[str, list]:
        return {
            "ddl": self.engine.get_related_ddl(question),
            "documentation": self.engine.get_related_documentation(question),
            "examples": self.engine.get_similar_question_sql(question),
        }

    def generate_sql(self, question: str) -> str:
        """Retrieve RAG context, print a prompt preview, then generate SQL."""
        context = self.get_retrieval_context(question)
        prompt = build_sql_prompt(
            question=question,
            ddl_list=context["ddl"],
            sql_list=context["examples"],
            doc_list=context["documentation"],
        )
        print(
            f"Retrieved: {len(context['ddl'])} DDL, "
            f"{len(context['documentation'])} docs, "
            f"{len(context['examples'])} SQL examples"
        )
        print("\nPrompt preview:\n", prompt)
        return self.engine.generate_sql(question)

    def run_sql(self, sql: str):
        run = getattr(self.engine, "run_sql", None)
        return run(sql) if run is not None else None

    def ask(self, question: str) -> tuple[str, Any]:
        print("=" * 80)
        print(f"Question: {question}")
        sql = self.generate_sql(question)
        print("\nGenerated SQL:\n", sql)
        result = self.run_sql(sql)
        if result is None:
            return sql, None
        print("\nResult:")
        display_result(result)
        return sql, result


def configure_runtime() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")


def build_sql_prompt(
    question: str,
    ddl_list: Sequence,
    sql_list: Sequence,
    doc_list: Sequence,
) -> str:
    """Compose a readable SQL prompt from retrieved DDL, examples, and docs."""
    sections = [
        "You are a SQLite expert. Generate SQL using only the provided context.",
        "\n# Question",
        question.strip(),
        "\n# Related DDL",
        _format_section(ddl_list),
        "\n# Related Documentation",
        _format_section(doc_list),
        "\n# Similar Question/SQL Examples",
        _format_section(sql_list),
        "\n# Instructions",
        (
            "Return one executable SQLite query. Prefer active products with "
            "is_active = 1 when the question asks for listed/current products."
        ),
    ]
    return "\n".join(sections)


def _format_section(items: Sequence) -> str:
    if not items:
        return "(none retrieved)"

    def _render(item) -> str:
        if isinstance(item, dict):
            return json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True)
        return str(item).strip()

    return "\n\n".join(f"[{i}] {_render(item)}" for i, item in enumerate(items, start=1))


def create_product_catalog(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                name_en TEXT NOT NULL UNIQUE,
                name_zh TEXT NOT NULL UNIQUE,
                description TEXT
            );

            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                category_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                sku TEXT NOT NULL UNIQUE,
                price REAL NOT NULL,
                stock_quantity INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            );
            """
        )
        conn.executemany("INSERT INTO categories VALUES (?, ?, ?, ?)", CATEGORIES)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", PRODUCTS)
    return db_path


def build_vanna(db_path: Path = DB_PATH, chroma_path: Path = CHROMA_PATH):
    from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
    from vanna.openai.openai_chat import OpenAI_Chat

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY before running the Vanna RAG example.")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    chroma_path.mkdir(parents=True, exist_ok=True)

    class ProductCatalogVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    vn = ProductCatalogVanna(
        config={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": DEFAULT_MODEL,
            "path": str(chroma_path),
        }
    )
    vn.connect_to_sqlite(str(db_path))
    return vn


def build_product_catalog_rag(
    db_path: Path = DB_PATH,
    chroma_path: Path = CHROMA_PATH,
) -> ProductCatalogRAG:
    return ProductCatalogRAG(build_vanna(db_path, chroma_path))


def train_product_catalog(vn) -> None:
    ddl_df = vn.run_sql(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL ORDER BY name"
    )
    for ddl in ddl_df["sql"].dropna():
        vn.train(ddl=ddl.strip())
    vn.train(documentation=CATALOG_DOCUMENTATION.strip())
    for example in QUESTION_SQL_EXAMPLES:
        vn.train(question=example["question"].strip(), sql=example["sql"].strip())


def display_result(result) -> None:
    print(result)


def run_use_cases(
    rag: ProductCatalogRAG,
    use_cases: Sequence[str] | None = None,
) -> dict[str, tuple[str, Any]]:
    questions = use_cases or USE_CASES
    return {q: rag.ask(q) for q in questions}


def main() -> None:
    configure_runtime()
    create_product_catalog(DB_PATH)
    rag = build_product_catalog_rag(DB_PATH, CHROMA_PATH)
    rag.train()
    print("Vanna RAG training complete.")
    run_use_cases(rag)


if __name__ == "__main__":
    main()
