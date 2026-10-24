"""Plain and contextual indexing must share the same chunk boundaries and IDs."""

import pytest

from chunking import DocumentChunker
from config import ChunkingConfig
from contextual_chunking import ContextualChunker


@pytest.mark.parametrize("paragraphs", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "",
        "without punctuation " * 15,
        "First. Second! 尾句没有标点",
        "short\n\n" + "long paragraph " * 30 + "\n\nend",
    ],
)
def test_basic_chunks_match_plain_chunker(paragraphs, text):
    config = ChunkingConfig(
        chunk_size=40,
        chunk_overlap=10,
        min_chunk_size=1,
        max_chunk_size=60,
        respect_paragraph_boundary=paragraphs,
    )
    plain = DocumentChunker(config)
    contextual = ContextualChunker(config, use_contextual=False)
    assert contextual._create_basic_chunks(text, "doc") == plain.chunk_text(text, "doc")
