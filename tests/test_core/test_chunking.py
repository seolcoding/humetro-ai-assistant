"""Tests for TextChunker."""

import pytest
from langchain.schema import Document

from src.core.chunking import TextChunker, create_chunker


class TestTextChunker:
    """Test TextChunker functionality."""

    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_chunk_documents(self):
        """Test chunking documents."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        documents = [
            Document(page_content="안녕하세요. " * 50, metadata={"source": "test"}),
            Document(page_content="반갑습니다. " * 50, metadata={"source": "test2"})
        ]

        chunks = chunker.chunk_documents(documents)

        assert len(chunks) > len(documents)  # Should split into multiple chunks
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_chunk_text(self):
        """Test chunking text."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        text = "안녕하세요. " * 100
        chunks = chunker.chunk_text(text, metadata={"source": "test"})

        assert len(chunks) > 1
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)

    def test_get_chunk_stats(self):
        """Test chunk statistics."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)

        documents = [
            Document(page_content="안녕하세요. " * 50)
        ]

        chunks = chunker.chunk_documents(documents)
        stats = chunker.get_chunk_stats(chunks)

        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_length"] > 0
        assert stats["min_length"] > 0
        assert stats["max_length"] > 0

    def test_create_chunker_function(self):
        """Test create_chunker helper function."""
        chunker = create_chunker(chunk_size=1024, chunk_overlap=256)

        assert isinstance(chunker, TextChunker)
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 256
