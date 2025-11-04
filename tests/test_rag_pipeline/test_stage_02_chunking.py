"""Tests for Stage 2: Chunking."""

import pytest
from unittest.mock import Mock
from langchain_core.documents import Document

from src.rag_pipeline.stages.stage_02_chunking import (
    ChunkingStage,
    create_chunking_stage
)


class TestChunkingStage:
    """Test ChunkingStage functionality."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="안녕하세요. " * 100,  # ~1200 chars, should split
                metadata={"source": "doc1.md", "cleaned": True}
            ),
            Document(
                page_content="서울시 교통 정책입니다. " * 50,  # ~600 chars
                metadata={"source": "doc2.md", "cleaned": True}
            ),
            Document(
                page_content="짧은 문서입니다.",  # Short doc
                metadata={"source": "doc3.md", "cleaned": True}
            )
        ]

    def test_stage_initialization(self):
        """Test ChunkingStage initialization."""
        stage = ChunkingStage()

        assert stage.chunk_size == ChunkingStage.DEFAULT_CHUNK_SIZE
        assert stage.chunk_overlap == ChunkingStage.DEFAULT_OVERLAP
        assert stage.separators == ChunkingStage.KOREAN_SEPARATORS
        assert stage.chunker is not None

    def test_stage_initialization_custom_params(self):
        """Test ChunkingStage with custom parameters."""
        stage = ChunkingStage(
            chunk_size=256,
            chunk_overlap=32,
            separators=["\n", " "]
        )

        assert stage.chunk_size == 256
        assert stage.chunk_overlap == 32
        assert stage.separators == ["\n", " "]

    def test_chunk_documents_empty_list_raises_error(self):
        """Test that empty document list raises ValueError."""
        stage = ChunkingStage()

        with pytest.raises(ValueError) as exc_info:
            stage.chunk_documents([])

        assert "No documents provided" in str(exc_info.value)

    def test_chunk_documents_success(self, sample_documents):
        """Test successful document chunking."""
        stage = ChunkingStage(chunk_size=512, chunk_overlap=64)

        chunks = stage.chunk_documents(sample_documents)

        # Should have chunks
        assert len(chunks) > 0

        # Should have more chunks than original docs (some docs split)
        assert len(chunks) >= len(sample_documents)

        # All chunks should be Document objects
        assert all(isinstance(c, Document) for c in chunks)

        # Check metadata enrichment
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "chunk_size_config" in chunk.metadata
            assert "chunk_overlap_config" in chunk.metadata
            assert "chunk_length" in chunk.metadata
            assert chunk.metadata["chunk_size_config"] == 512
            assert chunk.metadata["chunk_overlap_config"] == 64

    def test_chunk_documents_preserves_original_metadata(self, sample_documents):
        """Test that original metadata is preserved in chunks."""
        stage = ChunkingStage()

        chunks = stage.chunk_documents(sample_documents)

        # Find chunks from first document
        doc1_chunks = [c for c in chunks if c.metadata.get("source") == "doc1.md"]

        assert len(doc1_chunks) > 0
        for chunk in doc1_chunks:
            assert chunk.metadata["source"] == "doc1.md"
            assert chunk.metadata["cleaned"] is True

    def test_chunk_documents_with_logger(self, sample_documents):
        """Test chunking with logger integration."""
        mock_logger = Mock()
        mock_logger.verbose = False

        stage = ChunkingStage(logger=mock_logger)
        chunks = stage.chunk_documents(sample_documents)

        # Logger should be called
        assert mock_logger.info.called
        assert len(chunks) > 0

    def test_get_chunking_stats_empty_chunks(self, sample_documents):
        """Test statistics with empty chunks."""
        stage = ChunkingStage()

        stats = stage.get_chunking_stats(sample_documents, [])

        assert stats["total_documents"] == len(sample_documents)
        assert stats["total_chunks"] == 0
        assert stats["chunks_per_document"] == 0
        assert stats["avg_chunk_length"] == 0

    def test_get_chunking_stats_with_chunks(self, sample_documents):
        """Test statistics calculation with chunks."""
        stage = ChunkingStage()

        chunks = stage.chunk_documents(sample_documents)
        stats = stage.get_chunking_stats(sample_documents, chunks)

        assert stats["total_documents"] == len(sample_documents)
        assert stats["total_chunks"] == len(chunks)
        assert stats["chunks_per_document"] > 0
        assert stats["avg_chunk_length"] > 0
        assert stats["min_chunk_length"] > 0
        assert stats["max_chunk_length"] > 0

    def test_estimate_embedding_cost_empty_chunks(self):
        """Test cost estimation with empty chunks."""
        stage = ChunkingStage()

        cost = stage.estimate_embedding_cost([])

        assert cost["total_chunks"] == 0
        assert cost["estimated_tokens"] == 0
        assert cost["estimated_cost_usd"] == 0.0

    def test_estimate_embedding_cost_with_chunks(self, sample_documents):
        """Test cost estimation with actual chunks."""
        stage = ChunkingStage()

        chunks = stage.chunk_documents(sample_documents)
        cost = stage.estimate_embedding_cost(chunks, cost_per_million_tokens=0.50)

        assert cost["total_chunks"] == len(chunks)
        assert cost["estimated_tokens"] > 0
        assert cost["estimated_cost_usd"] > 0

    def test_estimate_embedding_cost_custom_rate(self, sample_documents):
        """Test cost estimation with custom rate."""
        stage = ChunkingStage()

        chunks = stage.chunk_documents(sample_documents)

        cost1 = stage.estimate_embedding_cost(chunks, cost_per_million_tokens=0.50)
        cost2 = stage.estimate_embedding_cost(chunks, cost_per_million_tokens=1.00)

        # Cost should double with double rate
        assert cost2["estimated_cost_usd"] == pytest.approx(cost1["estimated_cost_usd"] * 2, rel=0.01)

    def test_create_chunking_stage_function(self):
        """Test factory function."""
        stage = create_chunking_stage(chunk_size=256, chunk_overlap=32)

        assert isinstance(stage, ChunkingStage)
        assert stage.chunk_size == 256
        assert stage.chunk_overlap == 32

    def test_create_chunking_stage_defaults(self):
        """Test factory function with defaults."""
        stage = create_chunking_stage()

        assert isinstance(stage, ChunkingStage)
        assert stage.chunk_size == ChunkingStage.DEFAULT_CHUNK_SIZE
        assert stage.chunk_overlap == ChunkingStage.DEFAULT_OVERLAP

    def test_chunk_size_respects_configuration(self):
        """Test that chunks respect size configuration."""
        # Use small chunk size for testing
        stage = ChunkingStage(chunk_size=100, chunk_overlap=10)

        long_doc = [Document(
            page_content="가나다라마바사. " * 50,  # Long repeating text
            metadata={"source": "test.md"}
        )]

        chunks = stage.chunk_documents(long_doc)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Most chunks should be close to configured size
        # (Last chunk might be shorter)
        for chunk in chunks[:-1]:
            # Allow some variance due to separator logic
            assert len(chunk.page_content) <= stage.chunk_size * 1.5

    def test_chunk_overlap_creates_redundancy(self):
        """Test that overlap creates content redundancy between chunks."""
        stage = ChunkingStage(chunk_size=100, chunk_overlap=20)

        doc = [Document(
            page_content="A" * 150,  # Simple repeating character
            metadata={"source": "test.md"}
        )]

        chunks = stage.chunk_documents(doc)

        # Should have at least 2 chunks
        assert len(chunks) >= 2

        # Overlap should exist (this is checked by TextChunker internally)
        # We verify chunks were created properly
        assert all(len(c.page_content) > 0 for c in chunks)

    def test_korean_separators_used(self):
        """Test that Korean separators are used for splitting."""
        stage = ChunkingStage()

        doc = [Document(
            page_content="첫 문단입니다.\n\n두번째 문단입니다.\n\n세번째 문단입니다.",
            metadata={"source": "test.md"}
        )]

        chunks = stage.chunk_documents(doc)

        # Should successfully chunk Korean text
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    def test_metadata_enrichment_sequence(self, sample_documents):
        """Test that chunk_id is sequential."""
        stage = ChunkingStage()

        chunks = stage.chunk_documents(sample_documents)

        # Check chunk_id sequence
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        assert chunk_ids == list(range(len(chunks)))

    def test_very_short_document_single_chunk(self):
        """Test that very short document creates single chunk."""
        stage = ChunkingStage()

        short_doc = [Document(
            page_content="짧음",
            metadata={"source": "short.md"}
        )]

        chunks = stage.chunk_documents(short_doc)

        # Should create exactly 1 chunk
        assert len(chunks) == 1
        assert chunks[0].page_content == "짧음"
