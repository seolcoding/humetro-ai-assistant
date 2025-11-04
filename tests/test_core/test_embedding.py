"""Tests for CachedEmbeddingGenerator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.core.embedding import CachedEmbeddingGenerator, create_embedding_generator


class TestCachedEmbeddingGenerator:
    """Test CachedEmbeddingGenerator functionality."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        return tmp_path / "embedding_cache"

    @pytest.fixture
    def mock_embeddings(self):
        """Mock OpenAI embeddings."""
        with patch('src.core.embedding.OpenAIEmbeddings') as mock:
            # Mock embed_documents to return dummy vectors
            mock.return_value.embed_documents.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            # Mock embed_query to return dummy vector
            mock.return_value.embed_query.return_value = [0.7, 0.8, 0.9]
            yield mock

    def test_generator_initialization(self, temp_cache_dir, mock_embeddings):
        """Test generator initialization."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        assert generator.model == "text-embedding-3-small"
        assert generator.batch_size == 100
        assert generator.api_calls == 0
        assert generator.cache_hits == 0

    def test_embed_documents(self, temp_cache_dir, mock_embeddings):
        """Test embedding documents with caching."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        texts = ["테스트 텍스트 1", "테스트 텍스트 2"]

        # First call - should hit API
        embeddings1 = generator.embed_documents(texts)
        assert len(embeddings1) == 2
        assert len(embeddings1[0]) == 3  # Dummy vector dimension
        assert generator.api_calls == 2
        assert generator.cache_hits == 0

        # Second call - should hit cache
        embeddings2 = generator.embed_documents(texts)
        assert embeddings2 == embeddings1
        assert generator.api_calls == 2  # No new API calls
        assert generator.cache_hits == 2

    def test_embed_query(self, temp_cache_dir, mock_embeddings):
        """Test embedding query with caching."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        query = "테스트 쿼리"

        # First call - should hit API
        embedding1 = generator.embed_query(query)
        assert len(embedding1) == 3  # Dummy vector dimension
        assert generator.api_calls == 1
        assert generator.cache_hits == 0

        # Second call - should hit cache
        embedding2 = generator.embed_query(query)
        assert embedding2 == embedding1
        assert generator.api_calls == 1  # No new API calls
        assert generator.cache_hits == 1

    def test_estimate_tokens(self, temp_cache_dir, mock_embeddings):
        """Test token estimation."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        # Test with different lengths
        assert generator._estimate_tokens("test") == 1  # 4 chars / 4
        assert generator._estimate_tokens("test text") == 2  # 9 chars / 4
        assert generator._estimate_tokens("a" * 100) == 25  # 100 / 4

    def test_estimate_cost(self, temp_cache_dir, mock_embeddings):
        """Test cost estimation."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        # Test cost calculation
        # text-embedding-3-small: $0.02 per 1M tokens
        text = "a" * 1000  # ~250 tokens
        cost = generator._estimate_cost(text)
        expected_cost = (250 / 1_000_000) * 0.02
        assert abs(cost - expected_cost) < 0.0001

    def test_get_stats(self, temp_cache_dir, mock_embeddings):
        """Test statistics retrieval."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        texts = ["text1", "text2"]

        # First call
        generator.embed_documents(texts)

        # Second call (cache hit)
        generator.embed_documents(texts)

        stats = generator.get_stats()

        assert stats["api_calls"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_hit_rate"] == 0.5  # 2 hits / 4 total
        assert "total_cached_items" in stats
        assert "total_saved_cost" in stats

    def test_create_embedding_generator_function(self, temp_cache_dir, mock_embeddings):
        """Test create_embedding_generator helper function."""
        generator = create_embedding_generator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir
        )

        assert isinstance(generator, CachedEmbeddingGenerator)
        assert generator.model == "text-embedding-3-small"

    def test_batch_processing(self, temp_cache_dir, mock_embeddings):
        """Test batch processing of embeddings."""
        generator = CachedEmbeddingGenerator(
            model="text-embedding-3-small",
            cache_dir=temp_cache_dir,
            batch_size=2
        )

        # Create 5 texts (will need 3 batches: 2, 2, 1)
        texts = [f"text {i}" for i in range(5)]

        mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 3] * 5

        embeddings = generator.embed_documents(texts)

        assert len(embeddings) == 5
        assert generator.api_calls == 5
