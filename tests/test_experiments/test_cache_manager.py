"""Tests for VectorEmbeddingCache."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.experiments.cache_manager import VectorEmbeddingCache, get_cache


class TestVectorEmbeddingCache:
    """Test VectorEmbeddingCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        assert cache.cache_dir.exists()
        assert cache.hits == 0
        assert cache.misses == 0

    def test_put_and_get(self, temp_cache_dir):
        """Test putting and getting embeddings."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        # Put embedding
        text = "This is a test"
        model = "text-embedding-3-small"
        embedding = [0.1, 0.2, 0.3]

        cache.put(text, model, embedding, tokens=5, cost=0.001)

        # Get embedding
        result = cache.get(text, model)

        assert result == embedding
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        result = cache.get("nonexistent text", "model")

        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_different_models_different_cache(self, temp_cache_dir):
        """Test same text with different models creates different cache entries."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        text = "Same text"
        embedding1 = [0.1, 0.2]
        embedding2 = [0.3, 0.4]

        cache.put(text, "model1", embedding1)
        cache.put(text, "model2", embedding2)

        assert cache.get(text, "model1") == embedding1
        assert cache.get(text, "model2") == embedding2

    def test_batch_operations(self, temp_cache_dir):
        """Test batch get and put operations."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        # Put some embeddings
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1], [0.2], [0.3]]
        model = "test-model"

        cache.put_batch(texts, model, embeddings)

        # Get batch with some cached
        test_texts = ["text1", "text2", "text4"]  # text4 not cached
        results, uncached = cache.get_batch(test_texts, model)

        assert len(results) == 3
        assert results[0] == [0.1]
        assert results[1] == [0.2]
        assert results[2] is None
        assert uncached == ["text4"]

    def test_persistence(self, temp_cache_dir):
        """Test cache persistence across instances."""
        # Create cache and add embedding
        cache1 = VectorEmbeddingCache(cache_dir=temp_cache_dir)
        cache1.put("test", "model", [0.1, 0.2])

        # Create new cache instance
        cache2 = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        # Check embedding is still there
        result = cache2.get("test", "model")
        assert result == [0.1, 0.2]

    def test_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        # Add some data
        cache.put("text1", "model", [0.1], tokens=5, cost=0.001)
        cache.put("text2", "model", [0.2], tokens=5, cost=0.001)

        # Generate hits and misses
        cache.get("text1", "model")  # hit
        cache.get("text1", "model")  # hit
        cache.get("text3", "model")  # miss

        stats = cache.get_stats()

        assert stats["total_items"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3)

    def test_clear(self, temp_cache_dir):
        """Test clearing cache."""
        cache = VectorEmbeddingCache(cache_dir=temp_cache_dir)

        # Add data
        cache.put("test", "model", [0.1])

        # Clear
        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.metadata) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_get_cache_function(self, temp_cache_dir):
        """Test get_cache helper function."""
        cache = get_cache(cache_dir=temp_cache_dir)

        assert isinstance(cache, VectorEmbeddingCache)
