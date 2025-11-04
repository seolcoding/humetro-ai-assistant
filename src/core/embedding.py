"""
Embedding generation module with integrated caching.

This module provides OpenAI embedding generation with automatic caching
to avoid duplicate API calls and reduce costs.
"""

from typing import List, Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.experiments.cache_manager import VectorEmbeddingCache
from src.common.logger import RAGLogger


class CachedEmbeddingGenerator:
    """
    OpenAI embedding generator with integrated caching.

    Automatically caches embeddings to avoid duplicate API calls,
    significantly reducing costs for repeated operations.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[Path] = None,
        logger: Optional[RAGLogger] = None,
        batch_size: int = 100,
    ):
        """
        Initialize embedding generator with cache.

        Args:
            model: OpenAI embedding model name
            cache_dir: Directory for embedding cache
            logger: Optional RAG logger for tracking
            batch_size: Batch size for API calls
        """
        self.model = model
        self.batch_size = batch_size

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model=model)

        # Initialize cache
        self.cache = VectorEmbeddingCache(cache_dir=cache_dir)

        # Logger for API call tracking
        self.logger = logger

        # Statistics
        self.api_calls = 0
        self.cache_hits = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents with caching.

        Args:
            texts: List of text contents to embed

        Returns:
            List of embedding vectors
        """
        # Check cache for all texts
        cached_embeddings, uncached_texts = self.cache.get_batch(texts, self.model)

        # Track cache hits
        cache_hits = sum(1 for emb in cached_embeddings if emb is not None)
        self.cache_hits += cache_hits

        # Generate embeddings for uncached texts only
        if uncached_texts:
            new_embeddings = self._generate_embeddings(uncached_texts)

            # Cache new embeddings
            self.cache.put_batch(
                uncached_texts,
                self.model,
                new_embeddings,
                tokens_list=[self._estimate_tokens(text) for text in uncached_texts],
                costs_list=[self._estimate_cost(text) for text in uncached_texts],
            )

            # Merge cached and new embeddings
            new_emb_idx = 0
            for i, embedding in enumerate(cached_embeddings):
                if embedding is None:
                    cached_embeddings[i] = new_embeddings[new_emb_idx]
                    new_emb_idx += 1

        # Log API calls
        if self.logger and uncached_texts:
            total_tokens = sum(self._estimate_tokens(text) for text in uncached_texts)
            total_cost = sum(self._estimate_cost(text) for text in uncached_texts)

            self.logger.log_api_call(
                provider="openai",
                model=self.model,
                operation="embedding",
                tokens=total_tokens,
                cost=total_cost,
                cached=False,
            )

        return cached_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query with caching.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        cached = self.cache.get(text, self.model)

        if cached is not None:
            self.cache_hits += 1
            return cached

        # Generate new embedding
        embedding = self.embeddings.embed_query(text)
        self.api_calls += 1

        # Cache it
        self.cache.put(
            text,
            self.model,
            embedding,
            tokens=self._estimate_tokens(text),
            cost=self._estimate_cost(text),
        )

        # Log API call
        if self.logger:
            self.logger.log_api_call(
                provider="openai",
                model=self.model,
                operation="embedding",
                tokens=self._estimate_tokens(text),
                cost=self._estimate_cost(text),
                cached=False,
            )

        return embedding

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings via OpenAI API in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            self.api_calls += len(batch)

        return embeddings

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text content

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for Korean/English mix
        return len(text) // 4

    def _estimate_cost(self, text: str) -> float:
        """
        Estimate cost for embedding text.

        Args:
            text: Text content

        Returns:
            Estimated cost in USD
        """
        tokens = self._estimate_tokens(text)

        # text-embedding-3-small: $0.02 per 1M tokens
        return (tokens / 1_000_000) * 0.02

    def get_stats(self) -> dict:
        """
        Get embedding generation statistics.

        Returns:
            Dictionary with statistics
        """
        cache_stats = self.cache.get_stats()

        total_requests = self.api_calls + self.cache_hits
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "total_cached_items": cache_stats["total_items"],
            "total_saved_cost": cache_stats["total_saved_cost"],
        }

    def print_stats(self):
        """Print embedding statistics."""
        stats = self.get_stats()

        print(f"\n{'=' * 60}")
        print("EMBEDDING STATISTICS")
        print(f"{'=' * 60}")
        print(f"API calls: {stats['api_calls']:,}")
        print(f"Cache hits: {stats['cache_hits']:,}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"Total cached items: {stats['total_cached_items']:,}")
        print(f"Total saved cost: ${stats['total_saved_cost']:.4f}")
        print(f"{'=' * 60}\n")


def create_embedding_generator(
    model: str = "text-embedding-3-small",
    cache_dir: Optional[Path] = None,
    logger: Optional[RAGLogger] = None,
) -> CachedEmbeddingGenerator:
    """
    Create embedding generator with cache.

    Args:
        model: OpenAI embedding model name
        cache_dir: Directory for embedding cache
        logger: Optional RAG logger

    Returns:
        CachedEmbeddingGenerator instance
    """
    return CachedEmbeddingGenerator(model=model, cache_dir=cache_dir, logger=logger)
