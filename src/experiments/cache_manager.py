"""
Vector embedding cache manager.

This module provides content-hash based caching for vector embeddings
to avoid duplicate OpenAI API calls and reduce costs.
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


class VectorEmbeddingCache:
    """
    Content-hash based cache for vector embeddings.

    Prevents duplicate OpenAI API calls by caching embeddings
    based on content hash and model name.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory for cache storage (default: ./data/embeddings_cache)
        """
        if cache_dir is None:
            from src.common.project_paths import get_project_root
            cache_dir = get_project_root() / "data" / "embeddings_cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file paths
        self.embeddings_file = cache_dir / "embeddings.pkl"
        self.metadata_file = cache_dir / "metadata.json"
        self.stats_file = cache_dir / "stats.json"

        # Load existing cache
        self.cache: Dict[str, List[float]] = self._load_cache()
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_savings = 0.0  # Estimated cost savings in USD

    def _generate_key(self, text: str, model: str) -> str:
        """
        Generate cache key from text and model.

        Args:
            text: Text content
            model: Model name

        Returns:
            Cache key (hash)
        """
        content = f"{model}::{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load cache: {e}")
                return {}
        return {}

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load metadata: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"❌ Error saving cache: {e}")

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """
        Get cached embedding.

        Args:
            text: Text content
            model: Model name

        Returns:
            Cached embedding vector or None if not found
        """
        key = self._generate_key(text, model)

        if key in self.cache:
            self.hits += 1

            # Update access time in metadata
            if key in self.metadata:
                self.metadata[key]["last_accessed"] = datetime.now().isoformat()
                self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1

            return self.cache[key]

        self.misses += 1
        return None

    def put(
        self,
        text: str,
        model: str,
        embedding: List[float],
        tokens: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """
        Cache embedding.

        Args:
            text: Text content
            model: Model name
            embedding: Embedding vector
            tokens: Number of tokens (for statistics)
            cost: API call cost in USD (for savings tracking)
        """
        key = self._generate_key(text, model)

        # Store embedding
        self.cache[key] = embedding

        # Store metadata
        self.metadata[key] = {
            "model": model,
            "text_length": len(text),
            "tokens": tokens,
            "cost": cost,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 1
        }

        # Save to disk
        self._save_cache()
        self._save_metadata()

    def get_batch(
        self,
        texts: List[str],
        model: str
    ) -> Tuple[List[Optional[List[float]]], List[str]]:
        """
        Get batch of embeddings from cache.

        Args:
            texts: List of text contents
            model: Model name

        Returns:
            Tuple of (embeddings, uncached_texts)
            - embeddings: List with cached embeddings (None for uncached)
            - uncached_texts: List of texts that need embedding
        """
        embeddings = []
        uncached_texts = []

        for text in texts:
            embedding = self.get(text, model)
            embeddings.append(embedding)
            if embedding is None:
                uncached_texts.append(text)

        return embeddings, uncached_texts

    def put_batch(
        self,
        texts: List[str],
        model: str,
        embeddings: List[List[float]],
        tokens_list: Optional[List[int]] = None,
        costs_list: Optional[List[float]] = None
    ):
        """
        Cache batch of embeddings.

        Args:
            texts: List of text contents
            model: Model name
            embeddings: List of embedding vectors
            tokens_list: List of token counts
            costs_list: List of API call costs
        """
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            tokens = tokens_list[i] if tokens_list else None
            cost = costs_list[i] if costs_list else None
            self.put(text, model, embedding, tokens, cost)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        # Calculate total cached items and savings
        total_items = len(self.cache)
        total_saved_cost = sum(
            meta.get("cost", 0) * (meta.get("access_count", 1) - 1)
            for meta in self.metadata.values()
        )

        stats = {
            "total_items": total_items,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": hit_rate,
            "total_saved_cost": total_saved_cost,
            "cache_size_mb": self.embeddings_file.stat().st_size / (1024 * 1024) if self.embeddings_file.exists() else 0
        }

        return stats

    def save_stats(self):
        """Save statistics to disk."""
        stats = self.get_stats()
        stats["timestamp"] = datetime.now().isoformat()

        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving stats: {e}")

    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
        self.metadata.clear()
        self.hits = 0
        self.misses = 0
        self.total_savings = 0.0

        # Remove cache files
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        if self.stats_file.exists():
            self.stats_file.unlink()

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"EMBEDDING CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Total cached items: {stats['total_items']:,}")
        print(f"Cache hits: {stats['cache_hits']:,}")
        print(f"Cache misses: {stats['cache_misses']:,}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Total saved cost: ${stats['total_saved_cost']:.4f}")
        print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        print(f"{'='*60}\n")


def get_cache(cache_dir: Optional[Path] = None) -> VectorEmbeddingCache:
    """
    Get or create embedding cache.

    Args:
        cache_dir: Directory for cache storage

    Returns:
        VectorEmbeddingCache instance
    """
    return VectorEmbeddingCache(cache_dir)
