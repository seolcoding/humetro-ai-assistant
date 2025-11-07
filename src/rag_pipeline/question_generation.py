"""
Question Generation Module
==========================

Centralized question generation with caching support.
Provides clean API for benchmark scripts to generate test questions.

Usage:
    from question_generation import generate_questions, load_cached_questions

    # Generate new questions
    config, questions_df = generate_questions(
        model="gpt-4o-mini",
        num_documents=50,
        num_questions=50
    )

    # Load cached questions
    questions_df = load_cached_questions(cache_key="abc123")
"""

import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path for testset_generator import
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline.testset_generator import CachedTestsetGenerator, TestsetConfig


def generate_questions(
    model: str = "gpt-4o-mini",
    source: str = "data/crawled/seoul_traffic/markdown_filtered",
    num_documents: int = 50,
    num_questions: int = 50,
    cache_id: Optional[str] = None,
    use_cache: bool = True,
    force_regenerate: bool = False,
    use_korean_personas: bool = True,
    temperature: float = 0.3,
    embedding_model: str = "text-embedding-3-small",
    language: str = "korean",
    is_latest: bool = True,
    is_benchmark: bool = True,
    description: Optional[str] = None,
    **kwargs
) -> Tuple[TestsetConfig, pd.DataFrame]:
    """
    Generate test questions with automatic caching.

    Args:
        model: LLM model for question generation (e.g., "gpt-4o-mini", "gpt-4o")
        source: Document source directory path
        num_documents: Number of documents to use for generation
        num_questions: Number of questions to generate (testset size)
        cache_id: Optional manual cache identifier (auto-generated if None)
        use_cache: Whether to use cached results if available
        force_regenerate: Force regeneration ignoring cache
        use_korean_personas: Use Korean-specific personas for generation
        temperature: LLM temperature for generation (0.0-1.0)
        embedding_model: Embedding model for document retrieval
        language: Question language (default: "korean")
        is_latest: Mark as latest version
        is_benchmark: Mark as benchmark dataset
        description: Custom description for the testset
        **kwargs: Additional config passed to generator

    Returns:
        Tuple of (TestsetConfig, DataFrame):
            - config: Configuration object with cache_key and metadata
            - questions_df: DataFrame with columns [user_input, reference, reference_contexts, ...]

    Example:
        >>> config, df = generate_questions(
        ...     model="gpt-4o-mini",
        ...     num_documents=50,
        ...     num_questions=50,
        ...     description="Benchmark test for Seoul metro QA"
        ... )
        >>> print(f"Generated {len(df)} questions")
        >>> print(f"Cache key: {config.get_cache_key()}")
    """
    generator = CachedTestsetGenerator()

    # Auto-generate description if not provided
    if description is None:
        description = (
            f"Generated {num_questions} questions from {num_documents} documents "
            f"(filtered 2021-2025 + timeless content) using {model}"
        )

    # Generate or load from cache
    config, testset_df = generator.generate_or_load(
        llm_model=model,
        llm_temperature=temperature,
        embedding_model=embedding_model,
        document_source=source,
        num_documents=num_documents,
        testset_size=num_questions,
        language=language,
        force_regenerate=force_regenerate or not use_cache,
        use_korean_personas=use_korean_personas,
        is_latest=is_latest,
        is_benchmark=is_benchmark,
        description=description,
        **kwargs
    )

    return config, testset_df


def load_cached_questions(cache_key: str) -> pd.DataFrame:
    """
    Load previously generated questions from cache.

    Args:
        cache_key: Cache identifier (hash from config)

    Returns:
        DataFrame with cached questions

    Raises:
        FileNotFoundError: If cache key doesn't exist

    Example:
        >>> df = load_cached_questions("abc123def456")
        >>> print(f"Loaded {len(df)} cached questions")
    """
    cache_dir = Path("data/evaluation/testsets")
    cache_file = cache_dir / f"testset_{cache_key}.csv"

    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cached questions found for key: {cache_key}\n"
            f"Expected file: {cache_file}"
        )

    return pd.read_csv(cache_file)


def list_cached_questions() -> List[Dict[str, Any]]:
    """
    List all cached question sets with metadata.

    Returns:
        List of dictionaries with cache information:
        [
            {
                "cache_key": "abc123",
                "size": 50,
                "created": "2025-11-05T12:00:00",
                "model": "gpt-4o-mini",
                "num_documents": 50,
                "description": "..."
            },
            ...
        ]

    Example:
        >>> cached = list_cached_questions()
        >>> for item in cached:
        ...     print(f"{item['cache_key']}: {item['size']} questions")
    """
    cache_dir = Path("data/evaluation/testsets")

    if not cache_dir.exists():
        return []

    cached_sets = []

    # Find all JSON metadata files
    for json_file in sorted(cache_dir.glob("testset_*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            cache_key = json_file.stem.replace("testset_", "")
            config = metadata.get("config", {})

            cached_sets.append({
                "cache_key": cache_key,
                "size": metadata.get("size", 0),
                "created": metadata.get("created_at", "unknown"),
                "model": config.get("llm_model", "unknown"),
                "num_documents": config.get("num_documents", 0),
                "description": config.get("description", ""),
                "file_path": str(json_file)
            })

        except (json.JSONDecodeError, IOError) as e:
            # Skip corrupted files
            continue

    return cached_sets


def get_cache_info(cache_key: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a cached question set.

    Args:
        cache_key: Cache identifier

    Returns:
        Dictionary with complete metadata

    Raises:
        FileNotFoundError: If cache key doesn't exist

    Example:
        >>> info = get_cache_info("abc123")
        >>> print(info['config']['llm_model'])
        gpt-4o-mini
    """
    cache_dir = Path("data/evaluation/testsets")
    json_file = cache_dir / f"testset_{cache_key}.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"No metadata found for cache key: {cache_key}\n"
            f"Expected file: {json_file}"
        )

    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def delete_cached_questions(cache_key: str) -> bool:
    """
    Delete a cached question set (all associated files).

    Args:
        cache_key: Cache identifier

    Returns:
        True if deleted, False if not found

    Example:
        >>> if delete_cached_questions("old_cache_key"):
        ...     print("Cache deleted")
    """
    cache_dir = Path("data/evaluation/testsets")
    deleted = False

    # Delete all files with this cache key
    for ext in ['.json', '.csv', '.md']:
        cache_file = cache_dir / f"testset_{cache_key}{ext}"
        if cache_file.exists():
            cache_file.unlink()
            deleted = True

    return deleted


def generate_50q_benchmark(
    model: str = "gpt-4o-mini",
    num_documents: int = 50,
    force: bool = False
) -> Tuple[TestsetConfig, pd.DataFrame]:
    """
    Convenience function for 50-question benchmark generation.

    Args:
        model: LLM model to use
        num_documents: Number of documents
        force: Force regeneration

    Returns:
        (config, questions_df)
    """
    return generate_questions(
        model=model,
        num_documents=num_documents,
        num_questions=50,
        force_regenerate=force,
        description=f"Korean persona benchmark (50Q, {num_documents} docs, 2021-2025 filtered)"
    )


def generate_100q_benchmark(
    model: str = "gpt-4o-mini",
    num_documents: int = 100,
    force: bool = False
) -> Tuple[TestsetConfig, pd.DataFrame]:
    """
    Convenience function for 100-question benchmark generation.

    Args:
        model: LLM model to use
        num_documents: Number of documents
        force: Force regeneration

    Returns:
        (config, questions_df)
    """
    return generate_questions(
        model=model,
        num_documents=num_documents,
        num_questions=100,
        force_regenerate=force,
        description=f"100-question benchmark ({num_documents} docs, 2021-2025 filtered)"
    )


# Convenience exports
__all__ = [
    'generate_questions',
    'load_cached_questions',
    'list_cached_questions',
    'get_cache_info',
    'delete_cached_questions',
    'generate_50q_benchmark',
    'generate_100q_benchmark',
]
