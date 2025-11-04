"""
Vector store management module.

This module provides vector store creation and management with
FAISS backend and caching strategy.
"""

from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.common.logger import RAGLogger


class VectorStoreManager:
    """
    Manages FAISS vector store creation and caching.

    Provides automatic caching to avoid regenerating vector stores
    for the same documents.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: Path,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize vector store manager.

        Args:
            embeddings: Embedding function (should be CachedEmbeddingGenerator)
            persist_directory: Directory for vector store persistence
            logger: Optional RAG logger
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.logger = logger

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # FAISS index file
        self.index_file = persist_directory / "index.faiss"
        self.docstore_file = persist_directory / "index.pkl"

    def _cache_exists(self) -> bool:
        """
        Check if cached vector store exists.

        Returns:
            True if cache exists
        """
        return self.index_file.exists() and self.docstore_file.exists()

    def get_or_create_vectorstore(
        self,
        documents: Optional[List[Document]] = None
    ) -> FAISS:
        """
        Load cached vector store or create new one.

        Args:
            documents: Documents to create vector store from (required if no cache)

        Returns:
            FAISS vector store instance

        Raises:
            ValueError: If no cache exists and no documents provided
        """
        if self._cache_exists():
            if self.logger:
                self.logger.info(f"Loading cached vector store from {self.persist_directory}")

            return self._load_cached()

        elif documents:
            if self.logger:
                self.logger.info(f"Creating new vector store with {len(documents)} documents")

            return self._create_new(documents)

        else:
            raise ValueError(
                "No cached vector store found and no documents provided. "
                f"Expected cache at: {self.persist_directory}"
            )

    def _load_cached(self) -> FAISS:
        """
        Load cached FAISS vector store.

        Returns:
            FAISS vector store instance
        """
        vectorstore = FAISS.load_local(
            str(self.persist_directory),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        if self.logger:
            # Get vector count
            try:
                vector_count = vectorstore.index.ntotal
                self.logger.info(f"Loaded {vector_count:,} vectors from cache")
            except:
                self.logger.info("Vector store loaded from cache")

        return vectorstore

    def _create_new(self, documents: List[Document]) -> FAISS:
        """
        Create new FAISS vector store.

        Args:
            documents: Documents to create vector store from

        Returns:
            FAISS vector store instance
        """
        if self.logger:
            self.logger.info(f"Generating embeddings for {len(documents)} documents...")

        # Create FAISS index
        vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )

        # Save to disk
        if self.logger:
            self.logger.info(f"Saving vector store to {self.persist_directory}")

        vectorstore.save_local(str(self.persist_directory))

        if self.logger:
            self.logger.info("Vector store created and saved successfully")

        return vectorstore

    def delete_cache(self):
        """Delete cached vector store."""
        if self.index_file.exists():
            self.index_file.unlink()
        if self.docstore_file.exists():
            self.docstore_file.unlink()

        if self.logger:
            self.logger.info(f"Deleted cache at {self.persist_directory}")

    def get_cache_size(self) -> dict:
        """
        Get cache size information.

        Returns:
            Dictionary with cache size info
        """
        if not self._cache_exists():
            return {
                "exists": False,
                "size_mb": 0,
                "index_size_mb": 0,
                "docstore_size_mb": 0
            }

        index_size = self.index_file.stat().st_size / (1024 * 1024)
        docstore_size = self.docstore_file.stat().st_size / (1024 * 1024)
        total_size = index_size + docstore_size

        return {
            "exists": True,
            "size_mb": total_size,
            "index_size_mb": index_size,
            "docstore_size_mb": docstore_size
        }


def create_vector_store_manager(
    embeddings: Embeddings,
    persist_directory: Path,
    logger: Optional[RAGLogger] = None
) -> VectorStoreManager:
    """
    Create vector store manager.

    Args:
        embeddings: Embedding function
        persist_directory: Directory for vector store persistence
        logger: Optional RAG logger

    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        embeddings=embeddings,
        persist_directory=persist_directory,
        logger=logger
    )
