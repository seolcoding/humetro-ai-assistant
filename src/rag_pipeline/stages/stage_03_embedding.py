"""
Stage 3: Embedding

Generates embeddings for chunked documents and creates FAISS vector index.
Uses OpenAI embeddings with FAISS for efficient similarity search.
Uses Cosine Similarity for semantic similarity matching.
"""

from typing import List, Optional
from pathlib import Path
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.common.logger import RAGLogger


class EmbeddingStage:
    """
    Embedding Stage for RAG Pipeline.

    Generates embeddings using OpenAI and creates FAISS vector store
    for efficient similarity search.

    Features:
    - OpenAI text-embedding-3-small (best price/performance)
    - FAISS vector store (no caching, fresh index each time)
    - Batch processing for efficiency
    - Cost tracking and statistics
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Embedding Stage.

        Args:
            model: OpenAI embedding model (default: text-embedding-3-small)
            batch_size: Batch size for API calls (default: 100)
            logger: Optional logger instance

        Raises:
            ValueError: If batch_size is not positive
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model = model
        self.batch_size = batch_size
        self.logger = logger

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model=model)

        # Statistics
        self.total_chunks_embedded = 0
        self.total_api_calls = 0

        if self.logger:
            self.logger.info(
                f"EmbeddingStage initialized: model={model}, batch_size={batch_size}"
            )

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create FAISS vector store from document chunks using Cosine Similarity.

        Args:
            chunks: List of chunked documents from Stage 2

        Returns:
            FAISS vector store instance with Cosine Similarity

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot create vector store from empty chunks")

        if self.logger:
            self.logger.info(
                f"Creating FAISS vector store with {len(chunks)} chunks using Cosine Similarity..."
            )

        # Generate embeddings for all chunks
        texts = [doc.page_content for doc in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)

        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings_list, dtype='float32')

        # Normalize vectors for cosine similarity (L2 normalization)
        faiss.normalize_L2(embeddings_array)

        # Create FAISS index using Inner Product (which equals cosine similarity for normalized vectors)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # IP = Inner Product = Cosine Similarity for normalized vectors
        index.add(embeddings_array)

        # Create docstore and index_to_docstore_id mapping
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        for i, doc in enumerate(chunks):
            doc_id = str(i)
            docstore.add({doc_id: doc})
            index_to_docstore_id[i] = doc_id

        # Create FAISS vector store with the custom index
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True  # Important: tells FAISS to normalize query vectors too
        )

        # Update statistics
        self.total_chunks_embedded += len(chunks)
        self.total_api_calls += (len(chunks) // self.batch_size) + 1

        if self.logger:
            stats = self.get_embedding_stats(chunks)
            self.logger.info(
                f"Vector store created with Cosine Similarity: {len(chunks)} chunks embedded, "
                f"estimated cost: ${stats['estimated_cost_usd']:.4f}"
            )

        return vector_store

    def save_vector_store(self, vector_store: FAISS, save_path: Path):
        """
        Save FAISS vector store to disk.

        Args:
            vector_store: FAISS vector store instance
            save_path: Path to save directory
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        vector_store.save_local(str(save_path))

        if self.logger:
            self.logger.info(f"Vector store saved to: {save_path}")

    def load_vector_store(self, load_path: Path) -> FAISS:
        """
        Load FAISS vector store from disk.

        Args:
            load_path: Path to saved vector store

        Returns:
            FAISS vector store instance

        Raises:
            ValueError: If vector store path or index file doesn't exist
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise ValueError(f"Vector store path does not exist: {load_path}")

        # Check for index file
        index_file = load_path / "index.faiss"
        if not index_file.exists():
            raise ValueError(f"Vector store index file not found: {index_file}")

        # Load FAISS index
        vector_store = FAISS.load_local(
            str(load_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        if self.logger:
            self.logger.info(f"Vector store loaded from: {load_path}")

        return vector_store

    def get_embedding_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about embedding operation.

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary with embedding statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "estimated_tokens": 0,
                "estimated_cost_usd": 0.0
            }

        total_chars = sum(len(c.page_content) for c in chunks)

        # Estimate tokens: ~4 chars per token for Korean/English mix
        estimated_tokens = total_chars // 4

        # text-embedding-3-small: $0.02 per 1M tokens
        estimated_cost = (estimated_tokens / 1_000_000) * 0.02

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "estimated_cost_usd": estimated_cost
        }

    def get_stage_stats(self) -> dict:
        """
        Get stage-level statistics.

        Returns:
            Dictionary with stage statistics
        """
        return {
            "model": self.model,
            "total_chunks_embedded": self.total_chunks_embedded,
            "total_api_calls": self.total_api_calls,
            "batch_size": self.batch_size
        }


def create_embedding_stage(
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    logger: Optional[RAGLogger] = None
) -> EmbeddingStage:
    """
    Factory function to create EmbeddingStage.

    Args:
        model: OpenAI embedding model
        batch_size: Batch size for API calls
        logger: Optional logger instance

    Returns:
        Configured EmbeddingStage instance
    """
    return EmbeddingStage(
        model=model,
        batch_size=batch_size,
        logger=logger
    )
