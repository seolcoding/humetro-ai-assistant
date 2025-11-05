"""
Stage 5: Vector Store Integration

Integrates FAISS vector store with retrieval capabilities.
Provides similarity search and retrieval functionality for RAG.
"""

from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.common.logger import RAGLogger


class VectorStoreStage:
    """
    Vector Store Stage for RAG Pipeline.

    Manages FAISS vector store with retrieval capabilities.
    Provides similarity search and k-NN retrieval for RAG.

    Features:
    - Load existing FAISS vector store
    - Similarity search with configurable k
    - MMR (Maximum Marginal Relevance) search for diversity
    - Retriever creation for LangChain integration
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Vector Store Stage.

        Args:
            model: OpenAI embedding model (must match embedding stage)
                  Default: text-embedding-3-large (3072D) for KG compatibility
            logger: Optional logger instance
        """
        self.model = model
        self.logger = logger

        # Initialize OpenAI embeddings (for loading vector store)
        self.embeddings = OpenAIEmbeddings(model=model)

        # Vector store instance
        self.vector_store: Optional[FAISS] = None

        if self.logger:
            self.logger.info(f"VectorStoreStage initialized: model={model}")

    def load_vector_store(self, store_path: Path) -> FAISS:
        """
        Load FAISS vector store from disk.

        Note: The loaded vector store should use Cosine Similarity (normalized vectors + IP).

        Args:
            store_path: Path to saved vector store directory

        Returns:
            Loaded FAISS vector store

        Raises:
            ValueError: If store path or index file doesn't exist
        """
        store_path = Path(store_path)

        if not store_path.exists():
            raise ValueError(f"Vector store path does not exist: {store_path}")

        index_file = store_path / "index.faiss"
        if not index_file.exists():
            raise ValueError(f"Vector store index file not found: {index_file}")

        if self.logger:
            self.logger.info(f"Loading vector store from: {store_path}")

        # Load with normalize_L2=True to ensure query vectors are normalized for cosine similarity
        self.vector_store = FAISS.load_local(
            str(store_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True  # Ensure cosine similarity by normalizing query vectors
        )

        if self.logger:
            # Check if the index is using Inner Product (cosine similarity for normalized vectors)
            index_type = self.vector_store.index.__class__.__name__
            similarity_metric = "Cosine Similarity" if "IP" in index_type else "L2 Distance"
            self.logger.info(
                f"Vector store loaded: {self.vector_store.index.ntotal:,} vectors, "
                f"Metric: {similarity_metric}"
            )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search on vector store.

        Args:
            query: Query string to search
            k: Number of top results to return
            **kwargs: Additional arguments for FAISS search

        Returns:
            List of most similar documents

        Raises:
            ValueError: If vector store not loaded
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")

        if self.logger:
            self.logger.info(f"Similarity search: query='{query[:50]}...', k={k}")

        results = self.vector_store.similarity_search(query, k=k, **kwargs)

        if self.logger:
            self.logger.info(f"Found {len(results)} results")

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Query string to search
            k: Number of top results to return
            **kwargs: Additional arguments for FAISS search

        Returns:
            List of (document, score) tuples

        Raises:
            ValueError: If vector store not loaded
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")

        if self.logger:
            self.logger.info(
                f"Similarity search with scores: query='{query[:50]}...', k={k}"
            )

        results = self.vector_store.similarity_search_with_score(query, k=k, **kwargs)

        if self.logger:
            self.logger.info(f"Found {len(results)} results with scores")

        return results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs
    ) -> List[Document]:
        """
        Perform MMR search for diverse results.

        MMR balances relevance and diversity to avoid redundant results.

        Args:
            query: Query string to search
            k: Number of results to return
            fetch_k: Number of candidates to fetch before MMR
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
            **kwargs: Additional arguments

        Returns:
            List of diverse relevant documents

        Raises:
            ValueError: If vector store not loaded
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")

        if self.logger:
            self.logger.info(
                f"MMR search: query='{query[:50]}...', k={k}, "
                f"fetch_k={fetch_k}, lambda={lambda_mult}"
            )

        results = self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs
        )

        if self.logger:
            self.logger.info(f"MMR returned {len(results)} diverse results")

        return results

    def as_retriever(self, **kwargs):
        """
        Create LangChain retriever from vector store.

        Args:
            **kwargs: Retriever configuration (search_type, search_kwargs, etc.)

        Returns:
            VectorStoreRetriever instance for LangChain

        Raises:
            ValueError: If vector store not loaded

        Example:
            retriever = stage.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")

        if self.logger:
            self.logger.info(f"Creating retriever with config: {kwargs}")

        return self.vector_store.as_retriever(**kwargs)

    def get_store_info(self) -> dict:
        """
        Get vector store information.

        Returns:
            Dictionary with store metadata

        Raises:
            ValueError: If vector store not loaded
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")

        return {
            "total_vectors": self.vector_store.index.ntotal,
            "vector_dimension": self.vector_store.index.d,
            "index_type": self.vector_store.index.__class__.__name__,
            "embedding_model": self.model
        }


def create_vector_store_stage(
    model: str = "text-embedding-3-large",
    logger: Optional[RAGLogger] = None
) -> VectorStoreStage:
    """
    Factory function to create VectorStoreStage.

    Args:
        model: OpenAI embedding model
        logger: Optional logger instance

    Returns:
        Configured VectorStoreStage instance
    """
    return VectorStoreStage(model=model, logger=logger)
