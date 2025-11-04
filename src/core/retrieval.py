"""
RAG retrieval module.

This module provides retrieval functionality for RAG systems,
including similarity search and retriever creation.
"""

from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.common.logger import RAGLogger


class RAGRetriever:
    """
    RAG retrieval system.

    Provides similarity search and contextual compression
    for improved retrieval quality.
    """

    def __init__(
        self,
        vectorstore: FAISS,
        k: int = 5,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize RAG retriever.

        Args:
            vectorstore: FAISS vector store
            k: Number of documents to retrieve
            logger: Optional RAG logger
        """
        self.vectorstore = vectorstore
        self.k = k
        self.logger = logger

        # Create base retriever
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents for query.

        Args:
            query: Query string

        Returns:
            List of retrieved documents
        """
        if self.logger:
            self.logger.debug(f"Retrieving top-{self.k} documents for query")

        documents = self.retriever.get_relevant_documents(query)

        if self.logger:
            self.logger.debug(f"Retrieved {len(documents)} documents")

        return documents

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Query string
            k: Number of documents to retrieve (default: self.k)

        Returns:
            List of similar documents
        """
        k = k or self.k

        if self.logger:
            self.logger.debug(f"Similarity search for top-{k} documents")

        documents = self.vectorstore.similarity_search(query, k=k)

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Perform similarity search with scores.

        Args:
            query: Query string
            k: Number of documents to retrieve (default: self.k)

        Returns:
            List of (document, score) tuples
        """
        k = k or self.k

        if self.logger:
            self.logger.debug(f"Similarity search with scores for top-{k} documents")

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def get_retrieval_stats(self, query: str, k: Optional[int] = None) -> dict:
        """
        Get retrieval statistics.

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            Dictionary with retrieval statistics
        """
        k = k or self.k

        # Perform retrieval with scores
        results = self.similarity_search_with_score(query, k=k)

        if not results:
            return {
                "num_retrieved": 0,
                "avg_score": 0,
                "min_score": 0,
                "max_score": 0
            }

        scores = [score for _, score in results]

        return {
            "num_retrieved": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores)
        }


def create_retriever(
    vectorstore: FAISS,
    k: int = 5,
    logger: Optional[RAGLogger] = None
) -> RAGRetriever:
    """
    Create RAG retriever.

    Args:
        vectorstore: FAISS vector store
        k: Number of documents to retrieve
        logger: Optional RAG logger

    Returns:
        RAGRetriever instance
    """
    return RAGRetriever(
        vectorstore=vectorstore,
        k=k,
        logger=logger
    )
