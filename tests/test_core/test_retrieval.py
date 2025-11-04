"""Tests for RAGRetriever."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

from src.core.retrieval import RAGRetriever, create_retriever


class TestRAGRetriever:
    """Test RAGRetriever functionality."""

    @pytest.fixture
    def mock_vectorstore(self):
        """Mock FAISS vectorstore."""
        mock = Mock()

        # Mock as_retriever
        mock_retriever = Mock()
        mock.as_retriever.return_value = mock_retriever

        # Mock similarity_search
        mock.similarity_search.return_value = [
            Document(page_content="결과 1", metadata={"score": 0.95}),
            Document(page_content="결과 2", metadata={"score": 0.85}),
            Document(page_content="결과 3", metadata={"score": 0.75}),
        ]

        # Mock similarity_search_with_score
        mock.similarity_search_with_score.return_value = [
            (Document(page_content="결과 1", metadata={"score": 0.95}), 0.95),
            (Document(page_content="결과 2", metadata={"score": 0.85}), 0.85),
            (Document(page_content="결과 3", metadata={"score": 0.75}), 0.75),
        ]

        # Mock get_relevant_documents
        mock_retriever.get_relevant_documents.return_value = [
            Document(page_content="검색 결과 1"),
            Document(page_content="검색 결과 2"),
        ]

        return mock

    def test_retriever_initialization(self, mock_vectorstore):
        """Test retriever initialization."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=5
        )

        assert retriever.vectorstore == mock_vectorstore
        assert retriever.k == 5
        assert retriever.retriever is not None

        # Verify as_retriever was called with correct kwargs
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_kwargs={"k": 5}
        )

    def test_retrieve(self, mock_vectorstore):
        """Test document retrieval."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "테스트 쿼리"
        documents = retriever.retrieve(query)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        # Verify get_relevant_documents was called
        retriever.retriever.get_relevant_documents.assert_called_once_with(query)

    def test_similarity_search(self, mock_vectorstore):
        """Test similarity search."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=5
        )

        query = "테스트 쿼리"
        documents = retriever.similarity_search(query)

        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)

        # Verify similarity_search was called
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=5)

    def test_similarity_search_custom_k(self, mock_vectorstore):
        """Test similarity search with custom k."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=5
        )

        query = "테스트 쿼리"
        documents = retriever.similarity_search(query, k=10)

        # Verify custom k was used
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=10)

    def test_similarity_search_with_score(self, mock_vectorstore):
        """Test similarity search with scores."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "테스트 쿼리"
        results = retriever.similarity_search_with_score(query)

        assert len(results) == 3
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)

        # Check structure
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score == 0.95

        # Verify similarity_search_with_score was called
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=3)

    def test_similarity_search_with_score_custom_k(self, mock_vectorstore):
        """Test similarity search with scores and custom k."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "테스트 쿼리"
        results = retriever.similarity_search_with_score(query, k=7)

        # Verify custom k was used
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=7)

    def test_get_retrieval_stats(self, mock_vectorstore):
        """Test retrieval statistics."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "테스트 쿼리"
        stats = retriever.get_retrieval_stats(query)

        assert stats["num_retrieved"] == 3
        assert stats["avg_score"] == pytest.approx((0.95 + 0.85 + 0.75) / 3)
        assert stats["min_score"] == 0.75
        assert stats["max_score"] == 0.95

    def test_get_retrieval_stats_empty_results(self, mock_vectorstore):
        """Test retrieval statistics with empty results."""
        # Mock empty results
        mock_vectorstore.similarity_search_with_score.return_value = []

        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "빈 결과 쿼리"
        stats = retriever.get_retrieval_stats(query)

        assert stats["num_retrieved"] == 0
        assert stats["avg_score"] == 0
        assert stats["min_score"] == 0
        assert stats["max_score"] == 0

    def test_get_retrieval_stats_custom_k(self, mock_vectorstore):
        """Test retrieval statistics with custom k."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3
        )

        query = "테스트 쿼리"
        stats = retriever.get_retrieval_stats(query, k=5)

        # Verify custom k was used
        mock_vectorstore.similarity_search_with_score.assert_called_once_with(query, k=5)

    def test_create_retriever_function(self, mock_vectorstore):
        """Test create_retriever helper function."""
        retriever = create_retriever(
            vectorstore=mock_vectorstore,
            k=7
        )

        assert isinstance(retriever, RAGRetriever)
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.k == 7

    def test_retriever_with_logger(self, mock_vectorstore):
        """Test retriever with logger."""
        mock_logger = Mock()

        retriever = RAGRetriever(
            vectorstore=mock_vectorstore,
            k=3,
            logger=mock_logger
        )

        query = "테스트 쿼리"
        retriever.retrieve(query)

        # Verify logger was called
        assert mock_logger.debug.call_count >= 2  # Called before and after retrieval

    def test_default_k_value(self, mock_vectorstore):
        """Test default k value."""
        retriever = RAGRetriever(
            vectorstore=mock_vectorstore
        )

        assert retriever.k == 5  # Default value
