"""Tests for VectorStoreManager."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.core.vector_store import VectorStoreManager, create_vector_store_manager


class TestVectorStoreManager:
    """Test VectorStoreManager functionality."""

    @pytest.fixture
    def temp_persist_dir(self, tmp_path):
        """Create temporary persist directory."""
        return tmp_path / "vectorstore"

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings."""
        mock = Mock()
        mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock.embed_query.return_value = [0.4, 0.5, 0.6]
        return mock

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(page_content="테스트 문서 1", metadata={"source": "test1"}),
            Document(page_content="테스트 문서 2", metadata={"source": "test2"}),
            Document(page_content="테스트 문서 3", metadata={"source": "test3"}),
        ]

    def test_manager_initialization(self, temp_persist_dir, mock_embeddings):
        """Test manager initialization."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        assert manager.embeddings == mock_embeddings
        assert manager.persist_directory == temp_persist_dir
        assert temp_persist_dir.exists()  # Should create directory

    def test_cache_exists_check(self, temp_persist_dir, mock_embeddings):
        """Test cache existence check."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Initially no cache
        assert not manager._cache_exists()

        # Create dummy cache files
        manager.index_file.touch()
        manager.docstore_file.touch()

        # Now cache exists
        assert manager._cache_exists()

    @patch('src.core.vector_store.FAISS')
    def test_create_new_vectorstore(self, mock_faiss, temp_persist_dir, mock_embeddings, sample_documents):
        """Test creating new vector store."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Mock FAISS.from_documents
        mock_vectorstore = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        vectorstore = manager._create_new(sample_documents)

        # Verify FAISS was called correctly
        mock_faiss.from_documents.assert_called_once_with(
            sample_documents,
            mock_embeddings
        )

        # Verify save was called
        mock_vectorstore.save_local.assert_called_once()

    @patch('src.core.vector_store.FAISS')
    def test_load_cached_vectorstore(self, mock_faiss, temp_persist_dir, mock_embeddings):
        """Test loading cached vector store."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Create cache files
        manager.index_file.touch()
        manager.docstore_file.touch()

        # Mock FAISS.load_local
        mock_vectorstore = MagicMock()
        mock_vectorstore.index.ntotal = 100
        mock_faiss.load_local.return_value = mock_vectorstore

        vectorstore = manager._load_cached()

        # Verify FAISS.load_local was called
        mock_faiss.load_local.assert_called_once_with(
            str(temp_persist_dir),
            mock_embeddings,
            allow_dangerous_deserialization=True
        )

    @patch('src.core.vector_store.FAISS')
    def test_get_or_create_with_cache(self, mock_faiss, temp_persist_dir, mock_embeddings):
        """Test get_or_create when cache exists."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Create cache
        manager.index_file.touch()
        manager.docstore_file.touch()

        mock_vectorstore = MagicMock()
        mock_faiss.load_local.return_value = mock_vectorstore

        vectorstore = manager.get_or_create_vectorstore()

        # Should load from cache
        mock_faiss.load_local.assert_called_once()
        mock_faiss.from_documents.assert_not_called()

    @patch('src.core.vector_store.FAISS')
    def test_get_or_create_without_cache(self, mock_faiss, temp_persist_dir, mock_embeddings, sample_documents):
        """Test get_or_create when no cache exists."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        mock_vectorstore = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore

        vectorstore = manager.get_or_create_vectorstore(documents=sample_documents)

        # Should create new
        mock_faiss.from_documents.assert_called_once()
        mock_faiss.load_local.assert_not_called()

    def test_get_or_create_no_cache_no_documents_raises_error(self, temp_persist_dir, mock_embeddings):
        """Test that get_or_create raises error when no cache and no documents."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        with pytest.raises(ValueError) as exc_info:
            manager.get_or_create_vectorstore()

        assert "No cached vector store found" in str(exc_info.value)

    def test_delete_cache(self, temp_persist_dir, mock_embeddings):
        """Test cache deletion."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Create cache files
        manager.index_file.touch()
        manager.docstore_file.touch()

        assert manager._cache_exists()

        # Delete cache
        manager.delete_cache()

        assert not manager._cache_exists()

    def test_get_cache_size_no_cache(self, temp_persist_dir, mock_embeddings):
        """Test get_cache_size when no cache exists."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        cache_info = manager.get_cache_size()

        assert cache_info["exists"] is False
        assert cache_info["size_mb"] == 0

    def test_get_cache_size_with_cache(self, temp_persist_dir, mock_embeddings):
        """Test get_cache_size when cache exists."""
        manager = VectorStoreManager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        # Create cache files with content
        manager.index_file.write_bytes(b"x" * 1024 * 1024)  # 1 MB
        manager.docstore_file.write_bytes(b"x" * 2 * 1024 * 1024)  # 2 MB

        cache_info = manager.get_cache_size()

        assert cache_info["exists"] is True
        assert cache_info["size_mb"] == pytest.approx(3.0, abs=0.1)
        assert cache_info["index_size_mb"] == pytest.approx(1.0, abs=0.1)
        assert cache_info["docstore_size_mb"] == pytest.approx(2.0, abs=0.1)

    def test_create_vector_store_manager_function(self, temp_persist_dir, mock_embeddings):
        """Test create_vector_store_manager helper function."""
        manager = create_vector_store_manager(
            embeddings=mock_embeddings,
            persist_directory=temp_persist_dir
        )

        assert isinstance(manager, VectorStoreManager)
        assert manager.embeddings == mock_embeddings
        assert manager.persist_directory == temp_persist_dir
