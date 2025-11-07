"""
Unit tests for Stage 5: Vector Store

Tests the VectorStoreStage class for:
- Initialization
- Vector store loading
- Similarity search operations
- MMR search
- Retriever creation
- Error handling
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag_pipeline.stages.stage_05_vector_store import (
    VectorStoreStage,
    create_vector_store_stage
)
from src.common.logger import RAGLogger


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def logger():
    """Create test logger."""
    return RAGLogger(experiment_name="test_vector_store")


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = [
        Document(
            page_content="서울시 교통 정책 관련 문서입니다.",
            metadata={"source": "doc1.md", "chunk_id": 0}
        ),
        Document(
            page_content="지하철 노선도 정보입니다.",
            metadata={"source": "doc2.md", "chunk_id": 1}
        ),
    ]
    return docs


# =============================================================================
# Initialization Tests
# =============================================================================

def test_vector_store_stage_default_initialization(logger):
    """Test VectorStoreStage initialization with defaults."""
    stage = VectorStoreStage(logger=logger)

    assert stage.model == "text-embedding-3-small"
    assert stage.logger == logger
    assert stage.embeddings is not None
    assert stage.vector_store is None


def test_vector_store_stage_custom_initialization(logger):
    """Test VectorStoreStage initialization with custom model."""
    stage = VectorStoreStage(model="text-embedding-3-large", logger=logger)

    assert stage.model == "text-embedding-3-large"
    assert stage.logger == logger


def test_create_vector_store_stage_factory(logger):
    """Test factory function."""
    stage = create_vector_store_stage(logger=logger)

    assert isinstance(stage, VectorStoreStage)
    assert stage.model == "text-embedding-3-small"


def test_create_vector_store_stage_custom_params(logger):
    """Test factory with custom parameters."""
    stage = create_vector_store_stage(
        model="text-embedding-3-large",
        logger=logger
    )

    assert stage.model == "text-embedding-3-large"


# =============================================================================
# Loading Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_05_vector_store.FAISS')
def test_load_vector_store_success(mock_faiss, logger, temp_dir):
    """Test successful vector store loading."""
    # Setup
    mock_vector_store = MagicMock()
    mock_vector_store.index.ntotal = 1000
    mock_faiss.load_local.return_value = mock_vector_store

    stage = VectorStoreStage(logger=logger)
    store_path = temp_dir / "vector_store"
    store_path.mkdir(parents=True)
    (store_path / "index.faiss").touch()

    # Execute
    result = stage.load_vector_store(store_path)

    # Verify
    assert result == mock_vector_store
    assert stage.vector_store == mock_vector_store
    mock_faiss.load_local.assert_called_once()


def test_load_vector_store_missing_directory(logger, temp_dir):
    """Test loading from non-existent directory."""
    stage = VectorStoreStage(logger=logger)
    missing_path = temp_dir / "nonexistent"

    with pytest.raises(ValueError, match="Vector store path does not exist"):
        stage.load_vector_store(missing_path)


def test_load_vector_store_missing_index_file(logger, temp_dir):
    """Test loading when index.faiss file is missing."""
    stage = VectorStoreStage(logger=logger)
    store_path = temp_dir / "vector_store"
    store_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="Vector store index file not found"):
        stage.load_vector_store(store_path)


# =============================================================================
# Similarity Search Tests
# =============================================================================

def test_similarity_search_basic(logger, sample_documents):
    """Test basic similarity search."""
    # Setup
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = sample_documents

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    # Execute
    results = stage.similarity_search("교통 정책", k=2)

    # Verify
    assert len(results) == 2
    assert results == sample_documents
    mock_vector_store.similarity_search.assert_called_once_with("교통 정책", k=2)


def test_similarity_search_not_loaded(logger):
    """Test similarity search without loading vector store."""
    stage = VectorStoreStage(logger=logger)

    with pytest.raises(ValueError, match="Vector store not loaded"):
        stage.similarity_search("test query")


def test_similarity_search_with_kwargs(logger, sample_documents):
    """Test similarity search with additional kwargs."""
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = sample_documents

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    results = stage.similarity_search("query", k=3, filter={"source": "doc1.md"})

    mock_vector_store.similarity_search.assert_called_once_with(
        "query",
        k=3,
        filter={"source": "doc1.md"}
    )


# =============================================================================
# Similarity Search with Score Tests
# =============================================================================

def test_similarity_search_with_score(logger, sample_documents):
    """Test similarity search with scores."""
    mock_results = [(sample_documents[0], 0.95), (sample_documents[1], 0.87)]
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_score.return_value = mock_results

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    results = stage.similarity_search_with_score("query", k=2)

    assert len(results) == 2
    assert results[0][1] == 0.95
    assert results[1][1] == 0.87
    mock_vector_store.similarity_search_with_score.assert_called_once()


def test_similarity_search_with_score_not_loaded(logger):
    """Test similarity search with score without loading."""
    stage = VectorStoreStage(logger=logger)

    with pytest.raises(ValueError, match="Vector store not loaded"):
        stage.similarity_search_with_score("query")


# =============================================================================
# MMR Search Tests
# =============================================================================

def test_mmr_search_basic(logger, sample_documents):
    """Test MMR search."""
    mock_vector_store = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = sample_documents

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    results = stage.max_marginal_relevance_search("query", k=2)

    assert len(results) == 2
    mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
        "query",
        k=2,
        fetch_k=20,
        lambda_mult=0.5
    )


def test_mmr_search_custom_params(logger, sample_documents):
    """Test MMR search with custom parameters."""
    mock_vector_store = MagicMock()
    mock_vector_store.max_marginal_relevance_search.return_value = sample_documents

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    results = stage.max_marginal_relevance_search(
        "query",
        k=5,
        fetch_k=30,
        lambda_mult=0.7
    )

    mock_vector_store.max_marginal_relevance_search.assert_called_once_with(
        "query",
        k=5,
        fetch_k=30,
        lambda_mult=0.7
    )


def test_mmr_search_not_loaded(logger):
    """Test MMR search without loading vector store."""
    stage = VectorStoreStage(logger=logger)

    with pytest.raises(ValueError, match="Vector store not loaded"):
        stage.max_marginal_relevance_search("query")


# =============================================================================
# Retriever Creation Tests
# =============================================================================

def test_as_retriever_basic(logger):
    """Test retriever creation."""
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    retriever = stage.as_retriever()

    assert retriever == mock_retriever
    mock_vector_store.as_retriever.assert_called_once_with()


def test_as_retriever_with_config(logger):
    """Test retriever creation with configuration."""
    mock_retriever = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever

    stage = VectorStoreStage(logger=logger)
    stage.vector_store = mock_vector_store

    config = {
        "search_type": "similarity",
        "search_kwargs": {"k": 4}
    }

    retriever = stage.as_retriever(**config)

    mock_vector_store.as_retriever.assert_called_once_with(**config)


def test_as_retriever_not_loaded(logger):
    """Test retriever creation without loading vector store."""
    stage = VectorStoreStage(logger=logger)

    with pytest.raises(ValueError, match="Vector store not loaded"):
        stage.as_retriever()


# =============================================================================
# Store Info Tests
# =============================================================================

def test_get_store_info(logger):
    """Test getting store information."""
    mock_index = MagicMock()
    mock_index.ntotal = 1000
    mock_index.d = 1536
    mock_index.__class__.__name__ = "IndexFlatL2"

    mock_vector_store = MagicMock()
    mock_vector_store.index = mock_index

    stage = VectorStoreStage(model="text-embedding-3-small", logger=logger)
    stage.vector_store = mock_vector_store

    info = stage.get_store_info()

    assert info["total_vectors"] == 1000
    assert info["vector_dimension"] == 1536
    assert info["index_type"] == "IndexFlatL2"
    assert info["embedding_model"] == "text-embedding-3-small"


def test_get_store_info_not_loaded(logger):
    """Test get_store_info without loading vector store."""
    stage = VectorStoreStage(logger=logger)

    with pytest.raises(ValueError, match="Vector store not loaded"):
        stage.get_store_info()


# =============================================================================
# Logging Tests
# =============================================================================

def test_vector_store_stage_logs_initialization(logger):
    """Test that initialization is logged."""
    with patch.object(logger, 'info') as mock_info:
        stage = VectorStoreStage(logger=logger)

        assert mock_info.called


@patch('src.rag_pipeline.stages.stage_05_vector_store.FAISS')
def test_vector_store_stage_logs_loading(mock_faiss, logger, temp_dir):
    """Test that loading is logged."""
    mock_vector_store = MagicMock()
    mock_vector_store.index.ntotal = 1000
    mock_faiss.load_local.return_value = mock_vector_store

    store_path = temp_dir / "vector_store"
    store_path.mkdir(parents=True)
    (store_path / "index.faiss").touch()

    with patch.object(logger, 'info') as mock_info:
        stage = VectorStoreStage(logger=logger)
        stage.load_vector_store(store_path)

        # Should log both loading and loaded messages
        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Loading" in msg for msg in log_messages)
        assert any("loaded" in msg for msg in log_messages)


def test_vector_store_stage_logs_searches(logger, sample_documents):
    """Test that searches are logged."""
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.return_value = sample_documents

    with patch.object(logger, 'info') as mock_info:
        stage = VectorStoreStage(logger=logger)
        stage.vector_store = mock_vector_store

        stage.similarity_search("test query", k=3)

        # Should log search operation
        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Similarity search" in msg for msg in log_messages)
