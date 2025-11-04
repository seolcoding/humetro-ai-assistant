"""
Unit tests for Stage 3: Embedding

Tests the EmbeddingStage class for:
- Initialization and configuration
- Vector store creation from chunks
- Save/load functionality
- Statistics calculation
- Error handling
- Metadata preservation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag_pipeline.stages.stage_03_embedding import (
    EmbeddingStage,
    create_embedding_stage
)
from src.common.logger import RAGLogger


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for vector store tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def logger():
    """Create test logger."""
    return RAGLogger(experiment_name="test_embedding")


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    chunks = [
        Document(
            page_content="서울시 교통 정책에 대한 첫 번째 청크입니다. " * 10,
            metadata={
                "source": "data/test/doc1.md",
                "chunk_id": 0,
                "chunk_size_config": 512,
                "chunk_overlap_config": 64,
                "chunk_length": 230
            }
        ),
        Document(
            page_content="지하철 노선도와 운영 시간에 관한 정보입니다. " * 10,
            metadata={
                "source": "data/test/doc2.md",
                "chunk_id": 1,
                "chunk_size_config": 512,
                "chunk_overlap_config": 64,
                "chunk_length": 245
            }
        ),
        Document(
            page_content="버스 노선과 정류장 위치 안내 자료입니다. " * 10,
            metadata={
                "source": "data/test/doc3.md",
                "chunk_id": 2,
                "chunk_size_config": 512,
                "chunk_overlap_config": 64,
                "chunk_length": 220
            }
        ),
    ]
    return chunks


# =============================================================================
# Initialization Tests
# =============================================================================

def test_embedding_stage_default_initialization(logger):
    """Test EmbeddingStage initialization with default parameters."""
    stage = EmbeddingStage(logger=logger)

    assert stage.model == "text-embedding-3-small"
    assert stage.batch_size == 100
    assert stage.logger == logger
    assert stage.embeddings is not None


def test_embedding_stage_custom_initialization(logger):
    """Test EmbeddingStage initialization with custom parameters."""
    stage = EmbeddingStage(
        model="text-embedding-3-large",
        batch_size=50,
        logger=logger
    )

    assert stage.model == "text-embedding-3-large"
    assert stage.batch_size == 50
    assert stage.logger == logger


def test_create_embedding_stage_factory(logger):
    """Test factory function for creating EmbeddingStage."""
    stage = create_embedding_stage(logger=logger)

    assert isinstance(stage, EmbeddingStage)
    assert stage.model == "text-embedding-3-small"
    assert stage.batch_size == 100


def test_create_embedding_stage_factory_with_custom_params(logger):
    """Test factory function with custom parameters."""
    stage = create_embedding_stage(
        model="text-embedding-3-large",
        batch_size=75,
        logger=logger
    )

    assert stage.model == "text-embedding-3-large"
    assert stage.batch_size == 75


# =============================================================================
# Statistics Tests
# =============================================================================

def test_get_embedding_stats_basic(logger, sample_chunks):
    """Test embedding statistics calculation."""
    stage = EmbeddingStage(logger=logger)
    stats = stage.get_embedding_stats(sample_chunks)

    assert "total_chunks" in stats
    assert "total_characters" in stats
    assert "estimated_tokens" in stats
    assert "estimated_cost_usd" in stats

    assert stats["total_chunks"] == 3
    assert stats["total_characters"] > 0
    assert stats["estimated_tokens"] > 0
    assert stats["estimated_cost_usd"] > 0


def test_get_embedding_stats_empty_chunks(logger):
    """Test statistics calculation with empty chunks list."""
    stage = EmbeddingStage(logger=logger)
    stats = stage.get_embedding_stats([])

    assert stats["total_chunks"] == 0
    assert stats["total_characters"] == 0
    assert stats["estimated_tokens"] == 0
    assert stats["estimated_cost_usd"] == 0.0


def test_get_embedding_stats_cost_calculation(logger, sample_chunks):
    """Test cost calculation accuracy (OpenAI pricing: $0.02 per 1M tokens)."""
    stage = EmbeddingStage(logger=logger)
    stats = stage.get_embedding_stats(sample_chunks)

    # Cost should be: (estimated_tokens / 1_000_000) * 0.02
    expected_cost = (stats["estimated_tokens"] / 1_000_000) * 0.02

    assert abs(stats["estimated_cost_usd"] - expected_cost) < 0.0001


# =============================================================================
# Vector Store Creation Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_create_vector_store_basic(mock_faiss, logger, sample_chunks):
    """Test basic vector store creation."""
    # Mock FAISS
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    stage = EmbeddingStage(logger=logger)
    vector_store = stage.create_vector_store(sample_chunks)

    # Verify FAISS.from_documents was called correctly
    mock_faiss.from_documents.assert_called_once()
    call_kwargs = mock_faiss.from_documents.call_args[1]

    assert call_kwargs["documents"] == sample_chunks
    assert vector_store == mock_vector_store


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_create_vector_store_empty_chunks(mock_faiss, logger):
    """Test vector store creation with empty chunks list."""
    stage = EmbeddingStage(logger=logger)

    with pytest.raises(ValueError, match="Cannot create vector store from empty chunks"):
        stage.create_vector_store([])


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_create_vector_store_metadata_preserved(mock_faiss, logger, sample_chunks):
    """Test that chunk metadata is preserved in vector store."""
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    stage = EmbeddingStage(logger=logger)
    stage.create_vector_store(sample_chunks)

    # Verify documents passed to FAISS have all metadata
    call_args = mock_faiss.from_documents.call_args[1]
    documents = call_args["documents"]

    for doc in documents:
        assert "source" in doc.metadata
        assert "chunk_id" in doc.metadata
        assert "chunk_size_config" in doc.metadata
        assert "chunk_overlap_config" in doc.metadata


# =============================================================================
# Save/Load Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_save_vector_store_basic(mock_faiss, logger, temp_dir):
    """Test basic vector store saving."""
    mock_vector_store = MagicMock()

    stage = EmbeddingStage(logger=logger)
    save_path = temp_dir / "vector_store"

    stage.save_vector_store(mock_vector_store, save_path)

    # Verify save_local was called
    mock_vector_store.save_local.assert_called_once_with(str(save_path))

    # Verify directory was created
    assert save_path.exists()


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_save_vector_store_creates_parent_dirs(mock_faiss, logger, temp_dir):
    """Test that save creates parent directories if they don't exist."""
    mock_vector_store = MagicMock()

    stage = EmbeddingStage(logger=logger)
    save_path = temp_dir / "nested" / "dirs" / "vector_store"

    stage.save_vector_store(mock_vector_store, save_path)

    # Verify all parent directories were created
    assert save_path.exists()
    mock_vector_store.save_local.assert_called_once()


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_load_vector_store_basic(mock_faiss, logger, temp_dir):
    """Test basic vector store loading."""
    mock_vector_store = MagicMock()
    mock_faiss.load_local.return_value = mock_vector_store

    stage = EmbeddingStage(logger=logger)
    load_path = temp_dir / "vector_store"
    load_path.mkdir(parents=True, exist_ok=True)

    # Create dummy index.faiss file
    (load_path / "index.faiss").touch()

    loaded_store = stage.load_vector_store(load_path)

    # Verify load_local was called correctly
    mock_faiss.load_local.assert_called_once()
    call_args = mock_faiss.load_local.call_args

    assert str(load_path) in str(call_args[0][0])
    assert loaded_store == mock_vector_store


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_load_vector_store_missing_directory(mock_faiss, logger, temp_dir):
    """Test loading from non-existent directory."""
    stage = EmbeddingStage(logger=logger)
    missing_path = temp_dir / "nonexistent"

    with pytest.raises(ValueError, match="Vector store path does not exist"):
        stage.load_vector_store(missing_path)


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_load_vector_store_missing_index_file(mock_faiss, logger, temp_dir):
    """Test loading when index.faiss file is missing."""
    stage = EmbeddingStage(logger=logger)
    store_path = temp_dir / "vector_store"
    store_path.mkdir(parents=True, exist_ok=True)

    # Create directory but no index file
    with pytest.raises(ValueError, match="Vector store index file not found"):
        stage.load_vector_store(store_path)


# =============================================================================
# Integration Tests (if OpenAI API is available)
# =============================================================================

@pytest.mark.skipif(
    not Path("~/.openai_api_key").expanduser().exists(),
    reason="OpenAI API key not available"
)
def test_create_and_search_vector_store_integration(logger, sample_chunks, temp_dir):
    """
    Integration test: Create vector store and perform similarity search.

    Note: This test requires OpenAI API key and will be skipped if not available.
    """
    stage = EmbeddingStage(logger=logger)

    # Create vector store
    vector_store = stage.create_vector_store(sample_chunks)

    # Test similarity search
    query = "서울시 교통 정책"
    results = vector_store.similarity_search(query, k=2)

    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)

    # Save and load
    save_path = temp_dir / "test_store"
    stage.save_vector_store(vector_store, save_path)

    loaded_store = stage.load_vector_store(save_path)

    # Test loaded store works
    loaded_results = loaded_store.similarity_search(query, k=2)
    assert len(loaded_results) == len(results)


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_embedding_stage_with_invalid_model(logger):
    """Test initialization with invalid model name (should still initialize)."""
    # OpenAI client will validate the model at runtime, not at init
    stage = EmbeddingStage(model="invalid-model-name", logger=logger)
    assert stage.model == "invalid-model-name"


def test_embedding_stage_with_invalid_batch_size(logger):
    """Test initialization with invalid batch size."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        EmbeddingStage(batch_size=0, logger=logger)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        EmbeddingStage(batch_size=-10, logger=logger)


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_create_vector_store_with_malformed_chunks(mock_faiss, logger):
    """Test vector store creation with malformed chunk data."""
    # Chunks with missing metadata
    bad_chunks = [
        Document(
            page_content="Content without proper metadata",
            metadata={}  # Missing required fields
        )
    ]

    stage = EmbeddingStage(logger=logger)

    # Should still work - FAISS doesn't require specific metadata
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    result = stage.create_vector_store(bad_chunks)
    assert result == mock_vector_store


def test_get_embedding_stats_with_none_chunks(logger):
    """Test statistics calculation with None input."""
    stage = EmbeddingStage(logger=logger)

    # None is falsy, so it returns early with zeros (same as empty list)
    stats = stage.get_embedding_stats(None)

    assert stats["total_chunks"] == 0
    assert stats["total_characters"] == 0
    assert stats["estimated_tokens"] == 0
    assert stats["estimated_cost_usd"] == 0.0


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_save_vector_store_with_none_path(mock_faiss, logger):
    """Test saving with None path."""
    mock_vector_store = MagicMock()
    stage = EmbeddingStage(logger=logger)

    with pytest.raises((TypeError, AttributeError)):
        stage.save_vector_store(mock_vector_store, None)


# =============================================================================
# Logging Tests
# =============================================================================

def test_embedding_stage_logs_initialization(logger):
    """Test that initialization is logged."""
    with patch.object(logger, 'info') as mock_info:
        stage = EmbeddingStage(logger=logger)

        # Should log initialization
        assert mock_info.called


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_embedding_stage_logs_vector_store_creation(mock_faiss, logger, sample_chunks):
    """Test that vector store creation is logged."""
    mock_vector_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_vector_store

    with patch.object(logger, 'info') as mock_info:
        stage = EmbeddingStage(logger=logger)
        stage.create_vector_store(sample_chunks)

        # Should log creation
        assert any("Vector store created" in str(call) for call in mock_info.call_args_list)


@patch('src.rag_pipeline.stages.stage_03_embedding.FAISS')
def test_embedding_stage_logs_save_load(mock_faiss, logger, temp_dir):
    """Test that save/load operations are logged."""
    mock_vector_store = MagicMock()
    mock_faiss.load_local.return_value = mock_vector_store

    stage = EmbeddingStage(logger=logger)
    save_path = temp_dir / "vector_store"
    save_path.mkdir(parents=True, exist_ok=True)

    # Create index file for load test
    (save_path / "index.faiss").touch()

    with patch.object(logger, 'info') as mock_info:
        stage.save_vector_store(mock_vector_store, save_path)
        stage.load_vector_store(save_path)

        # Should log both operations
        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("saved" in msg.lower() for msg in log_messages)
        assert any("loaded" in msg.lower() for msg in log_messages)
