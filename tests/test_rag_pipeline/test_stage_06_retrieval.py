"""
Unit tests for Stage 6: Retrieval

Tests the RetrievalStage class for:
- Initialization
- RAG chain creation
- Query operations
- Query with sources
- Batch queries
- Prompt customization
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag_pipeline.stages.stage_06_retrieval import (
    RetrievalStage,
    create_retrieval_stage
)
from src.common.logger import RAGLogger


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def logger():
    """Create test logger."""
    return RAGLogger(experiment_name="test_retrieval")


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    retriever = MagicMock()

    # Mock invoke to return sample documents
    sample_docs = [
        Document(
            page_content="서울시 교통 정책 내용",
            metadata={"source": "doc1.md"}
        ),
        Document(
            page_content="지하철 운영 정보",
            metadata={"source": "doc2.md"}
        ),
    ]
    retriever.invoke.return_value = sample_docs

    return retriever


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        Document(
            page_content="서울시 대중교통 요금 정보",
            metadata={"source": "fare.md", "chunk_id": 0}
        ),
        Document(
            page_content="버스 노선 안내",
            metadata={"source": "bus.md", "chunk_id": 1}
        ),
    ]


# =============================================================================
# Initialization Tests
# =============================================================================

def test_retrieval_stage_default_initialization(mock_retriever, logger):
    """Test RetrievalStage initialization with defaults."""
    stage = RetrievalStage(retriever=mock_retriever, logger=logger)

    assert stage.retriever == mock_retriever
    assert stage.model_name == "gpt-4o-mini"
    assert stage.temperature == 0.0
    assert stage.logger == logger
    assert stage.llm is not None
    assert stage.prompt is not None
    assert stage.chain is not None


def test_retrieval_stage_custom_initialization(mock_retriever, logger):
    """Test RetrievalStage initialization with custom parameters."""
    stage = RetrievalStage(
        retriever=mock_retriever,
        model="gpt-4",
        temperature=0.7,
        logger=logger
    )

    assert stage.model_name == "gpt-4"
    assert stage.temperature == 0.7


def test_retrieval_stage_custom_prompt(mock_retriever, logger):
    """Test RetrievalStage with custom prompt template."""
    custom_prompt = "Context: {context}\nQuestion: {question}\nAnswer:"

    stage = RetrievalStage(
        retriever=mock_retriever,
        prompt_template=custom_prompt,
        logger=logger
    )

    assert stage.prompt is not None


def test_create_retrieval_stage_factory(mock_retriever, logger):
    """Test factory function."""
    stage = create_retrieval_stage(retriever=mock_retriever, logger=logger)

    assert isinstance(stage, RetrievalStage)
    assert stage.model_name == "gpt-4o-mini"
    assert stage.temperature == 0.0


def test_create_retrieval_stage_custom_params(mock_retriever, logger):
    """Test factory with custom parameters."""
    stage = create_retrieval_stage(
        retriever=mock_retriever,
        model="gpt-4",
        temperature=0.5,
        logger=logger
    )

    assert stage.model_name == "gpt-4"
    assert stage.temperature == 0.5


# =============================================================================
# Query Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_query_basic(mock_chat_openai, mock_retriever, logger):
    """Test basic query operation."""
    # Mock LLM response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="서울시 대중교통 요금은 1,250원입니다.")
    mock_chat_openai.return_value = mock_llm

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)

    # Mock the chain
    stage.chain = MagicMock()
    stage.chain.invoke.return_value = "서울시 대중교통 요금은 1,250원입니다."

    answer = stage.query("대중교통 요금은 얼마인가요?")

    assert isinstance(answer, str)
    assert len(answer) > 0
    stage.chain.invoke.assert_called_once()


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_query_with_logging(mock_chat_openai, mock_retriever, logger):
    """Test that query operations are logged."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    with patch.object(logger, 'info') as mock_info:
        stage = RetrievalStage(retriever=mock_retriever, logger=logger)
        stage.chain = MagicMock()
        stage.chain.invoke.return_value = "Test answer"

        stage.query("Test question")

        # Should log query and answer
        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Query" in msg for msg in log_messages)
        assert any("Answer generated" in msg for msg in log_messages)


# =============================================================================
# Query with Sources Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_query_with_sources_basic(mock_chat_openai, mock_retriever, logger, sample_documents):
    """Test query with sources."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    # Setup retriever to return sample documents
    mock_retriever.invoke.return_value = sample_documents

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)
    stage.chain = MagicMock()
    stage.chain.invoke.return_value = "Test answer"

    result = stage.query_with_sources("Test question")

    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert "num_sources" in result
    assert result["question"] == "Test question"
    assert result["answer"] == "Test answer"
    assert len(result["sources"]) == 2
    assert result["num_sources"] == 2


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_query_with_sources_metadata(mock_chat_openai, mock_retriever, logger, sample_documents):
    """Test that source documents preserve metadata."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    mock_retriever.invoke.return_value = sample_documents

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)
    stage.chain = MagicMock()
    stage.chain.invoke.return_value = "Answer"

    result = stage.query_with_sources("Question")

    # Verify metadata is preserved
    assert result["sources"][0].metadata["source"] == "fare.md"
    assert result["sources"][1].metadata["source"] == "bus.md"


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_query_with_sources_logging(mock_chat_openai, mock_retriever, logger, sample_documents):
    """Test that query_with_sources logs properly."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm
    mock_retriever.invoke.return_value = sample_documents

    with patch.object(logger, 'info') as mock_info:
        stage = RetrievalStage(retriever=mock_retriever, logger=logger)
        stage.chain = MagicMock()
        stage.chain.invoke.return_value = "Answer"

        stage.query_with_sources("Question")

        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Query with sources" in msg for msg in log_messages)
        assert any("source documents" in msg for msg in log_messages)


# =============================================================================
# Batch Query Tests
# =============================================================================

@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_batch_query_basic(mock_chat_openai, mock_retriever, logger):
    """Test batch query operation."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)
    stage.chain = MagicMock()
    stage.chain.batch.return_value = ["Answer 1", "Answer 2", "Answer 3"]

    questions = ["Question 1", "Question 2", "Question 3"]
    answers = stage.batch_query(questions)

    assert len(answers) == 3
    assert answers == ["Answer 1", "Answer 2", "Answer 3"]
    stage.chain.batch.assert_called_once_with(questions)


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_batch_query_empty_list(mock_chat_openai, mock_retriever, logger):
    """Test batch query with empty questions list."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)
    stage.chain = MagicMock()
    stage.chain.batch.return_value = []

    answers = stage.batch_query([])

    assert answers == []
    stage.chain.batch.assert_called_once_with([])


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_batch_query_logging(mock_chat_openai, mock_retriever, logger):
    """Test that batch queries are logged."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    with patch.object(logger, 'info') as mock_info:
        stage = RetrievalStage(retriever=mock_retriever, logger=logger)
        stage.chain = MagicMock()
        stage.chain.batch.return_value = ["A1", "A2"]

        stage.batch_query(["Q1", "Q2"])

        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Batch query" in msg for msg in log_messages)
        assert any("Batch completed" in msg for msg in log_messages)


# =============================================================================
# Chain Info Tests
# =============================================================================

def test_get_chain_info(mock_retriever, logger):
    """Test getting chain configuration info."""
    stage = RetrievalStage(
        retriever=mock_retriever,
        model="gpt-4",
        temperature=0.5,
        logger=logger
    )

    info = stage.get_chain_info()

    assert info["model"] == "gpt-4"
    assert info["temperature"] == 0.5
    assert info["has_retriever"] is True


def test_get_chain_info_custom_prompt(mock_retriever, logger):
    """Test chain info with custom prompt."""
    custom_prompt = "Custom: {context}\nQ: {question}"

    stage = RetrievalStage(
        retriever=mock_retriever,
        prompt_template=custom_prompt,
        logger=logger
    )

    info = stage.get_chain_info()

    assert info["has_retriever"] is True


# =============================================================================
# Prompt Template Tests
# =============================================================================

def test_default_prompt_template(mock_retriever, logger):
    """Test that default prompt template is used."""
    stage = RetrievalStage(retriever=mock_retriever, logger=logger)

    assert stage.prompt is not None
    # Default prompt should be in Korean
    assert "질문" in stage.DEFAULT_PROMPT_TEMPLATE or "답변" in stage.DEFAULT_PROMPT_TEMPLATE


def test_custom_prompt_template(mock_retriever, logger):
    """Test custom prompt template."""
    custom_prompt = "Context: {context}\nQuestion: {question}\nAnswer in English:"

    stage = RetrievalStage(
        retriever=mock_retriever,
        prompt_template=custom_prompt,
        logger=logger
    )

    assert stage.prompt is not None


# =============================================================================
# Integration-like Tests (with mocking)
# =============================================================================

@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_full_rag_flow_simulation(mock_chat_openai, mock_retriever, logger, sample_documents):
    """Simulate full RAG flow from question to answer."""
    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "서울시 대중교통 카드 요금은 1,250원입니다."
    mock_llm.invoke.return_value = mock_response
    mock_chat_openai.return_value = mock_llm

    # Mock retriever
    mock_retriever.invoke.return_value = sample_documents

    stage = RetrievalStage(retriever=mock_retriever, logger=logger)

    # This would normally call the full chain, but we're mocking it
    stage.chain = MagicMock()
    stage.chain.invoke.return_value = "서울시 대중교통 카드 요금은 1,250원입니다."

    result = stage.query_with_sources("대중교통 요금은?")

    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0


# =============================================================================
# Logging Tests
# =============================================================================

def test_retrieval_stage_logs_initialization(mock_retriever, logger):
    """Test that initialization is logged."""
    with patch.object(logger, 'info') as mock_info:
        stage = RetrievalStage(retriever=mock_retriever, logger=logger)

        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("RetrievalStage initialized" in msg for msg in log_messages)


@patch('src.rag_pipeline.stages.stage_06_retrieval.ChatOpenAI')
def test_retrieval_stage_logs_all_operations(mock_chat_openai, mock_retriever, logger, sample_documents):
    """Test comprehensive logging across all operations."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm
    mock_retriever.invoke.return_value = sample_documents

    with patch.object(logger, 'info') as mock_info:
        stage = RetrievalStage(retriever=mock_retriever, logger=logger)

        # Mock chains
        stage.chain = MagicMock()
        stage.chain.invoke.return_value = "Answer"
        stage.chain.batch.return_value = ["A1", "A2"]

        # Test all operations
        stage.query("Q1")
        stage.query_with_sources("Q2")
        stage.batch_query(["Q3", "Q4"])

        # Should have logged all operations
        log_messages = [str(call) for call in mock_info.call_args_list]
        assert any("Query" in msg for msg in log_messages)
        assert any("Batch" in msg for msg in log_messages)
