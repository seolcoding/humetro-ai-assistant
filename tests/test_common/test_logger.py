"""Tests for RAGLogger."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.common.logger import RAGLogger, get_logger


class TestRAGLogger:
    """Test RAGLogger functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_logger_initialization(self, temp_log_dir):
        """Test logger initialization."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        assert logger.experiment_name == "test_experiment"
        assert logger.log_dir.exists()
        assert (temp_log_dir / "test_experiment").exists()

    def test_log_files_created(self, temp_log_dir):
        """Test log files are created."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        # Write some logs
        logger.info("Test info message")
        logger.error("Test error message")

        # Check files exist
        assert logger.pipeline_log.exists()
        assert logger.error_log.exists()

    def test_api_call_logging(self, temp_log_dir):
        """Test API call logging."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        logger.log_api_call(
            provider="openai",
            model="text-embedding-3-small",
            operation="embedding",
            tokens=100,
            cost=0.002,
            cached=False
        )

        # Check API log exists
        assert logger.api_log.exists()

        # Check tracking
        assert len(logger.api_calls) == 1
        assert logger.total_cost == 0.002

    def test_metric_logging(self, temp_log_dir):
        """Test metric logging."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        logger.log_metric("accuracy", 0.95, metadata={"model": "gpt-4"})

        # Check metrics log exists
        assert logger.metrics_log.exists()

    def test_stage_logging(self, temp_log_dir):
        """Test stage start/end logging."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        logger.log_stage_start(1, "Data Collection")
        logger.log_stage_end(1, "Data Collection", 10.5)

        # Check logs written
        assert logger.pipeline_log.exists()

    def test_api_stats(self, temp_log_dir):
        """Test API statistics calculation."""
        logger = RAGLogger("test_experiment", log_dir=temp_log_dir)

        # Log some API calls
        logger.log_api_call("openai", "gpt-4", "chat", tokens=100, cost=0.01, cached=False)
        logger.log_api_call("openai", "gpt-4", "chat", tokens=100, cost=0.01, cached=True)
        logger.log_api_call("neo4j", "neo4j", "query", cached=False)

        stats = logger.get_api_stats()

        assert stats["total_calls"] == 3
        assert stats["cached_calls"] == 1
        assert stats["cache_hit_rate"] == pytest.approx(1/3)
        assert stats["total_cost"] == 0.02
        assert "openai" in stats["calls_by_provider"]
        assert "chat" in stats["calls_by_operation"]

    def test_get_logger_function(self, temp_log_dir):
        """Test get_logger helper function."""
        logger = get_logger("test_experiment", log_dir=temp_log_dir)

        assert isinstance(logger, RAGLogger)
        assert logger.experiment_name == "test_experiment"
