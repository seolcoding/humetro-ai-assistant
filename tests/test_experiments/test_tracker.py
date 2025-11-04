"""Tests for ExperimentTracker."""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.experiments.tracker import ExperimentTracker, StageStatus, get_tracker


class TestExperimentTracker:
    """Test ExperimentTracker functionality."""

    @pytest.fixture
    def temp_exp_dir(self):
        """Create temporary experiment directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_tracker_initialization(self, temp_exp_dir):
        """Test tracker initialization."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        assert tracker.experiment_name == "test_exp"
        assert tracker.experiment_dir.exists()
        assert tracker.stages_dir.exists()

    def test_initial_state(self, temp_exp_dir):
        """Test initial experiment state."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        assert tracker.get_current_stage() == 0
        assert not tracker.is_stage_completed(1)

        # All stages should be pending
        for i in range(1, 8):
            assert tracker.state["stages"][str(i)]["status"] == StageStatus.PENDING.value

    def test_start_stage(self, temp_exp_dir):
        """Test starting a stage."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        tracker.start_stage(1)

        assert tracker.get_current_stage() == 1
        assert tracker.state["stages"]["1"]["status"] == StageStatus.IN_PROGRESS.value
        assert tracker.state["stages"]["1"]["started_at"] is not None

    def test_complete_stage(self, temp_exp_dir):
        """Test completing a stage."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        tracker.start_stage(1)
        tracker.complete_stage(1, duration_seconds=10.5)

        assert tracker.is_stage_completed(1)
        assert tracker.state["stages"]["1"]["status"] == StageStatus.COMPLETED.value
        assert tracker.state["stages"]["1"]["duration_seconds"] == 10.5
        assert tracker.state["stages"]["1"]["completed_at"] is not None

    def test_fail_stage(self, temp_exp_dir):
        """Test marking stage as failed."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        tracker.start_stage(1)
        tracker.fail_stage(1, error="Test error")

        assert tracker.state["stages"]["1"]["status"] == StageStatus.FAILED.value
        assert tracker.state["stages"]["1"]["error"] == "Test error"

    def test_save_and_get_stage_result(self, temp_exp_dir):
        """Test saving and retrieving stage results."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        result_data = {
            "documents_processed": 100,
            "chunks_created": 500
        }

        tracker.save_stage_result(1, result_data)

        retrieved = tracker.get_stage_result(1)

        assert retrieved == result_data

    def test_get_stage_result_not_found(self, temp_exp_dir):
        """Test getting non-existent stage result."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        result = tracker.get_stage_result(1)

        assert result is None

    def test_get_next_pending_stage(self, temp_exp_dir):
        """Test getting next pending stage."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        # Initially, first stage is pending
        assert tracker.get_next_pending_stage() == 1

        # Complete first stage
        tracker.start_stage(1)
        tracker.complete_stage(1)

        # Second stage should be pending
        assert tracker.get_next_pending_stage() == 2

        # Complete all stages
        for i in range(2, 8):
            tracker.start_stage(i)
            tracker.complete_stage(i)

        # No more pending stages
        assert tracker.get_next_pending_stage() is None

    def test_save_and_get_final_results(self, temp_exp_dir):
        """Test saving and retrieving final results."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        final_results = {
            "accuracy": 0.95,
            "f1_score": 0.92,
            "total_time": 120.5
        }

        tracker.save_final_results(final_results)

        retrieved = tracker.get_final_results()

        assert retrieved == final_results

    def test_save_config(self, temp_exp_dir):
        """Test saving experiment configuration."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        config = {
            "model": "gpt-4",
            "chunk_size": 1024,
            "embedding_model": "text-embedding-3-small"
        }

        tracker.save_config(config)

        assert tracker.config == config
        assert tracker.config_file.exists()

    def test_save_metadata(self, temp_exp_dir):
        """Test saving experiment metadata."""
        tracker = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        tracker.save_metadata(
            description="Test experiment for RAG pipeline",
            tags=["test", "rag", "baseline"],
            author="Test User"
        )

        assert tracker.metadata["description"] == "Test experiment for RAG pipeline"
        assert tracker.metadata["tags"] == ["test", "rag", "baseline"]
        assert tracker.metadata["author"] == "Test User"

    def test_persistence(self, temp_exp_dir):
        """Test state persistence across instances."""
        # Create tracker and modify state
        tracker1 = ExperimentTracker("test_exp", base_dir=temp_exp_dir)
        tracker1.start_stage(1)
        tracker1.complete_stage(1, duration_seconds=5.0)

        # Create new tracker instance
        tracker2 = ExperimentTracker("test_exp", base_dir=temp_exp_dir)

        # Check state is preserved
        assert tracker2.is_stage_completed(1)
        assert tracker2.state["stages"]["1"]["duration_seconds"] == 5.0

    def test_get_tracker_function(self, temp_exp_dir):
        """Test get_tracker helper function."""
        tracker = get_tracker("test_exp", base_dir=temp_exp_dir)

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.experiment_name == "test_exp"
