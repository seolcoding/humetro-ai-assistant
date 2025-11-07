"""
Experiment tracking and state management.

This module provides experiment state persistence, allowing experiments
to be resumed from any stage and tracking intermediate results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class StageStatus(Enum):
    """Pipeline stage status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentTracker:
    """
    Track experiment state and intermediate results.

    Enables resuming experiments from any stage and comparing
    results across different configurations.
    """

    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for experiments (default: ./results/experiments)
        """
        self.experiment_name = experiment_name

        # Set up experiment directory
        if base_dir is None:
            from src.common.project_paths import get_project_root
            base_dir = get_project_root() / "results" / "experiments"

        self.experiment_dir = base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # State file paths
        self.state_file = self.experiment_dir / "state.json"
        self.config_file = self.experiment_dir / "config.json"
        self.results_file = self.experiment_dir / "results.json"
        self.metadata_file = self.experiment_dir / "metadata.json"

        # Stage results directory
        self.stages_dir = self.experiment_dir / "stages"
        self.stages_dir.mkdir(exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()
        self.config: Dict[str, Any] = self._load_config()
        self.metadata: Dict[str, Any] = self._load_metadata()

    def _load_state(self) -> Dict[str, Any]:
        """Load experiment state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load state: {e}")
                return self._create_initial_state()
        return self._create_initial_state()

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load config: {e}")
                return {}
        return {}

    def _load_metadata(self) -> Dict[str, Any]:
        """Load experiment metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Failed to load metadata: {e}")
                return self._create_initial_metadata()
        return self._create_initial_metadata()

    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial experiment state."""
        return {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "current_stage": 0,
            "stages": {
                str(i): {
                    "status": StageStatus.PENDING.value,
                    "started_at": None,
                    "completed_at": None,
                    "duration_seconds": None,
                    "error": None
                }
                for i in range(1, 8)  # 7 stages
            }
        }

    def _create_initial_metadata(self) -> Dict[str, Any]:
        """Create initial experiment metadata."""
        return {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "description": "",
            "tags": [],
            "environment": {
                "python_version": None,
                "dependencies": []
            }
        }

    def _save_state(self):
        """Save state to disk."""
        self.state["updated_at"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving state: {e}")

    def _save_config(self):
        """Save configuration to disk."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")

    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._save_config()

    def save_metadata(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Save experiment metadata.

        Args:
            description: Experiment description
            tags: List of tags
            **kwargs: Additional metadata fields
        """
        if description:
            self.metadata["description"] = description
        if tags:
            self.metadata["tags"] = tags

        self.metadata.update(kwargs)
        self._save_metadata()

    def start_stage(self, stage_num: int):
        """
        Mark stage as started.

        Args:
            stage_num: Stage number (1-7)
        """
        stage_key = str(stage_num)
        self.state["stages"][stage_key]["status"] = StageStatus.IN_PROGRESS.value
        self.state["stages"][stage_key]["started_at"] = datetime.now().isoformat()
        self.state["current_stage"] = stage_num
        self._save_state()

    def complete_stage(self, stage_num: int, duration_seconds: Optional[float] = None):
        """
        Mark stage as completed.

        Args:
            stage_num: Stage number (1-7)
            duration_seconds: Stage duration in seconds
        """
        stage_key = str(stage_num)
        self.state["stages"][stage_key]["status"] = StageStatus.COMPLETED.value
        self.state["stages"][stage_key]["completed_at"] = datetime.now().isoformat()

        if duration_seconds is not None:
            self.state["stages"][stage_key]["duration_seconds"] = duration_seconds

        self._save_state()

    def fail_stage(self, stage_num: int, error: str):
        """
        Mark stage as failed.

        Args:
            stage_num: Stage number (1-7)
            error: Error message
        """
        stage_key = str(stage_num)
        self.state["stages"][stage_key]["status"] = StageStatus.FAILED.value
        self.state["stages"][stage_key]["error"] = error
        self._save_state()

    def save_stage_result(self, stage_num: int, result: Dict[str, Any]):
        """
        Save stage result.

        Args:
            stage_num: Stage number (1-7)
            result: Stage result data
        """
        result_file = self.stages_dir / f"stage_{stage_num}_result.json"

        result_data = {
            "stage": stage_num,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving stage result: {e}")

    def get_stage_result(self, stage_num: int) -> Optional[Dict[str, Any]]:
        """
        Get stage result.

        Args:
            stage_num: Stage number (1-7)

        Returns:
            Stage result or None if not found
        """
        result_file = self.stages_dir / f"stage_{stage_num}_result.json"

        if not result_file.exists():
            return None

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("result")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load stage result: {e}")
            return None

    def is_stage_completed(self, stage_num: int) -> bool:
        """
        Check if stage is completed.

        Args:
            stage_num: Stage number (1-7)

        Returns:
            True if stage is completed
        """
        stage_key = str(stage_num)
        status = self.state["stages"][stage_key]["status"]
        return status == StageStatus.COMPLETED.value

    def get_current_stage(self) -> int:
        """
        Get current stage number.

        Returns:
            Current stage number (0 if not started)
        """
        return self.state.get("current_stage", 0)

    def get_next_pending_stage(self) -> Optional[int]:
        """
        Get next pending stage number.

        Returns:
            Next pending stage number or None if all completed
        """
        for stage_num in range(1, 8):
            if not self.is_stage_completed(stage_num):
                return stage_num
        return None

    def save_final_results(self, results: Dict[str, Any]):
        """
        Save final experiment results.

        Args:
            results: Final results dictionary
        """
        results_data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving final results: {e}")

    def get_final_results(self) -> Optional[Dict[str, Any]]:
        """
        Get final experiment results.

        Returns:
            Final results or None if not found
        """
        if not self.results_file.exists():
            return None

        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("results")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load final results: {e}")
            return None

    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"Created: {self.state['created_at']}")
        print(f"Updated: {self.state['updated_at']}")
        print(f"Current Stage: {self.state['current_stage']}/7")
        print(f"\nStage Status:")

        for stage_num in range(1, 8):
            stage_key = str(stage_num)
            stage = self.state["stages"][stage_key]
            status = stage["status"]
            duration = stage.get("duration_seconds")

            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(status, "❓")

            duration_str = f"({duration:.2f}s)" if duration else ""
            print(f"  Stage {stage_num}: {status_icon} {status.upper()} {duration_str}")

        print(f"{'='*80}\n")


def get_tracker(experiment_name: str, base_dir: Optional[Path] = None) -> ExperimentTracker:
    """
    Get or create experiment tracker.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments

    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name, base_dir)
