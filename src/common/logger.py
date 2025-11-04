"""
Centralized logging system for RAG pipeline.

This module provides a comprehensive logging system with:
- Multiple log files (pipeline, API calls, errors, metrics)
- Structured JSON logging for analysis
- Cost tracking for API calls
- Per-experiment log directories
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import sys


class RAGLogger:
    """Centralized logging system for RAG pipeline experiments."""

    def __init__(self, experiment_name: str, log_dir: Optional[Path] = None):
        """
        Initialize RAG logger for an experiment.

        Args:
            experiment_name: Name of the experiment (used for log directory)
            log_dir: Base directory for logs (default: ./logs)
        """
        self.experiment_name = experiment_name

        # Set up log directory
        if log_dir is None:
            from src.common.project_paths import get_project_root
            log_dir = get_project_root() / "logs"

        self.log_dir = log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this session
        self.session_start = datetime.now()
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")

        # Multiple log files
        self.pipeline_log = self.log_dir / f"pipeline_{timestamp}.log"
        self.api_log = self.log_dir / f"api_calls_{timestamp}.log"
        self.error_log = self.log_dir / f"errors_{timestamp}.log"
        self.metrics_log = self.log_dir / f"metrics_{timestamp}.jsonl"

        # Initialize loggers
        self._setup_loggers()

        # API call tracking
        self.api_calls = []
        self.total_cost = 0.0

    def _setup_loggers(self):
        """Set up multiple loggers with different handlers."""
        # Main pipeline logger
        self.pipeline_logger = logging.getLogger(f"rag_pipeline.{self.experiment_name}")
        self.pipeline_logger.setLevel(logging.DEBUG)
        self.pipeline_logger.handlers.clear()

        # Pipeline file handler
        pipeline_handler = logging.FileHandler(self.pipeline_log, encoding="utf-8")
        pipeline_handler.setLevel(logging.DEBUG)
        pipeline_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        pipeline_handler.setFormatter(pipeline_formatter)
        self.pipeline_logger.addHandler(pipeline_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.pipeline_logger.addHandler(console_handler)

        # Error logger
        self.error_logger = logging.getLogger(f"rag_pipeline.{self.experiment_name}.errors")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()

        error_handler = logging.FileHandler(self.error_log, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(pipeline_formatter)
        self.error_logger.addHandler(error_handler)

        # Prevent propagation to avoid duplicate logs
        self.pipeline_logger.propagate = False
        self.error_logger.propagate = False

    def info(self, message: str):
        """Log info message."""
        self.pipeline_logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.pipeline_logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.pipeline_logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self.pipeline_logger.error(message, exc_info=exc_info)
        self.error_logger.error(message, exc_info=exc_info)

    def log_api_call(
        self,
        provider: str,
        model: str,
        operation: str,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        cached: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log API call with cost tracking.

        Args:
            provider: API provider (e.g., "openai", "neo4j", "ollama")
            model: Model name
            operation: Operation type (e.g., "embedding", "chat", "query")
            tokens: Number of tokens used
            cost: Estimated cost in USD
            cached: Whether result was from cache
            metadata: Additional metadata
        """
        call_data = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "tokens": tokens,
            "cost": cost,
            "cached": cached,
            "metadata": metadata or {}
        }

        self.api_calls.append(call_data)

        if cost:
            self.total_cost += cost

        # Write to API log
        with open(self.api_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_data, ensure_ascii=False) + "\n")

        # Log summary
        cache_status = "CACHED" if cached else "API CALL"
        cost_str = f"${cost:.6f}" if cost else "N/A"
        tokens_str = f"{tokens} tokens" if tokens else "N/A"

        self.pipeline_logger.debug(
            f"{cache_status} | {provider}/{model} | {operation} | "
            f"{tokens_str} | {cost_str}"
        )

    def log_metric(self, metric_name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Log evaluation metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata
        """
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "metric": metric_name,
            "value": value,
            "metadata": metadata or {}
        }

        # Write to metrics log (JSONL format)
        with open(self.metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_data, ensure_ascii=False) + "\n")

        self.pipeline_logger.info(f"METRIC | {metric_name}: {value}")

    def log_stage_start(self, stage_num: int, stage_name: str):
        """Log pipeline stage start."""
        self.pipeline_logger.info(f"{'='*80}")
        self.pipeline_logger.info(f"STAGE {stage_num}: {stage_name} - START")
        self.pipeline_logger.info(f"{'='*80}")

    def log_stage_end(self, stage_num: int, stage_name: str, duration_seconds: float):
        """Log pipeline stage end."""
        self.pipeline_logger.info(f"{'='*80}")
        self.pipeline_logger.info(
            f"STAGE {stage_num}: {stage_name} - COMPLETED in {duration_seconds:.2f}s"
        )
        self.pipeline_logger.info(f"{'='*80}")

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API call statistics."""
        total_calls = len(self.api_calls)
        cached_calls = sum(1 for call in self.api_calls if call["cached"])

        stats = {
            "total_calls": total_calls,
            "cached_calls": cached_calls,
            "cache_hit_rate": cached_calls / total_calls if total_calls > 0 else 0,
            "total_cost": self.total_cost,
            "calls_by_provider": {},
            "calls_by_operation": {}
        }

        # Group by provider
        for call in self.api_calls:
            provider = call["provider"]
            operation = call["operation"]

            if provider not in stats["calls_by_provider"]:
                stats["calls_by_provider"][provider] = 0
            stats["calls_by_provider"][provider] += 1

            if operation not in stats["calls_by_operation"]:
                stats["calls_by_operation"][operation] = 0
            stats["calls_by_operation"][operation] += 1

        return stats

    def print_summary(self):
        """Print session summary."""
        duration = (datetime.now() - self.session_start).total_seconds()
        stats = self.get_api_stats()

        self.pipeline_logger.info(f"\n{'='*80}")
        self.pipeline_logger.info(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        self.pipeline_logger.info(f"{'='*80}")
        self.pipeline_logger.info(f"Duration: {duration:.2f}s")
        self.pipeline_logger.info(f"Total API Calls: {stats['total_calls']}")
        self.pipeline_logger.info(f"Cached Calls: {stats['cached_calls']}")
        self.pipeline_logger.info(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        self.pipeline_logger.info(f"Total Cost: ${stats['total_cost']:.4f}")
        self.pipeline_logger.info(f"{'='*80}\n")


def get_logger(experiment_name: str, log_dir: Optional[Path] = None) -> RAGLogger:
    """
    Get or create a RAG logger for an experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Base directory for logs

    Returns:
        RAGLogger instance
    """
    return RAGLogger(experiment_name, log_dir)
