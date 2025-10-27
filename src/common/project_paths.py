"""
Project path utilities for finding project root and common directories.

This module provides utilities to reliably locate the project root directory
from any file in the project, enabling consistent path resolution across
scripts, notebooks, and modules.
"""

from pathlib import Path
from typing import Optional


def find_project_root(
    marker_files: tuple[str, ...] = ("pyproject.toml", ".git", "README.md"),
    start_path: Optional[Path] = None,
    max_depth: int = 10
) -> Path:
    """
    Find the project root directory by searching upward for marker files.

    Args:
        marker_files: Files/directories that indicate project root
        start_path: Starting path (defaults to current file's directory)
        max_depth: Maximum depth to search upward

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If project root cannot be found

    Example:
        >>> from src.common.project_paths import find_project_root
        >>> project_root = find_project_root()
        >>> data_dir = project_root / "data"
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent

    current = start_path
    for _ in range(max_depth):
        # Check if any marker file exists
        if any((current / marker).exists() for marker in marker_files):
            return current

        # Move up one directory
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    raise RuntimeError(
        f"Could not find project root from {start_path}. "
        f"Searched for markers: {marker_files}"
    )


# Cache the project root to avoid repeated searches
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Get the project root directory (cached).

    Returns:
        Path to project root directory

    Example:
        >>> from src.common.project_paths import get_project_root
        >>> root = get_project_root()
        >>> config_path = root / "config" / "settings.yaml"
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = find_project_root()
    return _PROJECT_ROOT


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_output_dir() -> Path:
    """Get the output directory."""
    return get_project_root() / "output"


def get_config_dir() -> Path:
    """Get the src/config directory."""
    return get_project_root() / "src" / "config"


def get_cache_dir() -> Path:
    """Get the cache directory."""
    return get_project_root() / "data" / "cache"


def get_logs_dir() -> Path:
    """Get the logs directory."""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


# Convenience exports
PROJECT_ROOT = get_project_root()
DATA_DIR = get_data_dir()
OUTPUT_DIR = get_output_dir()
CONFIG_DIR = get_config_dir()
CACHE_DIR = get_cache_dir()
LOGS_DIR = get_logs_dir()


__all__ = [
    'find_project_root',
    'get_project_root',
    'get_data_dir',
    'get_output_dir',
    'get_config_dir',
    'get_cache_dir',
    'get_logs_dir',
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
    'CONFIG_DIR',
    'CACHE_DIR',
    'LOGS_DIR',
]
