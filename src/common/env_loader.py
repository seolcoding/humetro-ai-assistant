"""
Environment variable loading and validation.

This module provides utilities for loading and validating environment variables
from .env files with comprehensive error handling.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv


class EnvLoader:
    """Centralized environment variable loading and validation."""

    @staticmethod
    def load_env(
        env_path: Optional[Path] = None,
        required_vars: Optional[List[str]] = None,
        override: bool = False
    ) -> None:
        """
        Load and validate environment variables from .env file.

        Args:
            env_path: Path to .env file (default: project_root/.env)
            required_vars: List of required environment variable names
            override: Whether to override existing environment variables

        Raises:
            FileNotFoundError: If .env file not found
            ValueError: If required variables are missing
        """
        # Determine .env path
        if env_path is None:
            from src.common.project_paths import get_project_root
            env_path = get_project_root() / ".env"

        # Check if .env exists
        if not env_path.exists():
            raise FileNotFoundError(
                f"❌ .env file not found: {env_path}\n"
                f"Please create a .env file with required environment variables."
            )

        # Load .env file
        load_dotenv(env_path, override=override)

        # Validate required variables
        if required_vars:
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                raise ValueError(
                    f"❌ Missing required environment variables: {', '.join(missing)}\n"
                    f"Please add them to {env_path}"
                )

    @staticmethod
    def get_required_env(var_name: str, default: Optional[str] = None) -> str:
        """
        Get required environment variable with optional default.

        Args:
            var_name: Name of environment variable
            default: Default value if not found

        Returns:
            Environment variable value

        Raises:
            ValueError: If variable not found and no default provided
        """
        value = os.getenv(var_name, default)
        if value is None:
            raise ValueError(
                f"❌ Required environment variable '{var_name}' not found.\n"
                f"Please set it in your .env file."
            )
        return value

    @staticmethod
    def get_env_path(var_name: str, default: Optional[str] = None) -> Path:
        """
        Get environment variable as Path object.

        Args:
            var_name: Name of environment variable
            default: Default value if not found

        Returns:
            Path object

        Raises:
            ValueError: If variable not found and no default provided
        """
        value = EnvLoader.get_required_env(var_name, default)
        return Path(value)

    @staticmethod
    def get_env_int(var_name: str, default: Optional[int] = None) -> int:
        """
        Get environment variable as integer.

        Args:
            var_name: Name of environment variable
            default: Default value if not found

        Returns:
            Integer value

        Raises:
            ValueError: If variable not found or not a valid integer
        """
        value = os.getenv(var_name)
        if value is None:
            if default is not None:
                return default
            raise ValueError(
                f"❌ Required environment variable '{var_name}' not found."
            )

        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"❌ Environment variable '{var_name}' must be an integer, got: {value}"
            )

    @staticmethod
    def get_env_bool(var_name: str, default: bool = False) -> bool:
        """
        Get environment variable as boolean.

        Treats 'true', '1', 'yes', 'on' as True (case-insensitive).

        Args:
            var_name: Name of environment variable
            default: Default value if not found

        Returns:
            Boolean value
        """
        value = os.getenv(var_name)
        if value is None:
            return default

        return value.lower() in ('true', '1', 'yes', 'on')

    @staticmethod
    def validate_env_vars(required_vars: List[str]) -> Dict[str, str]:
        """
        Validate that all required environment variables are set.

        Args:
            required_vars: List of required variable names

        Returns:
            Dictionary of variable names to values

        Raises:
            ValueError: If any required variables are missing
        """
        missing = []
        values = {}

        for var_name in required_vars:
            value = os.getenv(var_name)
            if value is None:
                missing.append(var_name)
            else:
                values[var_name] = value

        if missing:
            raise ValueError(
                f"❌ Missing required environment variables: {', '.join(missing)}\n"
                f"Please add them to your .env file."
            )

        return values


# Common RAG pipeline environment variables
RAG_REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "DATA_DIR",
    "RESULTS_DIR"
]

RAG_OPTIONAL_VARS = [
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "OLLAMA_BASE_URL"
]


def load_rag_env(require_neo4j: bool = False, require_ollama: bool = False) -> None:
    """
    Load environment variables for RAG pipeline.

    Args:
        require_neo4j: Whether Neo4j variables are required
        require_ollama: Whether Ollama variables are required

    Raises:
        FileNotFoundError: If .env file not found
        ValueError: If required variables are missing
    """
    required = RAG_REQUIRED_VARS.copy()

    if require_neo4j:
        required.extend(["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"])

    if require_ollama:
        required.append("OLLAMA_BASE_URL")

    EnvLoader.load_env(required_vars=required)
