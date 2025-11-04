"""Tests for EnvLoader."""

import pytest
from pathlib import Path
import tempfile
import os

from src.common.env_loader import EnvLoader, load_rag_env


class TestEnvLoader:
    """Test EnvLoader functionality."""

    @pytest.fixture
    def temp_env_file(self):
        """Create temporary .env file."""
        temp_file = Path(tempfile.mktemp(suffix=".env"))
        with open(temp_file, 'w') as f:
            f.write("TEST_VAR=test_value\n")
            f.write("TEST_INT=42\n")
            f.write("TEST_BOOL=true\n")
            f.write("OPENAI_API_KEY=sk-test-key\n")
        yield temp_file
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()

    def test_load_env(self, temp_env_file):
        """Test loading .env file."""
        EnvLoader.load_env(env_path=temp_env_file)

        assert os.getenv("TEST_VAR") == "test_value"
        assert os.getenv("TEST_INT") == "42"

    def test_load_env_with_required_vars(self, temp_env_file):
        """Test loading with required variables validation."""
        EnvLoader.load_env(
            env_path=temp_env_file,
            required_vars=["TEST_VAR", "OPENAI_API_KEY"]
        )

        assert os.getenv("TEST_VAR") == "test_value"
        assert os.getenv("OPENAI_API_KEY") == "sk-test-key"

    def test_load_env_missing_required(self, temp_env_file):
        """Test error when required variables are missing."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            EnvLoader.load_env(
                env_path=temp_env_file,
                required_vars=["MISSING_VAR"]
            )

    def test_load_env_file_not_found(self):
        """Test error when .env file not found."""
        with pytest.raises(FileNotFoundError):
            EnvLoader.load_env(env_path=Path("/nonexistent/.env"))

    def test_get_required_env(self, temp_env_file):
        """Test getting required environment variable."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_required_env("TEST_VAR")
        assert value == "test_value"

    def test_get_required_env_with_default(self, temp_env_file):
        """Test getting env variable with default."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_required_env("MISSING_VAR", default="default_value")
        assert value == "default_value"

    def test_get_required_env_missing(self, temp_env_file):
        """Test error when required env variable missing."""
        EnvLoader.load_env(env_path=temp_env_file)

        with pytest.raises(ValueError, match="Required environment variable"):
            EnvLoader.get_required_env("MISSING_VAR")

    def test_get_env_int(self, temp_env_file):
        """Test getting integer environment variable."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_env_int("TEST_INT")
        assert value == 42
        assert isinstance(value, int)

    def test_get_env_int_with_default(self, temp_env_file):
        """Test getting integer with default."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_env_int("MISSING_INT", default=100)
        assert value == 100

    def test_get_env_int_invalid(self, temp_env_file):
        """Test error when env variable is not a valid integer."""
        EnvLoader.load_env(env_path=temp_env_file)

        with pytest.raises(ValueError, match="must be an integer"):
            EnvLoader.get_env_int("TEST_VAR")  # "test_value" is not an integer

    def test_get_env_bool(self, temp_env_file):
        """Test getting boolean environment variable."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_env_bool("TEST_BOOL")
        assert value is True

    def test_get_env_bool_default(self, temp_env_file):
        """Test getting boolean with default."""
        EnvLoader.load_env(env_path=temp_env_file)

        value = EnvLoader.get_env_bool("MISSING_BOOL", default=False)
        assert value is False

    def test_get_env_path(self, temp_env_file):
        """Test getting path environment variable."""
        EnvLoader.load_env(env_path=temp_env_file)

        # Note: TEST_VAR contains "test_value" which is treated as a path
        path = EnvLoader.get_env_path("TEST_VAR")
        assert isinstance(path, Path)

    def test_validate_env_vars(self, temp_env_file):
        """Test validating multiple environment variables."""
        EnvLoader.load_env(env_path=temp_env_file)

        values = EnvLoader.validate_env_vars(["TEST_VAR", "TEST_INT"])

        assert values == {"TEST_VAR": "test_value", "TEST_INT": "42"}

    def test_validate_env_vars_missing(self, temp_env_file):
        """Test error when validating missing variables."""
        EnvLoader.load_env(env_path=temp_env_file)

        with pytest.raises(ValueError, match="Missing required environment variables"):
            EnvLoader.validate_env_vars(["MISSING_VAR"])
