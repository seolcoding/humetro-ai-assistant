"""
LLM Configuration for KG Builder
Supports both OpenAI and Claude models via LiteLLM
"""

import os
from google.adk.models.lite_llm import LiteLlm
from typing import Optional


# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gpt-4o-mini": {
        "model": "openai/gpt-4o-mini",
        "api_base": None,
        "api_key_env": "OPENAI_API_KEY"
    },

    # Claude models (local OpenAI-compatible endpoint)
    # Use openai/ prefix to tell LiteLLM to use OpenAI-compatible endpoint
    "claude-sonnet": {
        "model": "openai/claude-sonnet-4-5-20250929",
        "api_base": os.getenv("CLAUDE_CODE_API_BASE", "http://localhost:25292/v1"),
        "api_key_env": "CLAUDE_CODE_API_KEY"
    },
    "claude-haiku": {
        "model": "openai/claude-haiku-4-5-20251001",
        "api_base": os.getenv("CLAUDE_CODE_API_BASE", "http://localhost:25292/v1"),
        "api_key_env": "CLAUDE_CODE_API_KEY"
    },
    "claude-opus": {
        "model": "openai/claude-opus-4-1-20250805",
        "api_base": os.getenv("CLAUDE_CODE_API_BASE", "http://localhost:25292/v1"),
        "api_key_env": "CLAUDE_CODE_API_KEY"
    },
}


def get_llm(model_name: str = "claude-sonnet") -> LiteLlm:
    """
    Get LiteLLM instance configured for specified model

    Args:
        model_name: Model identifier (default: "claude-sonnet")
                   Options: "gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "claude-opus"

    Returns:
        LiteLlm instance configured for the specified model

    Example:
        >>> llm = get_llm("claude-sonnet")
        >>> # Use with Google ADK Agent
        >>> agent = Agent(model=llm, ...)
    """

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    # Set up environment for LiteLLM
    if config["api_base"]:
        os.environ["OPENAI_API_BASE"] = config["api_base"]

    # Set API key
    api_key = os.getenv(config["api_key_env"], "dummy")
    if not api_key or api_key == "dummy":
        if "claude" in model_name:
            # For Claude local endpoint, dummy key is acceptable
            api_key = "dummy"
        else:
            raise ValueError(f"API key not found in environment: {config['api_key_env']}")

    os.environ["OPENAI_API_KEY"] = api_key

    # Create LiteLLM instance
    llm = LiteLlm(model=config["model"])

    return llm


def test_llm(model_name: str = "claude-sonnet") -> bool:
    """
    Test LLM connection with a simple query

    Args:
        model_name: Model to test

    Returns:
        True if successful, False otherwise
    """
    try:
        llm = get_llm(model_name)

        response = llm.llm_client.completion(
            model=llm.model,
            messages=[{"role": "user", "content": "Are you ready?"}],
            tools=[]
        )

        print(f"✅ {model_name} is ready!")
        print(f"   Response: {response}")
        return True

    except Exception as e:
        print(f"❌ {model_name} test failed: {e}")
        return False


# Default LLM for KG building
DEFAULT_KG_MODEL = "gpt-4o"


if __name__ == "__main__":
    # Test Claude Sonnet
    print("Testing Claude Sonnet...")
    test_llm("claude-sonnet")
