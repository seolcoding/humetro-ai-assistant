"""
Vertex AI LLM Wrapper for RAGAS Evaluation
Integrates Google Cloud Vertex AI with LiteLLM and RAGAS framework
"""

import os
from typing import Optional, Dict, Any, List
import litellm
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.outputs import LLMResult


class VertexAILLMConfig:
    """Configuration for Vertex AI models"""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        embedding_model: str = "text-embedding-004",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize Vertex AI configuration

        Args:
            model_name: Vertex AI model name (e.g., "gemini-2.5-flash", "gemini-1.5-pro")
            project_id: GCP project ID (defaults to VERTEXAI_PROJECT env var)
            location: GCP location (defaults to VERTEXAI_LOCATION env var or "us-central1")
            embedding_model: Embedding model name (e.g., "text-embedding-004")
            temperature: Model temperature (0.0 for deterministic)
            max_tokens: Maximum tokens for generation
        """
        self.model_name = model_name
        self.project_id = project_id or os.getenv("VERTEXAI_PROJECT")
        self.location = location or os.getenv("VERTEXAI_LOCATION", "us-central1")
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Validate configuration
        if not self.project_id:
            raise ValueError(
                "Vertex AI project ID not found. "
                "Set VERTEXAI_PROJECT environment variable or pass project_id"
            )

        # Set credentials if provided
        # Try service account file path from VERTEXAI_PROJECT_SERVICE_ACCOUNT or VERTEXAI_API_KEY
        credentials_path = os.getenv("VERTEXAI_PROJECT_SERVICE_ACCOUNT") or os.getenv("VERTEXAI_API_KEY")

        # If it's a file path (not an API key string), set it
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path


def create_vertex_ai_evaluator_llm(
    config: Optional[VertexAILLMConfig] = None
) -> LangchainLLMWrapper:
    """
    Create Vertex AI LLM wrapper for RAGAS evaluation

    Args:
        config: Vertex AI configuration (defaults to VertexAILLMConfig())

    Returns:
        LangchainLLMWrapper instance for RAGAS metrics

    Example:
        >>> from ragas.metrics import Faithfulness
        >>> evaluator_llm = create_vertex_ai_evaluator_llm()
        >>> faithfulness = Faithfulness(llm=evaluator_llm)
    """
    if config is None:
        config = VertexAILLMConfig()

    # Create ChatVertexAI instance
    chat_model = ChatVertexAI(
        model_name=config.model_name,
        project=config.project_id,
        location=config.location,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    # Custom is_finished_parser for Gemini models
    def gemini_is_finished_parser(response: LLMResult) -> bool:
        """Parse Gemini generation completion signals"""
        is_finished_list = []
        for g in response.flatten():
            resp = g.generations[0][0]

            # Check generation_info first
            if resp.generation_info is not None:
                finish_reason = resp.generation_info.get("finish_reason")
                if finish_reason is not None:
                    is_finished_list.append(
                        finish_reason in ["STOP", "MAX_TOKENS"]
                    )
                    continue

            # Check response_metadata as fallback
            from langchain_core.outputs import ChatGeneration
            if isinstance(resp, ChatGeneration) and resp.message is not None:
                metadata = resp.message.response_metadata
                if metadata.get("finish_reason"):
                    is_finished_list.append(
                        metadata["finish_reason"] in ["STOP", "MAX_TOKENS"]
                    )
                elif metadata.get("stop_reason"):
                    is_finished_list.append(
                        metadata["stop_reason"] in ["STOP", "MAX_TOKENS"]
                    )

            # If no finish reason found, default to True
            if not is_finished_list:
                is_finished_list.append(True)

        return all(is_finished_list)

    # Wrap with custom parser for Gemini
    return LangchainLLMWrapper(
        chat_model,
        is_finished_parser=gemini_is_finished_parser
    )


def create_vertex_ai_embeddings(
    config: Optional[VertexAILLMConfig] = None
) -> LangchainEmbeddingsWrapper:
    """
    Create Vertex AI embeddings wrapper for RAGAS evaluation

    Args:
        config: Vertex AI configuration (defaults to VertexAILLMConfig())

    Returns:
        LangchainEmbeddingsWrapper instance for RAGAS metrics

    Example:
        >>> from ragas.metrics import AnswerRelevancy
        >>> evaluator_embeddings = create_vertex_ai_embeddings()
        >>> answer_relevancy = AnswerRelevancy(embeddings=evaluator_embeddings)
    """
    if config is None:
        config = VertexAILLMConfig()

    embeddings = VertexAIEmbeddings(
        model_name=config.embedding_model,
        project=config.project_id,
        location=config.location,
    )

    return LangchainEmbeddingsWrapper(embeddings)


def create_vertex_ai_litellm_model(
    model_name: str = "gemini-2.5-flash",
    project_id: Optional[str] = None,
    location: str = "us-central1",
) -> str:
    """
    Create LiteLLM model string for Vertex AI

    This creates a model string that can be used with litellm.completion()
    for generating answers with Vertex AI models.

    Args:
        model_name: Vertex AI model name
        project_id: GCP project ID (defaults to VERTEXAI_PROJECT env var)
        location: GCP location

    Returns:
        LiteLLM model string (e.g., "vertex_ai/gemini-2.5-flash")

    Example:
        >>> from litellm import completion
        >>> model = create_vertex_ai_litellm_model()
        >>> response = completion(
        ...     model=model,
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    project_id = project_id or os.getenv("VERTEXAI_PROJECT")
    if not project_id:
        raise ValueError("Vertex AI project ID required")

    # Set environment variables for LiteLLM
    os.environ["VERTEXAI_PROJECT"] = project_id
    os.environ["VERTEXAI_LOCATION"] = location

    # Set credentials if provided
    credentials_path = os.getenv("VERTEXAI_PROJECT_SERVICE_ACCOUNT") or os.getenv("VERTEXAI_API_KEY")
    if credentials_path and os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    return f"vertex_ai/{model_name}"


def test_vertex_ai_connection(config: Optional[VertexAILLMConfig] = None) -> bool:
    """
    Test Vertex AI connection with a simple query

    Args:
        config: Vertex AI configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        print("🔍 Testing Vertex AI connection...")

        # Test LLM
        evaluator_llm = create_vertex_ai_evaluator_llm(config)
        print(f"✅ Vertex AI LLM initialized: {config.model_name if config else 'default'}")

        # Test embeddings
        evaluator_embeddings = create_vertex_ai_embeddings(config)
        print(f"✅ Vertex AI Embeddings initialized: {config.embedding_model if config else 'default'}")

        # Test LiteLLM
        if config is None:
            config = VertexAILLMConfig()

        litellm_model = create_vertex_ai_litellm_model(
            model_name=config.model_name,
            project_id=config.project_id,
            location=config.location
        )

        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": "테스트"}],
            max_tokens=10
        )

        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content:
                print(f"✅ LiteLLM Vertex AI connection successful")
                print(f"   Response: {content[:50]}...")
                return True
            else:
                print("✅ LiteLLM Vertex AI connection successful (empty response)")
                return True
        else:
            print("❌ No response from Vertex AI")
            return False

    except Exception as e:
        print(f"❌ Vertex AI connection failed: {e}")
        return False


# Available Vertex AI models
VERTEX_AI_MODELS = {
    "gemini-2.5-flash": {
        "model": "gemini-2.5-flash",
        "description": "Latest Gemini 2.5 Flash - Fast and efficient (Default)",
        "cost": "low",
    },
    "gemini-2.0-flash": {
        "model": "gemini-2.0-flash-001",
        "description": "Gemini 2.0 Flash - Fast and efficient",
        "cost": "low",
    },
    "gemini-1.5-pro": {
        "model": "gemini-1.5-pro",
        "description": "Gemini 1.5 Pro - Best quality",
        "cost": "medium",
    },
    "gemini-1.5-flash": {
        "model": "gemini-1.5-flash-001",
        "description": "Gemini 1.5 Flash - Balanced",
        "cost": "low",
    },
}


if __name__ == "__main__":
    # Test Vertex AI connection
    print("=" * 60)
    print("Vertex AI LLM Wrapper Test")
    print("=" * 60)

    # Create configuration
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0
    )

    # Test connection
    success = test_vertex_ai_connection(config)

    if success:
        print("\n🎉 Vertex AI integration ready!")
    else:
        print("\n⚠️ Please check your Vertex AI configuration")
        print("   - Set VERTEXAI_PROJECT environment variable")
        print("   - Set VERTEXAI_PROJECT_SERVICE_ACCOUNT to service account JSON path")
        print("   - Or run: gcloud auth application-default login")
