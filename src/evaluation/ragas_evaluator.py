#!/usr/bin/env python3
"""
RAGAS Evaluator with Multi-Provider Support
Supports OpenAI, Ollama, and Vertex AI models
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import litellm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.outputs import LLMResult

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# Vertex AI wrapper import
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
)

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Multi-provider RAGAS evaluator

    Supports:
    - OpenAI (GPT-4, GPT-4o, GPT-4o-mini, etc.)
    - Vertex AI (Gemini 2.5 Pro, Gemini 2.5 Flash)
    - Ollama (Local models)
    """

    def __init__(
        self,
        judge_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAGAS evaluator

        Args:
            judge_config: Judge model configuration
                Example for OpenAI: {"provider": "openai", "model": "gpt-4o"}
                Example for Vertex AI: {"provider": "vertex_ai", "model": "gemini-2.5-pro"}
            embedding_config: Embedding model configuration
        """
        self.judge_config = judge_config or self._default_judge_config()
        self.embedding_config = embedding_config or self._default_embedding_config()

        # Initialize judge LLM
        self.judge_llm = self._create_judge_llm()
        self.judge_embeddings = self._create_judge_embeddings()

        # Initialize metrics
        self.metrics = self._create_metrics()

    def _default_judge_config(self) -> Dict[str, Any]:
        """Default judge configuration (GPT-4o)"""
        return {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 4096,
        }

    def _default_embedding_config(self) -> Dict[str, Any]:
        """Default embedding configuration"""
        return {
            "provider": "openai",
            "model": "text-embedding-3-small",
        }

    def _create_judge_llm(self) -> LangchainLLMWrapper:
        """Create judge LLM based on provider"""
        provider = self.judge_config.get("provider", "openai")

        if provider == "vertex_ai":
            # Vertex AI (Gemini)
            logger.info(f"Creating Vertex AI judge: {self.judge_config['model']}")

            vertex_config = VertexAILLMConfig(
                model_name=self.judge_config["model"].replace("vertex_ai/", ""),
                temperature=self.judge_config.get("temperature", 0.0),
                max_tokens=self.judge_config.get("max_tokens", 4096),
            )

            return create_vertex_ai_evaluator_llm(vertex_config)

        elif provider == "openai":
            # OpenAI
            logger.info(f"Creating OpenAI judge: {self.judge_config['model']}")

            llm = ChatOpenAI(
                model=self.judge_config["model"],
                temperature=self.judge_config.get("temperature", 0.0),
                max_tokens=self.judge_config.get("max_tokens", 4096),
            )

            return LangchainLLMWrapper(llm)

        else:
            raise ValueError(f"Unsupported judge provider: {provider}")

    def _create_judge_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Create judge embeddings based on provider"""
        provider = self.embedding_config.get("provider", "openai")

        if provider == "vertex_ai":
            # Vertex AI embeddings
            logger.info("Creating Vertex AI embeddings")

            vertex_config = VertexAILLMConfig(
                embedding_model=self.embedding_config.get("model", "text-embedding-004"),
            )

            return create_vertex_ai_embeddings(vertex_config)

        elif provider == "openai":
            # OpenAI embeddings
            logger.info(f"Creating OpenAI embeddings: {self.embedding_config['model']}")

            embeddings = OpenAIEmbeddings(
                model=self.embedding_config["model"],
            )

            return LangchainEmbeddingsWrapper(embeddings)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_metrics(self) -> List:
        """Create RAGAS metrics"""
        return [
            Faithfulness(llm=self.judge_llm),
            AnswerRelevancy(
                llm=self.judge_llm,
                embeddings=self.judge_embeddings,
            ),
            AnswerCorrectness(llm=self.judge_llm),
        ]

    def evaluate(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List] = None,
    ) -> Any:
        """
        Evaluate RAG responses using RAGAS

        Args:
            dataset: EvaluationDataset with questions, answers, contexts
            metrics: Optional custom metrics (uses default if not provided)

        Returns:
            RAGAS evaluation results
        """
        metrics_to_use = metrics or self.metrics

        logger.info(f"Evaluating with judge: {self.judge_config['model']}")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Metrics: {[m.__class__.__name__ for m in metrics_to_use]}")

        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=self.judge_llm,
        )

        return results

    @staticmethod
    def create_from_preset(preset: str = "gpt-4o") -> "RAGASEvaluator":
        """
        Create evaluator from preset configuration

        Presets:
        - "gpt-4o": GPT-4o as judge (default, high quality)
        - "gpt-4o-mini": GPT-4o-mini as judge (fast, cheaper)
        - "gemini-pro": Gemini 2.5 Pro as judge (high quality, alternative)
        - "gemini-flash": Gemini 2.5 Flash as judge (fast, cheaper)

        Args:
            preset: Preset name

        Returns:
            RAGASEvaluator instance
        """
        presets = {
            "gpt-4o": {
                "judge_config": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "embedding_config": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                },
            },
            "gpt-4o-mini": {
                "judge_config": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "embedding_config": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                },
            },
            "gemini-pro": {
                "judge_config": {
                    "provider": "vertex_ai",
                    "model": "gemini-2.5-pro",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "embedding_config": {
                    "provider": "vertex_ai",
                    "model": "text-embedding-004",
                },
            },
            "gemini-flash": {
                "judge_config": {
                    "provider": "vertex_ai",
                    "model": "gemini-2.5-flash",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "embedding_config": {
                    "provider": "vertex_ai",
                    "model": "text-embedding-004",
                },
            },
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {list(presets.keys())}"
            )

        config = presets[preset]
        return RAGASEvaluator(
            judge_config=config["judge_config"],
            embedding_config=config["embedding_config"],
        )


# Convenience functions
def create_evaluator(
    judge_provider: str = "openai",
    judge_model: str = "gpt-4o",
    **kwargs
) -> RAGASEvaluator:
    """
    Create RAGAS evaluator with simple parameters

    Args:
        judge_provider: "openai" or "vertex_ai"
        judge_model: Model name (e.g., "gpt-4o", "gemini-2.5-pro")
        **kwargs: Additional configuration

    Returns:
        RAGASEvaluator instance
    """
    judge_config = {
        "provider": judge_provider,
        "model": judge_model,
        "temperature": kwargs.get("temperature", 0.0),
        "max_tokens": kwargs.get("max_tokens", 4096),
    }

    return RAGASEvaluator(judge_config=judge_config)


def evaluate_with_preset(
    dataset: EvaluationDataset,
    preset: str = "gpt-4o",
) -> Any:
    """
    Quick evaluation with preset configuration

    Args:
        dataset: EvaluationDataset to evaluate
        preset: Preset name (gpt-4o, gpt-4o-mini, gemini-pro, gemini-flash)

    Returns:
        RAGAS evaluation results
    """
    evaluator = RAGASEvaluator.create_from_preset(preset)
    return evaluator.evaluate(dataset)
