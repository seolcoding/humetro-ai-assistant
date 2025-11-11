#!/usr/bin/env python3
"""
Parallel Evaluation Module
===========================

병렬 평가를 위한 모듈 - I/O bound RAGAS 평가를 동시에 수행
RAGAS의 AsyncExecutor를 활용하여 5개 모델을 동시에 평가

Architecture:
1. Answer Generation (sequential) - 각 모델이 답변 생성
2. Parallel Evaluation (concurrent) - 5개 모델 동시 평가
3. Result Aggregation - 결과 통합 및 비교

Performance:
- Sequential: 5 models × 10 min = 50 min
- Parallel: max(model evaluation time) ≈ 10-12 min
- Speedup: ~5x
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import nest_asyncio
from dotenv import load_dotenv

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Vertex AI support
try:
    from langchain_google_vertexai import ChatVertexAI
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("langchain-google-vertexai not installed - Vertex AI models unavailable")

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AsyncEvaluator:
    """
    RAGAS 비동기 평가 래퍼

    RAGAS의 evaluate()는 내부적으로 async를 지원하므로
    여러 모델의 평가를 동시에 실행 가능
    """

    def __init__(self, judge_model: str = "gpt-5"):
        """
        Args:
            judge_model: 평가에 사용할 LLM (default: gpt-5)
                        Supports: gpt-5, gpt-4o, vertex_ai/gemini-2.5-pro, etc.
        """
        self.judge_model = judge_model

        # Setup Vertex AI environment if using vertex_ai models
        if judge_model.startswith("vertex_ai/"):
            self._setup_vertex_ai()

        self.metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ]

    def _setup_vertex_ai(self):
        """Setup Vertex AI environment variables for LiteLLM"""
        # Map VERTEXAI_* to VERTEX_* for LiteLLM compatibility
        if os.getenv("VERTEXAI_PROJECT"):
            os.environ["VERTEX_PROJECT"] = os.getenv("VERTEXAI_PROJECT")
            logger.info(f"  🔧 Vertex AI Project: {os.getenv('VERTEX_PROJECT')}")

        # Default location to us-central1 (required for Gemini 2.5 models)
        if not os.getenv("VERTEX_LOCATION"):
            os.environ["VERTEX_LOCATION"] = "us-central1"
            logger.info(f"  🌏 Vertex AI Location: {os.getenv('VERTEX_LOCATION')}")

        # Service account credentials
        service_account = os.getenv("VERTEXAI_PROJECT_SERVICE_ACCOUNT", "service_account.json")
        if Path(service_account).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account
            logger.info(f"  🔑 Service Account: {service_account}")
        else:
            logger.warning(f"  ⚠️ Service account not found: {service_account}")

        logger.info(f"  ✅ Vertex AI configured for judge model: {self.judge_model}")

    async def evaluate_single_model(
        self,
        model_name: str,
        dataset: Dict[str, List[str]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        단일 모델 평가 (비동기)

        Args:
            model_name: 모델 이름
            dataset: RAGAS Dataset format
                {
                    "user_input": ["Q1", "Q2", ...],
                    "response": ["A1", "A2", ...],
                    "reference": ["GT1", "GT2", ...],
                    "retrieved_contexts": [[C1], [C2], ...]
                }
            progress_callback: 진행 상황 콜백

        Returns:
            평가 결과
        """
        try:
            logger.info(f"🔄 {model_name} 평가 시작...")

            # Convert to HuggingFace Dataset
            hf_dataset = Dataset.from_dict(dataset)

            # Run evaluation (RAGAS handles async internally)
            # Note: We run this in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_ragas_evaluation,
                hf_dataset,
                model_name
            )

            if progress_callback:
                progress_callback(model_name)

            logger.info(f"✅ {model_name} 평가 완료")
            return result

        except Exception as e:
            logger.error(f"❌ {model_name} 평가 실패: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "metrics": {}
            }

    def _run_ragas_evaluation(self, dataset: Dataset, model_name: str) -> Dict[str, Any]:
        """
        실제 RAGAS 평가 실행 (동기)

        Note: RAGAS의 evaluate()는 내부적으로 async를 사용하지만
        호출 인터페이스는 동기이므로 thread pool에서 실행
        """
        # Configure judge LLM (support both OpenAI and Vertex AI)
        if self.judge_model.startswith("vertex_ai/"):
            if not VERTEX_AI_AVAILABLE:
                raise ImportError(
                    "langchain-google-vertexai required for Vertex AI models.\n"
                    "Install: uv add langchain-google-vertexai"
                )

            # Extract model name (vertex_ai/gemini-2.5-pro → gemini-2.5-pro)
            model_name_only = self.judge_model.replace("vertex_ai/", "")

            judge_llm = ChatVertexAI(
                model_name=model_name_only,
                project=os.getenv("VERTEX_PROJECT"),
                location=os.getenv("VERTEX_LOCATION", "asia-northeast3"),
                temperature=1.0  # Gemini 2.5 Pro default
            )
            logger.info(f"  🤖 Judge: Vertex AI {model_name_only}")
        else:
            # OpenAI or other ChatOpenAI-compatible models
            judge_llm = ChatOpenAI(model=self.judge_model)
            logger.info(f"  🤖 Judge: {self.judge_model}")

        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=judge_llm,
        )

        # Convert result to pandas DataFrame for easier access
        df = result.to_pandas()

        return {
            "model_name": model_name,
            "metrics": {
                "faithfulness": df["faithfulness"].mean() if "faithfulness" in df else 0.0,
                "answer_relevancy": df["answer_relevancy"].mean() if "answer_relevancy" in df else 0.0,
                "answer_correctness": df["answer_correctness"].mean() if "answer_correctness" in df else 0.0,
            },
            "per_question_scores": df.to_dict('records')
        }


class ParallelEvaluationManager:
    """
    병렬 평가 관리자

    여러 모델의 답변을 동시에 평가하고 결과를 통합
    """

    def __init__(self, judge_model: str = "gpt-5", max_concurrent: int = 5, request_delay: float = 0.5):
        """
        Args:
            judge_model: 평가 모델
            max_concurrent: 최대 동시 평가 수 (default: 5)
            request_delay: 요청 간 딜레이 (초, default: 0.5)
        """
        self.evaluator = AsyncEvaluator(judge_model=judge_model)
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _evaluate_with_semaphore(
        self,
        model_name: str,
        dataset: Dict[str, List[str]],
        progress_bar: Optional[tqdm] = None
    ) -> Dict[str, Any]:
        """세마포어를 사용한 평가 (동시 실행 제어)"""
        async with self.semaphore:
            # Add delay before starting evaluation to avoid rate limits
            if self.request_delay > 0:
                await asyncio.sleep(self.request_delay)

            result = await self.evaluator.evaluate_single_model(
                model_name=model_name,
                dataset=dataset,
                progress_callback=lambda name: progress_bar.update(1) if progress_bar else None
            )
            return result

    async def evaluate_all_models(
        self,
        model_datasets: Dict[str, Dict[str, List[str]]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        모든 모델 병렬 평가

        Args:
            model_datasets: {
                "GPT-4o-mini": {"user_input": [...], "response": [...], ...},
                "EXAONE-3.5": {...},
                ...
            }
            show_progress: 진행 상황 표시

        Returns:
            통합 평가 결과
        """
        logger.info(f"🚀 병렬 평가 시작 (최대 동시 실행: {self.max_concurrent})")
        logger.info(f"📊 평가 대상: {list(model_datasets.keys())}")

        # Create progress bar
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=len(model_datasets),
                desc="모델 평가 진행",
                unit="model"
            )

        # Create evaluation tasks
        tasks = [
            self._evaluate_with_semaphore(model_name, dataset, progress_bar)
            for model_name, dataset in model_datasets.items()
        ]

        # Execute all tasks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()

        if progress_bar:
            progress_bar.close()

        # Process results
        evaluation_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"평가 실패: {result}")
                continue

            model_name = result.get("model_name")
            evaluation_results[model_name] = result

        # Calculate statistics
        elapsed = (end_time - start_time).total_seconds()
        logger.info(f"✅ 병렬 평가 완료: {elapsed:.1f}초 ({len(evaluation_results)}/{len(model_datasets)} 성공)")

        return {
            "results": evaluation_results,
            "metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "elapsed_seconds": elapsed,
                "total_models": len(model_datasets),
                "successful_models": len(evaluation_results),
                "max_concurrent": self.max_concurrent
            }
        }

    def aggregate_results(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        평가 결과 통합 및 비교 테이블 생성

        Args:
            evaluation_results: evaluate_all_models()의 반환값

        Returns:
            비교 DataFrame
        """
        results = evaluation_results.get("results", {})

        rows = []
        for model_name, result in results.items():
            if "error" in result:
                continue

            metrics = result.get("metrics", {})
            rows.append({
                "Model": model_name,
                "Faithfulness": metrics.get("faithfulness", 0.0),
                "Answer Relevancy": metrics.get("answer_relevancy", 0.0),
                "Answer Correctness": metrics.get("answer_correctness", 0.0),
                "Average": sum(metrics.values()) / len(metrics) if metrics else 0.0
            })

        df = pd.DataFrame(rows)

        # Sort by average score
        if not df.empty:
            df = df.sort_values("Average", ascending=False)

        return df

    def save_results(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: Path,
        experiment_id: int
    ):
        """
        평가 결과 저장

        Args:
            evaluation_results: 평가 결과
            output_dir: 출력 디렉토리
            experiment_id: 실험 ID
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        df = self.aggregate_results(evaluation_results)
        csv_path = output_dir / f"exp_{experiment_id}_parallel_evaluation_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 비교 테이블 저장: {csv_path}")

        # Save detailed results
        import json
        json_path = output_dir / f"exp_{experiment_id}_parallel_evaluation_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 상세 결과 저장: {json_path}")

        # Print summary
        print("\n" + "="*70)
        print("병렬 평가 결과 요약")
        print("="*70)
        print(df.to_string(index=False))
        print()

        metadata = evaluation_results.get("metadata", {})
        print(f"실행 시간: {metadata.get('elapsed_seconds', 0):.1f}초")
        print(f"성공 모델: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")
        print("="*70)


def run_parallel_evaluation(
    model_datasets: Dict[str, Dict[str, List[str]]],
    judge_model: str = "gpt-5",
    max_concurrent: int = 5,
    sequential: bool = False,
    output_dir: Optional[Path] = None,
    experiment_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    병렬/순차 평가 실행 (메인 인터페이스)

    Args:
        model_datasets: 모델별 답변 데이터셋
        judge_model: 평가 모델
        max_concurrent: 최대 동시 평가 수 (sequential=False일 때만 사용)
        sequential: True이면 순차 평가, False이면 병렬 평가
        output_dir: 결과 저장 디렉토리
        experiment_id: 실험 ID

    Returns:
        평가 결과

    Example:
        >>> model_datasets = {
        ...     "GPT-4o-mini": {
        ...         "user_input": ["Q1", "Q2"],
        ...         "response": ["A1", "A2"],
        ...         "reference": ["GT1", "GT2"],
        ...         "retrieved_contexts": [["C1"], ["C2"]]
        ...     },
        ...     "EXAONE-3.5": {...}
        ... }
        >>> results = run_parallel_evaluation(model_datasets, sequential=True)
    """
    # Create manager
    if sequential:
        logger.info("🔄 순차 평가 모드 (API 안정성 우선)")
        max_concurrent = 1  # Force sequential
    else:
        logger.info(f"⚡ 병렬 평가 모드 (동시 실행: {max_concurrent})")

    manager = ParallelEvaluationManager(
        judge_model=judge_model,
        max_concurrent=max_concurrent
    )

    # Run evaluation
    results = asyncio.run(manager.evaluate_all_models(model_datasets))

    # Save results if output_dir provided
    if output_dir and experiment_id is not None:
        manager.save_results(results, output_dir, experiment_id)

    return results


if __name__ == "__main__":
    # Test example
    logging.basicConfig(level=logging.INFO)

    # Mock data for testing
    test_datasets = {
        "Model-A": {
            "user_input": ["What is RAG?", "How does it work?"],
            "response": ["RAG is...", "It works by..."],
            "reference": ["RAG stands for...", "The system..."],
            "retrieved_contexts": [["Context 1"], ["Context 2"]]
        },
        "Model-B": {
            "user_input": ["What is RAG?", "How does it work?"],
            "response": ["RAG means...", "The mechanism..."],
            "reference": ["RAG stands for...", "The system..."],
            "retrieved_contexts": [["Context 1"], ["Context 2"]]
        }
    }

    print("병렬 평가 테스트 시작...")
    results = run_parallel_evaluation(
        model_datasets=test_datasets,
        judge_model="gpt-4o-mini",  # Use cheaper model for testing
        max_concurrent=2
    )

    print("\n결과:")
    print(results)
