#!/usr/bin/env python3
"""
Parallel Benchmark Runner
==========================

Wrapper for parallel evaluation - integrates with run_benchmark.py
Handles answer generation + parallel evaluation pipeline
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag_pipeline.answer_generator import AnswerGenerator
from src.evaluation.parallel_evaluator import ParallelEvaluationManager

logger = logging.getLogger(__name__)


class ParallelBenchmarkRunner:
    """
    병렬 평가 실행기

    run_benchmark.py와 통합되어 사용됨
    """

    def __init__(
        self,
        models: List[Dict[str, Any]],
        judge_model: str,
        max_concurrent: int = 5,
        k_documents: int = 4,
        retriever: Optional[Any] = None
    ):
        """
        Args:
            models: 모델 설정 리스트
            judge_model: 평가 모델
            max_concurrent: 최대 동시 평가 수
            k_documents: 검색 문서 수
            retriever: Custom retriever (optional)
        """
        self.models = models
        self.judge_model = judge_model
        self.max_concurrent = max_concurrent
        self.k_documents = k_documents
        self.retriever = retriever

        # Initialize components
        self.answer_generator = AnswerGenerator(
            retriever=retriever,
            k_documents=k_documents
        )

        self.eval_manager = ParallelEvaluationManager(
            judge_model=judge_model,
            max_concurrent=max_concurrent
        )

    def run_benchmark(
        self,
        questions: List[Dict[str, Any]],
        output_dir: Path,
        use_fixed_context: bool = True
    ) -> Dict[str, Any]:
        """
        병렬 평가 벤치마크 실행

        Args:
            questions: 질문 리스트
            output_dir: 결과 저장 디렉토리
            use_fixed_context: 고정 컨텍스트 사용 여부

        Returns:
            평가 결과
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*70)
        logger.info("병렬 평가 모드 (Parallel Evaluation)")
        logger.info("="*70)

        # Phase 1: 답변 생성
        logger.info("\n📝 Phase 1: 답변 생성 (순차)")
        logger.info(f"   모델: {len(self.models)}개")
        logger.info(f"   질문: {len(questions)}개")

        all_datasets = self.answer_generator.generate_all_answers(
            models=self.models,
            questions=questions,
            use_fixed_context=use_fixed_context
        )

        # Save answers checkpoint
        experiment_id = output_dir.name.split('_')[-1] if '_' in output_dir.name else "default"
        self.answer_generator.save_answers(all_datasets, output_dir, experiment_id)

        # Phase 2: 병렬 평가
        logger.info("\n🚀 Phase 2: 병렬 평가")
        logger.info(f"   최대 동시 실행: {self.max_concurrent}개")
        logger.info(f"   Judge 모델: {self.judge_model}")

        evaluation_results = self.eval_manager.evaluate_all_models(
            model_datasets=all_datasets,
            show_progress=True
        )

        # Save results
        self.eval_manager.save_results(
            evaluation_results=evaluation_results,
            output_dir=output_dir,
            experiment_id=experiment_id
        )

        # Convert to GenerationBenchmark-compatible format
        return self._convert_to_benchmark_format(evaluation_results, all_datasets)

    def _convert_to_benchmark_format(
        self,
        evaluation_results: Dict[str, Any],
        all_datasets: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Any]:
        """
        GenerationBenchmark 결과 포맷으로 변환

        기존 시스템과의 호환성 유지
        """
        results = evaluation_results.get("results", {})

        # Build compatible result structure
        benchmark_results = {
            "metadata": {
                "total_questions": len(next(iter(all_datasets.values()))["user_input"]) if all_datasets else 0,
                "models": list(results.keys()),
                "evaluation_mode": "parallel",
                "elapsed_seconds": evaluation_results.get("metadata", {}).get("elapsed_seconds", 0)
            },
            "single_hop": {},
            "multi_hop": {}
        }

        # Convert per-model results
        for model_name, result in results.items():
            if "error" in result:
                continue

            metrics = result.get("metrics", {})
            per_question = result.get("per_question_scores", [])

            model_result = {
                "raw_data": per_question,
                "metrics": metrics,
                "summary": {
                    "faithfulness": metrics.get("faithfulness", 0.0),
                    "answer_relevancy": metrics.get("answer_relevancy", 0.0),
                    "answer_correctness": metrics.get("answer_correctness", 0.0),
                    "num_questions": len(per_question)
                }
            }

            # Store in both single_hop and multi_hop for compatibility
            # (actual split would require question metadata)
            benchmark_results["single_hop"][model_name] = model_result
            benchmark_results["multi_hop"][model_name] = model_result

        return benchmark_results


def create_parallel_benchmark(
    models: List[Dict[str, Any]],
    judge_model: str,
    k_documents: int = 4,
    max_concurrent: int = 5,
    retriever: Optional[Any] = None
) -> ParallelBenchmarkRunner:
    """
    병렬 벤치마크 생성 (팩토리 함수)

    Args:
        models: 모델 설정
        judge_model: 평가 모델
        k_documents: 검색 문서 수
        max_concurrent: 최대 동시 실행
        retriever: Custom retriever

    Returns:
        ParallelBenchmarkRunner 인스턴스
    """
    return ParallelBenchmarkRunner(
        models=models,
        judge_model=judge_model,
        max_concurrent=max_concurrent,
        k_documents=k_documents,
        retriever=retriever
    )
