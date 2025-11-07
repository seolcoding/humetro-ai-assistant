#!/usr/bin/env python3
"""
병렬 평가 시스템 테스트
======================

작은 규모로 전체 파이프라인 테스트
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.evaluation.parallel_evaluator import run_parallel_evaluation
from src.rag_pipeline.answer_generator import generate_answers_only

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parallel_evaluation_small():
    """소규모 테스트 (2개 모델, 3개 질문)"""
    logger.info("="*70)
    logger.info("병렬 평가 시스템 테스트")
    logger.info("="*70)

    # Test models (작은 모델 사용)
    test_models = [
        {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api_base": None},
        {"name": "GPT-4o-mini-2", "model": "gpt-4o-mini", "api_base": None}  # Same model, different name for testing
    ]

    # Test questions
    test_questions = [
        {
            "question": "서울 지하철 1호선의 운행 시간은?",
            "ground_truth": "서울 지하철 1호선은 오전 5시 30분부터 자정까지 운행합니다."
        },
        {
            "question": "서울 버스 노선은 몇 개인가요?",
            "ground_truth": "서울 버스 노선은 약 400여 개입니다."
        },
        {
            "question": "교통카드는 어디서 충전할 수 있나요?",
            "ground_truth": "교통카드는 편의점, 지하철역, 은행 등에서 충전할 수 있습니다."
        }
    ]

    logger.info(f"\n📊 테스트 설정:")
    logger.info(f"  - 모델: {len(test_models)}개")
    logger.info(f"  - 질문: {len(test_questions)}개")

    # Phase 1: Generate answers
    logger.info("\n" + "="*70)
    logger.info("Phase 1: 답변 생성")
    logger.info("="*70)

    all_datasets = generate_answers_only(
        models=test_models,
        questions=test_questions,
        use_fixed_context=False  # No retrieval for testing
    )

    logger.info(f"\n✅ 답변 생성 완료: {len(all_datasets)} 모델")
    for model_name, dataset in all_datasets.items():
        logger.info(f"  - {model_name}: {len(dataset['user_input'])} 답변")

    # Phase 2: Parallel evaluation
    logger.info("\n" + "="*70)
    logger.info("Phase 2: 병렬 평가")
    logger.info("="*70)

    evaluation_results = run_parallel_evaluation(
        model_datasets=all_datasets,
        judge_model="gpt-4o-mini",  # Use cheaper model for testing
        max_concurrent=2
    )

    # Print results
    logger.info("\n" + "="*70)
    logger.info("평가 결과")
    logger.info("="*70)

    results = evaluation_results.get("results", {})
    for model_name, result in results.items():
        if "error" in result:
            logger.error(f"❌ {model_name}: {result['error']}")
            continue

        metrics = result.get("metrics", {})
        logger.info(f"\n{model_name}:")
        logger.info(f"  Faithfulness: {metrics.get('faithfulness', 0):.3f}")
        logger.info(f"  Answer Relevancy: {metrics.get('answer_relevancy', 0):.3f}")
        logger.info(f"  Answer Correctness: {metrics.get('answer_correctness', 0):.3f}")

    # Performance metrics
    metadata = evaluation_results.get("metadata", {})
    logger.info("\n성능 지표:")
    logger.info(f"  - 실행 시간: {metadata.get('elapsed_seconds', 0):.1f}초")
    logger.info(f"  - 성공 모델: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")

    logger.info("\n" + "="*70)
    logger.info("✅ 테스트 완료!")
    logger.info("="*70)


if __name__ == "__main__":
    test_parallel_evaluation_small()
