#!/usr/bin/env python3
"""
Test 3: RAGAS 평가 문제 재현 테스트
목적: 실제 RAGAS evaluate() 사용 시 발생하는 문제 재현
"""

import logging
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def test_ragas_with_vertex_ai():
    """RAGAS 평가 with Vertex AI - 문제 재현"""
    logger.warning("=" * 60)
    logger.warning("Test 3: RAGAS 평가 with Vertex AI (실제 문제 재현)")
    logger.warning("=" * 60)

    # Create Vertex AI components
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    judge_llm = create_vertex_ai_evaluator_llm(config)
    judge_embeddings = create_vertex_ai_embeddings(config)

    logger.warning(f"✅ Judge LLM created: {config.model_name}")
    logger.warning(f"✅ Judge embeddings created")

    # Create sample dataset (3 questions)
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": "교통카드는 어떻게 충전하나요?",
                "response": "교통카드는 편의점이나 지하철역의 무인충전기에서 충전할 수 있습니다.",
                "reference": "교통카드 충전은 편의점, 지하철역 무인충전기, 또는 모바일 앱을 통해 가능합니다.",
                "retrieved_contexts": [
                    "교통카드는 편의점에서 충전 가능합니다.",
                    "지하철역 무인충전기를 이용하세요.",
                    "모바일 앱으로도 충전할 수 있습니다.",
                ],
            },
            {
                "user_input": "심야버스는 몇 시까지 운행하나요?",
                "response": "심야버스는 자정부터 새벽 5시까지 운행합니다.",
                "reference": "심야버스는 00:00~05:00 사이에 운행되며 주요 노선을 커버합니다.",
                "retrieved_contexts": [
                    "심야버스는 자정부터 운행합니다.",
                    "새벽 5시에 운행이 종료됩니다.",
                    "주요 노선을 운행합니다.",
                ],
            },
            {
                "user_input": "분실물은 어디에 문의하나요?",
                "response": "분실물은 각 지하철역 역무실이나 종합관제실에 문의하시면 됩니다.",
                "reference": "분실물 센터는 종합관제실(02-1234-5678)로 연락하거나 역무실에 직접 방문하세요.",
                "retrieved_contexts": [
                    "역무실에서 분실물을 관리합니다.",
                    "종합관제실로 문의 가능합니다.",
                    "전화번호는 02-1234-5678입니다.",
                ],
            },
        ]
    )

    logger.warning(f"📊 Dataset size: {len(dataset)}")

    # Create metrics
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        AnswerCorrectness(llm=judge_llm),
    ]

    logger.warning(f"📊 Metrics: {[m.__class__.__name__ for m in metrics]}")

    # Run evaluation (이 부분에서 에러 발생 예상)
    try:
        logger.warning("📤 Starting RAGAS evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
        )

        logger.warning("✅ RAGAS evaluation completed!")
        logger.warning(f"\n📊 Results:\n{results}")

        # Check for empty responses
        df = results.to_pandas()
        null_count = df.isnull().sum().sum()

        if null_count > 0:
            logger.error(f"❌ Found {null_count} null/empty responses")
            logger.error(f"\n{df}")

        assert null_count == 0, f"Empty responses detected: {null_count}"

    except Exception as e:
        logger.error(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_ragas_with_vertex_ai()
