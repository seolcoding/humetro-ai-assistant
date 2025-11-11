#!/usr/bin/env python3
"""
Test 4: Vertex AI with Concurrency Limiter (Gemini 2.5 Pro)
목적: 동시성 제어가 적용된 Vertex AI로 RAGAS 평가 검증
"""

import logging
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gemini_pro_with_limiter():
    """Gemini 2.5 Pro with concurrency limiter"""
    logger.info("=" * 60)
    logger.info("Test 4: Gemini 2.5 Pro + Concurrency Limiter")
    logger.info("=" * 60)

    # Create Vertex AI config (Gemini 2.5 Pro)
    config = VertexAILLMConfig(
        model_name="gemini-2.5-pro",  # Use Pro model
        temperature=0.0,
    )

    # Create components with max_concurrent_requests=3
    judge_llm = create_vertex_ai_evaluator_llm(
        config,
        max_concurrent_requests=3,  # Limit to 3 concurrent requests
    )
    judge_embeddings = create_vertex_ai_embeddings(config)

    logger.info(f"✅ Judge LLM: {config.model_name} (max_concurrent=3)")
    logger.info(f"✅ Judge embeddings created")

    # Create larger dataset (10 questions)
    questions = [
        {
            "user_input": f"질문 {i}: 교통 관련 문의입니다.",
            "response": f"답변 {i}: 교통카드는 편의점에서 충전 가능합니다.",
            "reference": f"참조 {i}: 교통카드 충전 방법 안내",
            "retrieved_contexts": [
                f"컨텍스트 {i}-1: 편의점 충전 가능",
                f"컨텍스트 {i}-2: 지하철역 무인충전기",
                f"컨텍스트 {i}-3: 모바일 앱 사용 가능",
            ],
        }
        for i in range(10)
    ]

    dataset = EvaluationDataset.from_list(questions)

    logger.info(f"📊 Dataset size: {len(dataset)}")

    # Create metrics
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        AnswerCorrectness(llm=judge_llm),
    ]

    logger.info(f"📊 Metrics: {[m.__class__.__name__ for m in metrics]}")

    # Run evaluation
    try:
        logger.info("📤 Starting RAGAS evaluation with Gemini 2.5 Pro...")
        logger.info("   Expected: No BlockingIOError with concurrency limiter")

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
        )

        logger.info("✅ RAGAS evaluation completed!")
        logger.info(f"\n📊 Results:\n{results}")

        # Check for empty responses
        df = results.to_pandas()
        null_count = df.isnull().sum().sum()

        if null_count > 0:
            logger.error(f"❌ Found {null_count} null/empty responses")
            logger.error(f"\n{df}")
        else:
            logger.info(f"✅ No empty responses - concurrency limiter working!")

        # Print detailed stats
        logger.info("\n📈 Detailed Stats:")
        logger.info(f"   Faithfulness: {df['faithfulness'].mean():.4f} (±{df['faithfulness'].std():.4f})")
        logger.info(f"   Answer Relevancy: {df['answer_relevancy'].mean():.4f} (±{df['answer_relevancy'].std():.4f})")
        logger.info(f"   Answer Correctness: {df['answer_correctness'].mean():.4f} (±{df['answer_correctness'].std():.4f})")

        assert null_count == 0, f"Empty responses detected: {null_count}"

    except Exception as e:
        logger.error(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_gemini_pro_with_limiter()
