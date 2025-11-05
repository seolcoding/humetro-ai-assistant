#!/usr/bin/env python3
"""
Test script to verify GenerationBenchmark fix
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_benchmark_initialization():
    """Test that GenerationBenchmark initializes correctly"""
    logger.info("=" * 70)
    logger.info("TEST: GenerationBenchmark Initialization")
    logger.info("=" * 70)

    try:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent / "src" / "rag_pipeline"))

        from generation_benchmark import GenerationBenchmark

        # Test models configuration
        models = [
            {
                "name": "GPT-4o-mini",
                "model": "gpt-4o-mini",
                "api_base": None
            }
        ]

        # Initialize benchmark
        logger.info("\n1️⃣ Creating GenerationBenchmark instance...")
        benchmark = GenerationBenchmark(
            models=models,
            use_fixed_context=True,
            k_documents=4,
            judge_model="gpt-4o"
        )

        # Check retriever
        if benchmark.retriever is None:
            logger.error("❌ FAILED: Retriever is None")
            return False

        logger.info(f"✅ Retriever initialized: {type(benchmark.retriever).__name__}")

        # Test context retrieval
        logger.info("\n2️⃣ Testing context retrieval...")
        test_question = "서울시 대중교통 요금은 얼마인가요?"
        contexts = benchmark._retrieve_context(test_question)

        if not contexts:
            logger.error("❌ FAILED: No contexts retrieved")
            return False

        logger.info(f"✅ Retrieved {len(contexts)} contexts")
        logger.info(f"   First context preview: {contexts[0][:100]}...")

        # Verify contexts are not empty
        for i, ctx in enumerate(contexts, 1):
            if not ctx or not ctx.strip():
                logger.error(f"❌ FAILED: Context {i} is empty")
                return False

        logger.info("\n✅ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_small_benchmark():
    """Test small benchmark with 2 questions"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Small Benchmark Execution")
    logger.info("=" * 70)

    try:
        sys.path.insert(0, str(Path(__file__).parent / "src" / "rag_pipeline"))

        from generation_benchmark import GenerationBenchmark

        # Test models
        models = [
            {
                "name": "GPT-4o-mini",
                "model": "gpt-4o-mini",
                "api_base": None
            }
        ]

        # Initialize benchmark
        logger.info("\n1️⃣ Creating benchmark...")
        benchmark = GenerationBenchmark(
            models=models,
            use_fixed_context=True,
            k_documents=4,
            judge_model="gpt-4o"
        )

        # Create small test questions
        test_questions = {
            "single_hop": [
                {
                    "question": "서울시 대중교통 요금은 얼마인가요?",
                    "reference_answer": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다."
                },
                {
                    "question": "기후동행카드는 무엇인가요?",
                    "reference_answer": "기후동행카드는 서울시 대중교통 무제한 정기권입니다."
                }
            ]
        }

        # Run benchmark
        logger.info("\n2️⃣ Running benchmark on 2 questions...")
        output_dir = Path("data/evaluation/test_benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Just test the evaluation of complexity group
        results = benchmark.evaluate_complexity_group(
            questions=test_questions["single_hop"],
            complexity="single_hop"
        )

        # Check results
        if "GPT-4o-mini" not in results:
            logger.error("❌ FAILED: No results for GPT-4o-mini")
            return False

        model_results = results["GPT-4o-mini"]

        if "raw_data" not in model_results:
            logger.error("❌ FAILED: No raw_data in results")
            return False

        # Check retrieved contexts
        for i, data in enumerate(model_results["raw_data"], 1):
            contexts = data.get("retrieved_contexts", [])
            logger.info(f"\n   Question {i}: {data['user_input']}")
            logger.info(f"   Retrieved contexts: {len(contexts)}")

            if not contexts:
                logger.error(f"❌ FAILED: Question {i} has no retrieved contexts!")
                return False

            logger.info(f"   ✅ Context preview: {contexts[0][:80]}...")

        logger.info("\n✅ Benchmark test passed!")
        logger.info(f"   All {len(model_results['raw_data'])} questions have retrieved contexts")

        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Test 1: Initialization
    init_success = test_benchmark_initialization()

    # Test 2: Small benchmark (skip if init failed)
    if init_success:
        bench_success = test_small_benchmark()
    else:
        logger.warning("⚠️ Skipping benchmark test due to init failure")
        bench_success = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Initialization: {'✅ PASSED' if init_success else '❌ FAILED'}")
    logger.info(f"Small Benchmark: {'✅ PASSED' if bench_success else '❌ FAILED'}")

    sys.exit(0 if (init_success and bench_success) else 1)
