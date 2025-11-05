#!/usr/bin/env python3
"""
Simple benchmark test with GPT-4o-mini only
Quick evaluation with RAGAS metrics
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "rag_pipeline"))

from generation_benchmark import GenerationBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_simple_benchmark():
    """Run a simple benchmark with GPT-4o-mini only"""

    logger.info("=" * 70)
    logger.info("GPT-4o-mini Simple Benchmark Test")
    logger.info("=" * 70)

    # 1. Configure just GPT-4o-mini
    models = [
        {
            "name": "GPT-4o-mini",
            "model": "gpt-4o-mini",
            "api_base": None
        }
    ]

    # 2. Create small test dataset (5 questions)
    test_questions = {
        "single_hop": [
            {
                "question": "서울시 대중교통 기본요금은 얼마인가요?",
                "reference_answer": "서울시 대중교통 기본요금은 지하철 1,400원, 시내버스 1,500원입니다."
            },
            {
                "question": "기후동행카드는 무엇인가요?",
                "reference_answer": "기후동행카드는 월 정액으로 서울시 대중교통을 무제한 이용할 수 있는 정기권입니다."
            },
            {
                "question": "심야버스 운행시간은 언제인가요?",
                "reference_answer": "심야버스(올빼미버스)는 밤 23:30부터 새벽 4:10까지 운행합니다."
            }
        ],
        "multi_hop": [
            {
                "question": "청소년이 기후동행카드를 사용하면 한 달에 얼마나 절약할 수 있나요?",
                "reference_answer": "청소년 기후동행카드는 월 55,000원이며, 일반 교통카드 대비 약 30-40% 절약 가능합니다."
            },
            {
                "question": "외국인이 서울에서 대중교통을 이용하려면 어떤 카드를 구매해야 하나요?",
                "reference_answer": "외국인은 T-money 카드나 SEOUL CITYPASS를 구매하여 대중교통을 이용할 수 있습니다."
            }
        ]
    }

    # 3. Initialize benchmark
    logger.info("\n📊 Initializing benchmark...")
    benchmark = GenerationBenchmark(
        models=models,
        use_fixed_context=True,  # Use retrieved contexts
        k_documents=4,           # Retrieve 4 documents per question
        judge_model="gpt-4o"     # Use GPT-4o as judge for evaluation
    )

    # Check if retriever is working
    if benchmark.retriever is None:
        logger.error("❌ Retriever not initialized! Check vector store.")
        return

    logger.info(f"✅ Retriever ready with k={benchmark.k_documents} documents")

    # 4. Run evaluation for each complexity
    all_results = {}

    for complexity, questions in test_questions.items():
        logger.info(f"\n🔍 Evaluating {complexity} questions ({len(questions)} questions)...")

        try:
            results = benchmark.evaluate_complexity_group(
                questions=questions,
                complexity=complexity
            )
            all_results[complexity] = results

            # Show quick results
            if "GPT-4o-mini" in results:
                model_results = results["GPT-4o-mini"]
                logger.info(f"\n📈 {complexity.upper()} Results for GPT-4o-mini:")

                # Check metrics
                if "metrics" in model_results:
                    metrics = model_results["metrics"]
                    for metric_name in ['faithfulness', 'answer_relevancy', 'answer_correctness']:
                        metric_val = metrics.get(metric_name, 'N/A')
                        if isinstance(metric_val, list):
                            # Handle list of values (per question)
                            avg_val = sum(metric_val) / len(metric_val) if metric_val else 0
                            logger.info(f"  - {metric_name}: {avg_val:.3f} (avg of {len(metric_val)} questions)")
                        elif metric_val != 'N/A':
                            logger.info(f"  - {metric_name}: {metric_val:.3f}")
                        else:
                            logger.info(f"  - {metric_name}: N/A")

                # Check retrieved contexts
                if "raw_data" in model_results:
                    for i, data in enumerate(model_results["raw_data"], 1):
                        contexts = data.get("retrieved_contexts", [])
                        logger.info(f"  Q{i}: Retrieved {len(contexts)} contexts")
                        if contexts:
                            logger.info(f"       Preview: {contexts[0][:100]}...")

        except Exception as e:
            logger.error(f"❌ Error evaluating {complexity}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 5. Save results
    output_dir = Path("data/evaluation/test_gpt4o_mini")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")

    # 6. Summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)

    total_questions = sum(len(q) for q in test_questions.values())
    logger.info(f"✅ Evaluated {total_questions} questions with GPT-4o-mini")

    # Calculate average metrics
    all_metrics = {"faithfulness": [], "answer_relevancy": [], "answer_correctness": []}

    for complexity_results in all_results.values():
        if "GPT-4o-mini" in complexity_results:
            metrics = complexity_results["GPT-4o-mini"].get("metrics", {})
            for key in all_metrics:
                if key in metrics and metrics[key] is not None:
                    metric_val = metrics[key]
                    if isinstance(metric_val, list):
                        # If it's a list, extend our collection
                        all_metrics[key].extend(metric_val)
                    else:
                        # If it's a single value, append it
                        all_metrics[key].append(metric_val)

    logger.info("\n📊 Overall Average Metrics:")
    for metric, values in all_metrics.items():
        if values:
            avg = sum(values) / len(values)
            logger.info(f"  - {metric}: {avg:.3f} (from {len(values)} data points)")
        else:
            logger.info(f"  - {metric}: No data")

    return all_results


if __name__ == "__main__":
    try:
        results = run_simple_benchmark()

        if results:
            logger.info("\n✅ Test completed successfully!")

            # Show one example Q&A
            logger.info("\n📝 Example Q&A:")
            for complexity_results in results.values():
                if "GPT-4o-mini" in complexity_results:
                    raw_data = complexity_results["GPT-4o-mini"].get("raw_data", [])
                    if raw_data:
                        example = raw_data[0]
                        logger.info(f"Q: {example['user_input']}")
                        logger.info(f"A: {example['response'][:200]}...")
                        logger.info(f"Contexts: {len(example.get('retrieved_contexts', []))} documents")
                        break
        else:
            logger.error("❌ No results generated")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)