#!/usr/bin/env python3
"""
평가 전용 스크립트 - 저장된 답변 생성 결과 평가
========================================================

JSONL 형식으로 저장된 답변 생성 결과를 로드하여 RAGAS 평가 실행

워크플로우:
1. generations/ 디렉토리에서 모델별 JSONL 로드
2. RAGAS 평가 실행 (Vertex AI Gemini Judge)
3. 평가 결과를 evaluations/ 디렉토리에 JSONL로 저장

사용 예시:
---------
# Naive RAG 평가
python src/evaluation/evaluate_saved_generations.py \
    --generations-dir data/evaluation/generations/naive_rag/ \
    --judge-model vertex_ai/gemini-2.5-pro \
    --output data/evaluation/evaluations/naive_rag_eval.jsonl

# 다른 Judge 모델로 재평가
python src/evaluation/evaluate_saved_generations.py \
    --generations-dir data/evaluation/generations/naive_rag/ \
    --judge-model vertex_ai/gemini-2.5-flash \
    --output data/evaluation/evaluations/naive_rag_eval_flash.jsonl
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import evaluation module
from src.evaluation.parallel_evaluator import run_parallel_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_jsonl_generations(generations_dir: Path) -> Dict[str, Dict[str, List]]:
    """
    JSONL 형식의 답변 생성 결과 로드

    Args:
        generations_dir: 모델별 JSONL 파일이 저장된 디렉토리

    Returns:
        {
            "GPT-4o-mini": {
                "user_input": [...],
                "response": [...],
                "reference": [...],
                "retrieved_contexts": [...]
            },
            ...
        }
    """
    logger.info("="*70)
    logger.info("📂 JSONL 답변 생성 결과 로드")
    logger.info(f"   위치: {generations_dir}")
    logger.info("="*70)

    if not generations_dir.exists():
        raise FileNotFoundError(f"Directory not found: {generations_dir}")

    all_datasets = {}

    # Find all JSONL files
    jsonl_files = list(generations_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {generations_dir}")

    for jsonl_file in jsonl_files:
        # Extract model name from filename
        model_name = jsonl_file.stem.replace("_", " ").title()

        logger.info(f"   📄 {jsonl_file.name}")

        # Load JSONL
        user_inputs = []
        responses = []
        references = []
        retrieved_contexts = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                user_inputs.append(record["question"])
                responses.append(record["answer"])
                references.append(record["ground_truth"])
                retrieved_contexts.append(record["contexts"])

        dataset = {
            "user_input": user_inputs,
            "response": responses,
            "reference": references,
            "retrieved_contexts": retrieved_contexts
        }

        all_datasets[model_name] = dataset

        logger.info(f"      ✅ {len(user_inputs)}개 Q&A 로드")

    logger.info(f"\n📊 총 {len(all_datasets)}개 모델 로드 완료")

    return all_datasets


def save_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_file: Path
):
    """
    평가 결과를 JSONL로 저장

    Args:
        evaluation_results: 평가 결과 (from run_parallel_evaluation)
        output_file: 출력 JSONL 파일 경로
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("💾 평가 결과 저장")
    logger.info(f"   위치: {output_file}")
    logger.info("="*70)

    # Write summary JSON
    summary_file = output_file.with_suffix('.summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    logger.info(f"   ✅ Summary: {summary_file.name}")

    # Write detailed JSONL (question-level)
    with open(output_file, 'w', encoding='utf-8') as f:
        for model_name, model_results in evaluation_results.get("models", {}).items():
            metrics = model_results.get("metrics", {})

            # If we have individual scores, write per-question
            if "individual_scores" in model_results:
                for i, scores in enumerate(model_results["individual_scores"]):
                    record = {
                        "model": model_name,
                        "question_id": f"q{i+1:03d}",
                        **scores
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Write aggregate metrics only
                record = {
                    "model": model_name,
                    "aggregate": True,
                    **metrics
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"   ✅ Details: {output_file.name}")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="저장된 답변 생성 결과 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
---------
# Naive RAG 평가
python src/evaluation/evaluate_saved_generations.py \\
    --generations-dir data/evaluation/generations/naive_rag/ \\
    --judge-model vertex_ai/gemini-2.5-pro \\
    --output data/evaluation/evaluations/naive_rag_eval.jsonl

# KG Cypher 평가 (Flash 모델)
python src/evaluation/evaluate_saved_generations.py \\
    --generations-dir data/evaluation/generations/kg_cypher/ \\
    --judge-model vertex_ai/gemini-2.5-flash \\
    --output data/evaluation/evaluations/kg_cypher_eval_flash.jsonl
        """
    )

    # Input/Output
    parser.add_argument(
        "--generations-dir",
        type=str,
        required=True,
        help="답변 생성 결과 JSONL 디렉토리 (예: data/evaluation/generations/naive_rag/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="평가 결과 JSONL 파일 경로 (예: data/evaluation/evaluations/naive_rag_eval.jsonl)"
    )

    # Evaluation settings
    parser.add_argument(
        "--judge-model",
        default="vertex_ai/gemini-2.5-pro",
        help="평가 모델 (기본: Vertex AI Gemini 2.5 Pro)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="최대 동시 평가 수 (기본: 2)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="순차 평가 모드 (병렬 대신)"
    )

    # Misc
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    print("="*70)
    print(" "*20 + "평가 전용 스크립트")
    print("="*70)
    print(f"Generations Dir: {args.generations_dir}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"Sequential: {args.sequential}")
    print(f"Output: {args.output}")
    print()

    try:
        # 1. Load generations
        generations_dir = Path(args.generations_dir)
        all_datasets = load_jsonl_generations(generations_dir)

        # 2. Run evaluation
        eval_mode = "순차 평가" if args.sequential else "병렬 평가"
        logger.info("="*70)
        logger.info(f"🧪 {eval_mode} 시작")
        logger.info("="*70)

        evaluation_results = run_parallel_evaluation(
            model_datasets=all_datasets,
            judge_model=args.judge_model,
            max_concurrent=args.max_concurrent,
            sequential=args.sequential,
            output_dir=None,  # No checkpoint needed
            experiment_id=None
        )

        # 3. Save results
        output_file = Path(args.output)
        save_evaluation_results(evaluation_results, output_file)

        # 4. Print summary
        logger.info("="*70)
        logger.info("✅ 평가 완료!")
        logger.info("="*70)

        metadata = evaluation_results.get("metadata", {})
        logger.info(f"\n📊 평가 성능:")
        logger.info(f"  - 평가 시간: {metadata.get('elapsed_seconds', 0):.1f}초")
        logger.info(f"  - 동시 실행: {metadata.get('max_concurrent', 0)}개 모델")
        logger.info(f"  - 성공 모델: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")

        logger.info(f"\n🏆 모델별 성능:")
        for model_name, model_results in evaluation_results.get("models", {}).items():
            metrics = model_results.get("metrics", {})
            logger.info(f"\n  {model_name}:")
            logger.info(f"    - Faithfulness:        {metrics.get('faithfulness', 0):.3f}")
            logger.info(f"    - Answer Relevancy:    {metrics.get('answer_relevancy', 0):.3f}")
            logger.info(f"    - Answer Correctness:  {metrics.get('answer_correctness', 0):.3f}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자 중단")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
