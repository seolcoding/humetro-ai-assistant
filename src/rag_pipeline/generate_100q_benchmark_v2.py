#!/usr/bin/env python3
"""
100-Question Generation Benchmark (Optimized Version)
=====================================================

개선사항:
1. RAGAS의 synthesizer_name 태그 활용 (재분류 불필요)
2. 벡터 스토어 경로 수정
3. 불필요한 단계 제거
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd

# Project imports
from question_generation import generate_questions
from generation_benchmark import GenerationBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_and_classify_questions(
    force_regenerate: bool = False,
    num_documents: int = 100,
    testset_size: int = 100
) -> Tuple[Any, Dict[str, List[dict]]]:
    """
    Generate questions using RAGAS and classify using built-in tags

    RAGAS synthesizer_name:
    - single_hop_specifc_query_synthesizer -> Single-hop
    - multi_context_query_synthesizer -> Multi-hop
    - reasoning_query_synthesizer -> Multi-hop
    """
    logger.info("="*70)
    logger.info("질문 생성 및 분류")
    logger.info("="*70)

    config, testset_df = generate_questions(
        model="gpt-4o-mini",
        num_documents=num_documents,
        num_questions=testset_size,
        force_regenerate=force_regenerate,
        use_korean_personas=True,
        is_latest=False,
        is_benchmark=True,
        description="100-question generation benchmark using RAGAS synthesizer types"
    )

    if len(testset_df) == 0:
        logger.error("❌ 질문이 생성되지 않았습니다!")
        logger.error("해결 방법:")
        logger.error("  1. --num-documents 값을 늘려보세요 (최소 20개 이상 권장)")
        logger.error("  2. --target-size 값을 줄여보세요")
        sys.exit(1)

    logger.info(f"✅ 생성된 질문 수: {len(testset_df)}")

    # Classify using RAGAS synthesizer_name
    single_hop = []
    multi_hop = []

    for idx, row in testset_df.iterrows():
        synthesizer = row.get("synthesizer_name", "")

        question_data = {
            "id": idx + 1,
            "question": row.get("user_input", ""),
            "reference_answer": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
            "synthesizer_type": synthesizer
        }

        # RAGAS classification
        if "single_hop" in synthesizer.lower():
            single_hop.append(question_data)
        else:  # multi_context, reasoning -> multi-hop
            multi_hop.append(question_data)

    logger.info(f"RAGAS 자동 분류 결과:")
    logger.info(f"  - Single-hop: {len(single_hop)}")
    logger.info(f"  - Multi-hop: {len(multi_hop)}")

    return config, {
        "single_hop": single_hop,
        "multi_hop": multi_hop,
        "classification_metadata": {
            "method": "ragas_synthesizer",
            "total_questions": len(testset_df),
            "single_hop_count": len(single_hop),
            "multi_hop_count": len(multi_hop),
            "timestamp": datetime.now().isoformat()
        }
    }


def balance_and_sample(
    classified: Dict[str, List[dict]],
    target_per_group: int = 25
) -> Dict[str, List[dict]]:
    """Sample target number from each group"""

    import random
    random.seed(42)  # Reproducibility

    balanced = {}

    for group in ["single_hop", "multi_hop"]:
        questions = classified.get(group, [])

        if len(questions) >= target_per_group:
            balanced[group] = random.sample(questions, target_per_group)
            logger.info(f"  {group}: {len(questions)} → {target_per_group} (랜덤 샘플링)")
        else:
            balanced[group] = questions
            logger.warning(f"  ⚠️ {group}: {len(questions)}개만 사용 가능")

    balanced["classification_metadata"] = classified.get("classification_metadata", {})
    balanced["classification_metadata"]["balanced"] = True
    balanced["classification_metadata"]["target_per_group"] = target_per_group

    return balanced


def save_question_bank(
    classified_questions: Dict[str, List],
    output_dir: Path,
    timestamp: str
) -> Path:
    """Save reusable question bank"""

    question_bank = {
        "metadata": {
            "name": "Seoul Traffic QA Benchmark v1.0",
            "version": "1.0",
            "created_at": timestamp,
            "total_questions": (
                len(classified_questions.get("single_hop", [])) +
                len(classified_questions.get("multi_hop", []))
            ),
            "complexity_distribution": {
                "single_hop": len(classified_questions.get("single_hop", [])),
                "multi_hop": len(classified_questions.get("multi_hop", []))
            },
            "domain": "Seoul Metro and Bus Transportation",
            "language": "Korean",
            "classification_method": "RAGAS synthesizer_name",
            "usage": "Generation quality benchmark and future KG-RAG evaluation"
        },
        "questions": {
            "single_hop": classified_questions.get("single_hop", []),
            "multi_hop": classified_questions.get("multi_hop", [])
        },
        "classification_metadata": classified_questions.get("classification_metadata", {})
    }

    # Save with timestamp
    output_file = output_dir / f"question_bank_v1.0_{timestamp.replace(':', '-')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(question_bank, f, ensure_ascii=False, indent=2)

    # Save as latest
    latest_file = output_dir / "question_bank_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(question_bank, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 질문 뱅크 저장: {output_file}")
    logger.info(f"💾 최신 버전: {latest_file}")

    return output_file


def main():
    """Main execution"""

    parser = argparse.ArgumentParser(
        description="100-Question Generation Benchmark (Optimized)"
    )
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--num-documents", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--questions-per-group", type=int, default=25)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "ollama/exaone3.5:7.8b"]
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path("data/evaluation/generation_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    print("="*70)
    print("100-Question Generation Benchmark (Optimized)".center(70))
    print("="*70)
    print(f"\n설정:")
    print(f"  - 문서 수: {args.num_documents}")
    print(f"  - 목표 질문 수: {args.target_size}")
    print(f"  - 그룹당 질문 수: {args.questions_per_group}")
    print(f"  - 분류 방법: RAGAS synthesizer_name (자동)")
    print(f"  - 벤치마크: {'건너뜀' if args.skip_benchmark else '실행'}")

    try:
        # Step 1: Generate & Classify (using RAGAS tags)
        print("\n" + "="*70)
        print("[Step 1/4] 질문 생성 & RAGAS 자동 분류")
        print("="*70)
        config, classified = generate_and_classify_questions(
            force_regenerate=args.force_generate,
            num_documents=args.num_documents,
            testset_size=args.target_size
        )

        # Step 2: Balance
        print("\n" + "="*70)
        print(f"[Step 2/4] 균형 조정 (각 {args.questions_per_group}개)")
        print("="*70)
        balanced = balance_and_sample(classified, args.questions_per_group)

        # Save balanced
        balanced_file = output_dir / f"balanced_{timestamp.replace(':', '-')}.json"
        with open(balanced_file, 'w', encoding='utf-8') as f:
            json.dump(balanced, f, ensure_ascii=False, indent=2)
        logger.info(f"저장: {balanced_file}")

        # Step 3: Save question bank
        print("\n" + "="*70)
        print("[Step 3/4] 재사용 가능한 질문 뱅크 저장")
        print("="*70)
        question_bank_file = save_question_bank(balanced, output_dir, timestamp)

        # Step 4: Benchmark (optional)
        if not args.skip_benchmark:
            print("\n" + "="*70)
            print("[Step 4/4] Generation 벤치마크 실행")
            print("="*70)

            # Configure models
            model_configs = []
            for model_spec in args.models:
                if model_spec == "gpt-4o-mini":
                    model_configs.append({
                        "name": "GPT-4o-mini",
                        "model": "gpt-4o-mini",
                        "api_base": None
                    })
                elif model_spec.startswith("ollama/"):
                    model_name = model_spec.split("/")[1]
                    model_configs.append({
                        "name": f"Ollama-{model_name}",
                        "model": model_spec,
                        "api_base": "http://100.95.220.92:11434"
                    })

            logger.info(f"평가 모델: {[m['name'] for m in model_configs]}")

            benchmark = GenerationBenchmark(
                models=model_configs,
                use_fixed_context=True,
                k_documents=4,
                judge_model="gpt-5"
            )

            results = benchmark.run_benchmark(balanced, output_dir)

            print("\n" + "="*70)
            print("벤치마크 완료!")
            print("="*70)
        else:
            print("\n[Step 4/4] 벤치마크 건너뜀")

        # Final summary
        print("\n" + "="*70)
        print("✅ 실험 완료!")
        print("="*70)
        print(f"\n📁 결과: {output_dir}")
        print(f"📄 질문 뱅크: {question_bank_file}")

    except Exception as e:
        logger.error(f"실험 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()