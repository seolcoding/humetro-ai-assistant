#!/usr/bin/env python3
"""
100-Question Generation Benchmark
==================================

실험 스크립트:
1. RAGAS를 사용해 100개 질문 생성
2. Single-hop (25개) 및 Multi-hop (25개)로 분류
3. GPT-4o-mini와 Ollama 모델들의 생성 능력 평가
4. Retrieval이 아닌 Generation 벤치마크에 초점
5. 향후 KG-RAG 실험을 위한 재사용 가능한 질문 뱅크 생성

Note: 이 스크립트로 생성된 테스트 질문은 KG 구축 후 재활용될 예정
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import random
from tqdm import tqdm

# Project imports
from testset_generator import CachedTestsetGenerator, TestsetConfig
from question_classifier import QuestionComplexityClassifier, QuestionClassification
from generation_benchmark import GenerationBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_100_questions(
    force_regenerate: bool = False,
    num_documents: int = 100,
    testset_size: int = 100
) -> Tuple[TestsetConfig, pd.DataFrame]:
    """
    Generate 100 questions using RAGAS testset generator

    Args:
        force_regenerate: Force regeneration even if cache exists
        num_documents: Number of documents to sample from
        testset_size: Target number of questions to generate

    Returns:
        Tuple of (config, dataframe with questions)
    """
    logger.info("="*70)
    logger.info("질문 생성 시작")
    logger.info("="*70)

    generator = CachedTestsetGenerator()

    config, testset_df = generator.generate_or_load(
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        embedding_model="text-embedding-3-small",
        document_source="data/crawled/seoul_traffic/markdown_deduplicated",
        num_documents=num_documents,
        testset_size=testset_size,
        language="korean",
        force_regenerate=force_regenerate,
        use_korean_personas=True,
        is_latest=False,
        is_benchmark=True,
        description="100-question generation benchmark for single/multi-hop analysis"
    )

    logger.info(f"✅ 생성된 질문 수: {len(testset_df)}")
    return config, testset_df


def classify_questions(
    testset_df: pd.DataFrame,
    method: str = "hybrid"
) -> Dict[str, List[dict]]:
    """
    Classify questions into single-hop and multi-hop

    Args:
        testset_df: DataFrame with questions
        method: Classification method ("llm", "rules", or "hybrid")

    Returns:
        Dictionary with classified questions
    """
    logger.info("="*70)
    logger.info("질문 분류 시작")
    logger.info("="*70)

    classifier = QuestionComplexityClassifier(method=method)
    classifications = classifier.classify_batch(testset_df)

    # Group by classification
    single_hop = []
    multi_hop = []

    for idx, (_, row) in enumerate(testset_df.iterrows()):
        if idx >= len(classifications):
            break

        classification = classifications[idx]

        question_data = {
            "id": idx + 1,
            "question": row.get("user_input", ""),
            "reference_answer": row.get("reference", ""),
            "reference_contexts": row.get("reference_contexts", []),
            "classification": classification.classification,
            "confidence": classification.confidence,
            "features": classification.features,
            "reasoning": classification.classifier_reasoning
        }

        if classification.classification == "single_hop":
            single_hop.append(question_data)
        else:
            multi_hop.append(question_data)

    # Get statistics
    stats = classifier.get_statistics(classifications)

    logger.info(f"분류 완료:")
    if stats['total'] > 0:
        logger.info(f"  - Single-hop: {len(single_hop)} ({stats['single_hop_percentage']:.1f}%)")
        logger.info(f"  - Multi-hop: {len(multi_hop)} ({stats['multi_hop_percentage']:.1f}%)")
        logger.info(f"  - 평균 신뢰도: {stats['avg_confidence']:.3f}")
    else:
        logger.warning("  ⚠️ 생성된 질문이 없습니다!")

    return {
        "single_hop": single_hop,
        "multi_hop": multi_hop,
        "classification_metadata": {
            "method": method,
            "total_questions": len(testset_df),
            "single_hop_count": len(single_hop),
            "multi_hop_count": len(multi_hop),
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    }


def balance_question_groups(
    classified: Dict[str, List[dict]],
    target_per_group: int = 25
) -> Dict[str, List[dict]]:
    """
    Balance to exactly 25 single-hop and 25 multi-hop questions

    Strategy:
    - If too many: Sample highest confidence questions
    - If too few: Log warning and use all available

    Args:
        classified: Classified questions
        target_per_group: Target number per group (default: 25)

    Returns:
        Balanced question groups
    """
    logger.info("="*70)
    logger.info(f"질문 균형 조정 (각 {target_per_group}개)")
    logger.info("="*70)

    balanced = {}

    for group in ["single_hop", "multi_hop"]:
        questions = classified.get(group, [])

        if len(questions) > target_per_group:
            # Sort by confidence and sample top N
            sorted_questions = sorted(
                questions,
                key=lambda x: x["confidence"],
                reverse=True
            )
            balanced[group] = sorted_questions[:target_per_group]
            logger.info(f"  {group}: {len(questions)} → {target_per_group} (신뢰도 기준 선택)")

        elif len(questions) < target_per_group:
            balanced[group] = questions
            logger.warning(
                f"  ⚠️ {group}: {len(questions)}개만 사용 가능 "
                f"(목표: {target_per_group})"
            )
        else:
            balanced[group] = questions
            logger.info(f"  {group}: {len(questions)} (목표 달성)")

    # Update classification metadata
    balanced["classification_metadata"] = classified.get("classification_metadata", {})
    balanced["classification_metadata"]["balanced"] = True
    balanced["classification_metadata"]["target_per_group"] = target_per_group
    balanced["classification_metadata"]["final_counts"] = {
        "single_hop": len(balanced.get("single_hop", [])),
        "multi_hop": len(balanced.get("multi_hop", []))
    }

    return balanced


def save_reusable_question_bank(
    classified_questions: Dict[str, List],
    output_dir: Path,
    timestamp: str
) -> Path:
    """
    Save question bank in reusable format for future KG-RAG experiments

    이 질문 뱅크는 KG 구축 후 동일한 질문으로 비교 평가할 때 재사용됨

    Args:
        classified_questions: Classified and balanced questions
        output_dir: Output directory
        timestamp: Timestamp for versioning

    Returns:
        Path to saved question bank file
    """
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
            "usage": "Generation quality benchmark and future KG-RAG evaluation",
            "note": "이 질문들은 KG 구축 후 동일한 벤치마크로 재사용될 예정입니다"
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

    # Also save as "latest" for easy access
    latest_file = output_dir / "question_bank_latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(question_bank, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 질문 뱅크 저장: {output_file}")
    logger.info(f"💾 최신 버전 링크: {latest_file}")

    return output_file


def print_sample_questions(classified_questions: Dict[str, List], n: int = 3):
    """Print sample questions for verification"""

    print("\n" + "="*70)
    print("샘플 질문")
    print("="*70)

    for group in ["single_hop", "multi_hop"]:
        questions = classified_questions.get(group, [])
        if not questions:
            continue

        print(f"\n{group.replace('_', ' ').upper()} (처음 {n}개):")
        print("-"*50)

        for q in questions[:n]:
            print(f"\n[ID: {q['id']}]")
            print(f"Q: {q['question']}")
            print(f"분류 신뢰도: {q['confidence']:.2f}")
            print(f"이유: {q['reasoning'][:100]}...")


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="100-Question Generation Benchmark for RAG Systems"
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Force regeneration of questions even if cache exists"
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=100,
        help="Number of source documents to use (default: 100)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=100,
        help="Target number of questions to generate (default: 100)"
    )
    parser.add_argument(
        "--questions-per-group",
        type=int,
        default=25,
        help="Number of questions per complexity group (default: 25)"
    )
    parser.add_argument(
        "--classification-method",
        choices=["llm", "rules", "hybrid"],
        default="hybrid",
        help="Question classification method (default: hybrid)"
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip model benchmark evaluation (only generate and classify)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "ollama/exaone3.5:7.8b"],
        help="Models to benchmark (default: gpt-4o-mini and EXAONE)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path("data/evaluation/generation_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    print("="*70)
    print("100-Question Generation Benchmark".center(70))
    print("="*70)
    print(f"\n설정:")
    print(f"  - 문서 수: {args.num_documents}")
    print(f"  - 목표 질문 수: {args.target_size}")
    print(f"  - 그룹당 질문 수: {args.questions_per_group}")
    print(f"  - 분류 방법: {args.classification_method}")
    print(f"  - 모델 벤치마크: {'건너뜀' if args.skip_benchmark else '실행'}")

    try:
        # Step 1: Generate questions
        print("\n" + "="*70)
        print("[Step 1/5] 질문 생성")
        print("="*70)
        config, testset_df = generate_100_questions(
            force_regenerate=args.force_generate,
            num_documents=args.num_documents,
            testset_size=args.target_size
        )

        # Check if any questions were generated
        if len(testset_df) == 0:
            logger.error("❌ 질문이 생성되지 않았습니다!")
            logger.error("해결 방법:")
            logger.error("  1. --num-documents 값을 늘려보세요 (최소 20개 이상 권장)")
            logger.error("  2. --target-size 값을 줄여보세요")
            logger.error("  3. 소스 문서를 확인하세요: data/crawled/seoul_traffic/markdown_deduplicated")
            sys.exit(1)

        # Save raw testset
        raw_file = output_dir / f"raw_testset_{timestamp.replace(':', '-')}.json"
        testset_df.to_json(raw_file, orient='records', force_ascii=False, indent=2)
        logger.info(f"원본 테스트셋 저장: {raw_file}")

        # Step 2: Classify questions
        print("\n" + "="*70)
        print("[Step 2/5] 질문 분류 (Single-hop vs Multi-hop)")
        print("="*70)
        classified = classify_questions(
            testset_df,
            method=args.classification_method
        )

        # Save classification results
        class_file = output_dir / f"classification_{timestamp.replace(':', '-')}.json"
        with open(class_file, 'w', encoding='utf-8') as f:
            json.dump(classified, f, ensure_ascii=False, indent=2)
        logger.info(f"분류 결과 저장: {class_file}")

        # Step 3: Balance question groups
        print("\n" + "="*70)
        print(f"[Step 3/5] 균형 조정 (각 {args.questions_per_group}개)")
        print("="*70)
        balanced = balance_question_groups(
            classified,
            target_per_group=args.questions_per_group
        )

        # Save balanced questions
        balanced_file = output_dir / f"balanced_{timestamp.replace(':', '-')}.json"
        with open(balanced_file, 'w', encoding='utf-8') as f:
            json.dump(balanced, f, ensure_ascii=False, indent=2)
        logger.info(f"균형 조정 결과 저장: {balanced_file}")

        # Print sample questions
        print_sample_questions(balanced)

        # Step 4: Save reusable question bank
        print("\n" + "="*70)
        print("[Step 4/5] 재사용 가능한 질문 뱅크 저장")
        print("="*70)
        question_bank_file = save_reusable_question_bank(
            balanced,
            output_dir,
            timestamp
        )

        # Step 5: Run generation benchmark (optional)
        if not args.skip_benchmark:
            print("\n" + "="*70)
            print("[Step 5/5] Generation 벤치마크 실행")
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

            # Run benchmark
            benchmark = GenerationBenchmark(
                models=model_configs,
                use_fixed_context=True,
                k_documents=4,
                judge_model="gpt-5"  # GPT-5를 기본 judge model로 사용 (temperature 문제 해결됨)
            )

            results = benchmark.run_benchmark(balanced, output_dir)

            print("\n" + "="*70)
            print("벤치마크 완료!")
            print("="*70)
        else:
            print("\n[Step 5/5] 벤치마크 건너뜀 (--skip-benchmark)")

        # Final summary
        print("\n" + "="*70)
        print("✅ 실험 완료!")
        print("="*70)
        print(f"\n📁 결과 디렉토리: {output_dir}")
        print(f"📄 질문 뱅크: {question_bank_file}")
        print(f"\n생성된 파일:")
        print(f"  - raw_testset_{timestamp.replace(':', '-')}.json")
        print(f"  - classification_{timestamp.replace(':', '-')}.json")
        print(f"  - balanced_{timestamp.replace(':', '-')}.json")
        print(f"  - question_bank_latest.json (재사용용)")

        if not args.skip_benchmark:
            print(f"  - benchmark_results_{timestamp.replace(':', '-')}.json")
            print(f"  - summary_report_{timestamp.replace(':', '-')}.md")

        print("\n💡 다음 단계:")
        print("1. KG 구축 완료 후 동일한 질문 뱅크로 평가")
        print("2. 전통적 RAG vs KG-RAG 성능 비교")
        print("3. Single-hop vs Multi-hop 복잡도별 개선도 분석")

    except Exception as e:
        logger.error(f"실험 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()