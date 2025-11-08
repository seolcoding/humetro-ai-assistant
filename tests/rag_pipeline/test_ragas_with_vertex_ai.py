#!/usr/bin/env python3
"""
RAGAS Evaluation with Vertex AI Support
Tests evaluation system with OpenAI and Vertex AI judges
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# 타임스탬프 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 로그 디렉토리 생성
log_dir = Path("logs/vertex_ai_evaluation")
log_dir.mkdir(parents=True, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"vertex_eval_{timestamp}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# RAGAS imports
from ragas import EvaluationDataset

# 우리 evaluator
from src.evaluation.ragas_evaluator import RAGASEvaluator, evaluate_with_preset


def create_test_dataset() -> EvaluationDataset:
    """테스트용 데이터셋 생성"""
    test_data = [
        {
            "user_input": "지하철역에서 반려동물 용품을 살 수 있나요?",
            "response": "네, 서울교통공사가 일부 지하철역에 반려동물 용품 자판기를 설치했습니다.",
            "retrieved_contexts": [
                "서울교통공사는 반려동물 키우는 지하철 이용고객을 위해 반려동물용품 자판기를 설치했다."
            ],
            "reference": "네, 서울교통공사가 일부 지하철역에 반려동물 용품 자판기를 설치했습니다.",
        },
        {
            "user_input": "서울시 대중교통 요금은 얼마인가요?",
            "response": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다.",
            "retrieved_contexts": [
                "서울시 대중교통 요금체계는 거리비례제를 적용하며 기본구간 요금이 적용된다."
            ],
            "reference": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다.",
        },
        {
            "user_input": "심야버스 운행시간은 언제인가요?",
            "response": "서울 심야버스(올빼미버스)는 밤 23:30부터 새벽 04:30까지 운행합니다.",
            "retrieved_contexts": [
                "심야버스는 심야시간대 대중교통 이용을 위해 특별 운행된다."
            ],
            "reference": "서울 심야버스(올빼미버스)는 밤 23:30부터 새벽 04:30까지 운행합니다.",
        },
    ]

    return EvaluationDataset.from_list(test_data)


def test_judge_comparison():
    """OpenAI와 Vertex AI Judge 비교 테스트"""
    print("\n" + "=" * 80)
    print("  🎯 RAGAS Judge 비교 테스트")
    print("  OpenAI GPT-4o vs Vertex AI Gemini 2.5 Pro")
    print("=" * 80)

    # 테스트 데이터셋
    dataset = create_test_dataset()
    print(f"\n  📊 테스트 데이터셋: {len(dataset)}개 샘플")

    judges = [
        ("gpt-4o", "OpenAI GPT-4o"),
        ("gemini-pro", "Vertex AI Gemini 2.5 Pro"),
        ("gemini-flash", "Vertex AI Gemini 2.5 Flash"),
    ]

    results = {}

    for preset, name in judges:
        print(f"\n  {'='*76}")
        print(f"  🔍 Judge: {name}")
        print(f"  {'='*76}")

        try:
            # 평가 실행
            evaluator = RAGASEvaluator.create_from_preset(preset)
            result = evaluator.evaluate(dataset)

            # 결과 저장
            results[name] = result

            # 결과 출력
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                print(f"\n  📈 평가 결과:")
                for metric in ["faithfulness", "answer_relevancy", "answer_correctness"]:
                    if metric in df.columns:
                        score = df[metric].mean()
                        print(f"    {metric:20s}: {score:.3f}")
            else:
                print(f"\n  📈 평가 결과:")
                for metric in ["faithfulness", "answer_relevancy", "answer_correctness"]:
                    if metric in result:
                        print(f"    {metric:20s}: {result[metric]:.3f}")

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            logger.error(f"Judge {name} 평가 실패: {e}", exc_info=True)

    # 결과 비교 출력
    print("\n" + "=" * 80)
    print("  📊 Judge 성능 비교")
    print("=" * 80)

    comparison_data = []
    metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]

    for metric in metrics:
        row = {"Metric": metric}

        for judge_name, result in results.items():
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                if metric in df.columns:
                    score = df[metric].mean()
                    row[judge_name] = f"{score:.3f}"
            else:
                if metric in result:
                    row[judge_name] = f"{result[metric]:.3f}"

        comparison_data.append(row)

    # DataFrame으로 출력
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("  ✅ 테스트 완료")
    print("=" * 80)

    return results


def test_custom_judge():
    """커스텀 Judge 설정 테스트"""
    print("\n" + "=" * 80)
    print("  🔧 커스텀 Judge 설정 테스트")
    print("=" * 80)

    dataset = create_test_dataset()

    # OpenAI Judge
    print("\n  1️⃣ OpenAI GPT-4o-mini Judge")
    evaluator_openai = RAGASEvaluator(
        judge_config={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2048,
        }
    )

    result_openai = evaluator_openai.evaluate(dataset)
    print(f"    ✅ 평가 완료")

    # Vertex AI Judge
    print("\n  2️⃣ Vertex AI Gemini 2.5 Pro Judge")
    evaluator_vertex = RAGASEvaluator(
        judge_config={
            "provider": "vertex_ai",
            "model": "gemini-2.5-pro",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        embedding_config={
            "provider": "vertex_ai",
            "model": "text-embedding-004",
        }
    )

    result_vertex = evaluator_vertex.evaluate(dataset)
    print(f"    ✅ 평가 완료")

    print("\n  ✅ 커스텀 설정 테스트 완료")


def test_quick_evaluation():
    """빠른 평가 함수 테스트"""
    print("\n" + "=" * 80)
    print("  ⚡ 빠른 평가 함수 테스트")
    print("=" * 80)

    dataset = create_test_dataset()

    # Preset 사용
    print("\n  Preset 'gpt-4o' 사용...")
    result = evaluate_with_preset(dataset, preset="gpt-4o")
    print(f"    ✅ 평가 완료")

    print("\n  Preset 'gemini-pro' 사용...")
    result = evaluate_with_preset(dataset, preset="gemini-pro")
    print(f"    ✅ 평가 완료")

    print("\n  ✅ 빠른 평가 테스트 완료")


def main():
    """메인 함수"""
    try:
        print("\n" + "🎯" * 40)
        print("  Vertex AI RAGAS 통합 테스트")
        print("🎯" * 40)

        # 1. Judge 비교 테스트
        test_judge_comparison()

        # 2. 커스텀 Judge 설정 테스트
        test_custom_judge()

        # 3. 빠른 평가 함수 테스트
        test_quick_evaluation()

        print("\n" + "=" * 80)
        print("  🎉 모든 테스트 완료!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        logger.error(f"Main error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
