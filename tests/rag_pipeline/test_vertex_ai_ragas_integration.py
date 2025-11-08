#!/usr/bin/env python3
"""
Vertex AI + RAGAS 통합 테스트
단계별 테스트:
1. Single call (동기 단일 호출)
2. Async call (비동기 단일 호출)
3. RAGAS call (RAGAS 평가 호출)
4. RAGAS async call (RAGAS 비동기 평가 호출)
"""

import sys
import asyncio
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

# Vertex AI wrapper
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
    create_vertex_ai_litellm_model,
    test_vertex_ai_connection,
)

# LiteLLM
import litellm

# RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/vertex_ai_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """섹션 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ========================================
# 테스트 1: Single Call (동기 단일 호출)
# ========================================
def test_single_call():
    """동기 단일 호출 테스트"""
    print_section("1️⃣ TEST 1: Single Call (동기 단일 호출)")

    try:
        # Vertex AI 모델 생성
        config = VertexAILLMConfig(
            model_name="gemini-2.5-flash",
            temperature=0.0
        )

        litellm_model = create_vertex_ai_litellm_model(
            model_name=config.model_name,
            project_id=config.project_id,
            location=config.location
        )

        # Single completion call
        print(f"  📤 호출: {litellm_model}")
        response = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
                {"role": "user", "content": "지하철 1호선의 첫차 시간은 언제인가요?"}
            ],
            max_tokens=100,
            temperature=0.0
        )

        answer = response.choices[0].message.content
        print(f"  ✅ 응답 성공")
        if answer:
            print(f"  📝 답변: {answer[:100]}...")
        else:
            print(f"  📝 답변: (빈 응답)")

        return True

    except Exception as e:
        print(f"  ❌ 실패: {e}")
        logger.error(f"Single call failed: {e}", exc_info=True)
        return False


# ========================================
# 테스트 2: Async Call (비동기 단일 호출)
# ========================================
async def test_async_call():
    """비동기 다중 호출 테스트"""
    print_section("2️⃣ TEST 2: Async Call (비동기 다중 호출)")

    try:
        # Vertex AI 모델 생성
        config = VertexAILLMConfig(
            model_name="gemini-2.5-flash",
            temperature=0.0
        )

        litellm_model = create_vertex_ai_litellm_model(
            model_name=config.model_name,
            project_id=config.project_id,
            location=config.location
        )

        # 여러 개의 질문을 비동기로 동시 호출
        questions = [
            "지하철 2호선은 언제 개통했나요?",
            "서울 지하철 1호선의 총 역 개수는?",
            "심야버스 운행 시간은 언제인가요?",
        ]

        print(f"  📤 비동기 다중 호출: {len(questions)}개 질문")
        print(f"  🔧 모델: {litellm_model}")

        # 비동기 태스크 생성
        tasks = [
            litellm.acompletion(
                model=litellm_model,
                messages=[
                    {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100,
                temperature=0.0
            )
            for question in questions
        ]

        # 모든 비동기 호출 동시 실행
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 취합 및 출력
        print(f"  ✅ 모든 응답 완료")
        success_count = 0
        warning_count = 0

        for idx, (question, response) in enumerate(zip(questions, responses), 1):
            print(f"\n  [{idx}/{len(questions)}] 질문: {question[:30]}...")

            if isinstance(response, Exception):
                print(f"       ❌ 오류: {response}")
                logger.error(f"Async call {idx} failed: {response}")
            else:
                answer = response.choices[0].message.content
                if answer:
                    print(f"       ✅ 답변: {answer[:50]}...")
                    success_count += 1
                else:
                    print(f"       ⚠️ 경고: 빈 응답")
                    logger.warning(f"Empty response from async call {idx}")
                    warning_count += 1

        print(f"\n  📊 결과: 성공 {success_count}/{len(questions)}, 경고 {warning_count}")
        return success_count > 0

    except Exception as e:
        print(f"  ❌ 실패: {e}")
        logger.error(f"Async call failed: {e}", exc_info=True)
        return False


# ========================================
# 테스트 3: RAGAS Call (RAGAS 평가 호출)
# ========================================
def test_ragas_call():
    """RAGAS 평가 호출 테스트"""
    print_section("3️⃣ TEST 3: RAGAS Call (RAGAS 평가)")

    try:
        # Vertex AI LLM 및 Embeddings 생성
        config = VertexAILLMConfig(
            model_name="gemini-2.5-flash",
            temperature=0.0
        )

        print(f"  🔧 Evaluator LLM 생성: {config.model_name}")
        evaluator_llm = create_vertex_ai_evaluator_llm(config)

        print(f"  🔧 Evaluator Embeddings 생성: {config.embedding_model}")
        evaluator_embeddings = create_vertex_ai_embeddings(config)

        # RAGAS 메트릭 설정
        print("  🔧 RAGAS 메트릭 설정")
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            AnswerCorrectness(llm=evaluator_llm),
        ]

        # 테스트 데이터셋 생성
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
        ]

        print(f"  📊 테스트 데이터: {len(test_data)}개")
        dataset = EvaluationDataset.from_list(test_data)

        # RAGAS 평가 실행
        print("  🚀 RAGAS 평가 실행 중...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )

        print("  ✅ 평가 완료")

        # 결과 출력
        df = result.to_pandas()
        print("\n  📊 평가 결과:")
        for metric in ["faithfulness", "answer_relevancy", "answer_correctness"]:
            if metric in df.columns:
                score = df[metric].mean()
                print(f"    {metric}: {score:.3f}")

        return True

    except Exception as e:
        print(f"  ❌ 실패: {e}")
        logger.error(f"RAGAS call failed: {e}", exc_info=True)
        return False


# ========================================
# 테스트 4: RAGAS Async Call (RAGAS 비동기 평가)
# ========================================
async def test_ragas_async_call():
    """RAGAS 비동기 평가 테스트"""
    print_section("4️⃣ TEST 4: RAGAS Async Call (RAGAS 비동기 평가)")

    try:
        # Vertex AI LLM 및 Embeddings 생성
        config = VertexAILLMConfig(
            model_name="gemini-2.5-flash",
            temperature=0.0
        )

        print(f"  🔧 Evaluator LLM 생성: {config.model_name}")
        evaluator_llm = create_vertex_ai_evaluator_llm(config)

        print(f"  🔧 Evaluator Embeddings 생성: {config.embedding_model}")
        evaluator_embeddings = create_vertex_ai_embeddings(config)

        # RAGAS 메트릭 설정
        print("  🔧 RAGAS 메트릭 설정")
        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ]

        # 테스트 데이터셋 생성 (더 큰 데이터셋)
        test_data = [
            {
                "user_input": f"테스트 질문 {i}",
                "response": f"테스트 답변 {i}",
                "retrieved_contexts": [f"테스트 컨텍스트 {i}"],
            }
            for i in range(3)
        ]

        print(f"  📊 테스트 데이터: {len(test_data)}개")
        dataset = EvaluationDataset.from_list(test_data)

        # RAGAS 비동기 평가 실행
        print("  🚀 RAGAS 비동기 평가 실행 중...")

        # Note: RAGAS evaluate()는 기본적으로 동기 함수이지만
        # 내부적으로 비동기 처리를 지원합니다
        # 여기서는 asyncio.to_thread()를 사용하여 비동기 컨텍스트에서 실행
        result = await asyncio.to_thread(
            evaluate,
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm
        )

        print("  ✅ 평가 완료")

        # 결과 출력
        df = result.to_pandas()
        print("\n  📊 평가 결과:")
        for metric in ["faithfulness", "answer_relevancy"]:
            if metric in df.columns:
                score = df[metric].mean()
                print(f"    {metric}: {score:.3f}")

        return True

    except Exception as e:
        print(f"  ❌ 실패: {e}")
        logger.error(f"RAGAS async call failed: {e}", exc_info=True)
        return False


# ========================================
# 메인 테스트 실행
# ========================================
async def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "🎯" * 30)
    print("Vertex AI + RAGAS 통합 테스트 시작")
    print("🎯" * 30)

    # 연결 테스트
    print_section("0️⃣ 연결 테스트")

    try:
        config = VertexAILLMConfig()
    except ValueError as e:
        print(f"\n❌ 설정 오류: {e}")
        print("\n📝 환경 변수 설정 방법:")
        print("  1. .env 파일에 다음 추가:")
        print("     VERTEXAI_PROJECT=your-project-id")
        print("     VERTEXAI_LOCATION=us-central1")
        print("     VERTEXAI_API_KEY=/path/to/service-account.json  # 선택사항")
        print("\n  2. 또는 터미널에서 직접 설정:")
        print("     export VERTEXAI_PROJECT=your-project-id")
        print("\n  3. 또는 gcloud 인증:")
        print("     gcloud auth application-default login")
        return

    if not test_vertex_ai_connection(config):
        print("\n❌ 연결 테스트 실패. 환경 설정을 확인하세요.")
        print("  - VERTEXAI_PROJECT 환경 변수 설정")
        print("  - VERTEXAI_API_KEY 환경 변수 설정 (서비스 계정 JSON 경로)")
        print("  - 또는 gcloud auth application-default login 실행")
        return

    results = []

    # 테스트 1: Single Call
    results.append(("Single Call", test_single_call()))

    # 테스트 2: Async Call
    results.append(("Async Call", await test_async_call()))

    # 테스트 3: RAGAS Call
    results.append(("RAGAS Call", test_ragas_call()))

    # 테스트 4: RAGAS Async Call
    results.append(("RAGAS Async Call", await test_ragas_async_call()))

    # 최종 결과 출력
    print_section("📊 테스트 결과 요약")
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {test_name}: {status}")

    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️ 일부 테스트 실패")
    print("=" * 60)


def main():
    """메인 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)

        # 비동기 테스트 실행
        asyncio.run(run_all_tests())

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        logger.error(f"Main error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
