#!/usr/bin/env python3
"""
Vertex AI 동시 호출 스트레스 테스트 - Gemini 2.5 Pro
30개 질문을 비동기로 동시에 호출하여 성능 및 API 크레딧 차감 확인
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Vertex AI wrapper
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_litellm_model,
)

# LiteLLM
import litellm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/vertex_ai_pro_concurrent_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 30개 질문 생성
QUESTIONS = [
    # 지하철 관련 (1-10)
    "지하철 1호선의 첫차 시간은 언제인가요?",
    "지하철 2호선은 언제 개통했나요?",
    "서울 지하철 3호선의 총 역 개수는?",
    "지하철 4호선의 막차는 몇 시인가요?",
    "지하철 5호선은 어디부터 어디까지 운행하나요?",
    "지하철 6호선의 환승역은 어디인가요?",
    "지하철 7호선의 운행 간격은?",
    "지하철 8호선은 몇 개 역이 있나요?",
    "지하철 9호선의 급행역은 어디인가요?",
    "지하철 요금은 얼마인가요?",

    # 버스 관련 (11-20)
    "심야버스 운행 시간은 언제인가요?",
    "서울 버스 종류에는 무엇이 있나요?",
    "간선버스는 무슨 색인가요?",
    "지선버스의 역할은 무엇인가요?",
    "광역버스는 어디로 운행하나요?",
    "마을버스는 어떤 지역을 운행하나요?",
    "버스 환승 할인은 얼마나 되나요?",
    "버스 요금은 어떻게 계산되나요?",
    "저상버스는 무엇인가요?",
    "심야버스 번호는 어떻게 구분하나요?",

    # 교통카드 관련 (21-30)
    "티머니 카드는 어디서 구매할 수 있나요?",
    "교통카드 충전은 어디서 하나요?",
    "교통카드 환승 시간은 얼마나 되나요?",
    "1회용 교통카드는 어떻게 사용하나요?",
    "교통카드 분실 시 어떻게 하나요?",
    "모바일 교통카드는 어떻게 사용하나요?",
    "청소년 교통카드 할인율은?",
    "어린이 교통카드 요금은?",
    "교통카드 잔액 조회는 어떻게 하나요?",
    "교통카드 사용 내역은 어디서 확인하나요?",
]


async def test_concurrent_30_calls():
    """30개 질문을 비동기로 동시에 호출하는 테스트 (Gemini 2.5 Pro)"""
    print("\n" + "=" * 80)
    print("  🚀 Vertex AI 동시 호출 스트레스 테스트 (30개 질문)")
    print("  📦 모델: Gemini 2.5 Pro")
    print("=" * 80)

    try:
        # Vertex AI 모델 생성 - Gemini 2.5 Pro
        config = VertexAILLMConfig(
            model_name="gemini-2.5-pro",
            temperature=0.0
        )

        litellm_model = create_vertex_ai_litellm_model(
            model_name=config.model_name,
            project_id=config.project_id,
            location=config.location
        )

        print(f"\n  📊 테스트 정보:")
        print(f"    - 모델: {litellm_model}")
        print(f"    - 질문 개수: {len(QUESTIONS)}개")
        print(f"    - 실행 방식: 비동기 동시 호출")

        # 시작 시간 기록
        start_time = time.time()
        print(f"\n  ⏱️  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 비동기 태스크 생성
        print(f"\n  📤 30개 질문을 비동기로 동시 호출 시작...")
        tasks = [
            litellm.acompletion(
                model=litellm_model,
                messages=[
                    {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다. 간결하게 답변하세요."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100,
                temperature=0.0
            )
            for question in QUESTIONS
        ]

        # 모든 비동기 호출 동시 실행
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 종료 시간 기록
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"  ✅ 모든 호출 완료")
        print(f"  ⏱️  종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  ⏱️  소요 시간: {elapsed_time:.2f}초")

        # 결과 분석
        success_count = 0
        error_count = 0
        empty_count = 0
        total_tokens_input = 0
        total_tokens_output = 0

        print(f"\n  📊 결과 분석:")
        print("  " + "-" * 76)

        for idx, (question, response) in enumerate(zip(QUESTIONS, responses), 1):
            # 짧게 표시
            question_short = question[:40] + "..." if len(question) > 40 else question

            if isinstance(response, Exception):
                print(f"  [{idx:2d}] ❌ 오류: {question_short}")
                print(f"        {str(response)[:60]}...")
                logger.error(f"Call {idx} failed: {response}")
                error_count += 1
            else:
                # 토큰 사용량 집계
                if hasattr(response, 'usage') and response.usage:
                    total_tokens_input += getattr(response.usage, 'prompt_tokens', 0)
                    total_tokens_output += getattr(response.usage, 'completion_tokens', 0)

                answer = response.choices[0].message.content
                if answer:
                    answer_short = answer[:40] + "..." if len(answer) > 40 else answer
                    print(f"  [{idx:2d}] ✅ {question_short}")
                    print(f"        답변: {answer_short}")
                    success_count += 1
                else:
                    print(f"  [{idx:2d}] ⚠️  빈 응답: {question_short}")
                    logger.warning(f"Empty response from call {idx}")
                    empty_count += 1

        print("  " + "-" * 76)

        # 통계 출력
        print(f"\n  📈 통계:")
        print(f"    - 성공: {success_count}/{len(QUESTIONS)}개")
        print(f"    - 오류: {error_count}/{len(QUESTIONS)}개")
        print(f"    - 빈 응답: {empty_count}/{len(QUESTIONS)}개")
        print(f"    - 성공률: {success_count/len(QUESTIONS)*100:.1f}%")

        # 토큰 사용량
        print(f"\n  💰 API 사용량:")
        print(f"    - 입력 토큰: {total_tokens_input:,}개")
        print(f"    - 출력 토큰: {total_tokens_output:,}개")
        print(f"    - 총 토큰: {total_tokens_input + total_tokens_output:,}개")

        # 성능 지표
        print(f"\n  ⚡ 성능 지표:")
        print(f"    - 총 소요 시간: {elapsed_time:.2f}초")
        print(f"    - 평균 응답 시간: {elapsed_time/len(QUESTIONS):.2f}초/질문")
        print(f"    - 처리량: {len(QUESTIONS)/elapsed_time:.2f}개/초")

        # 비용 추정 (Gemini 2.5 Pro는 Flash보다 비쌈)
        input_cost = (total_tokens_input / 1000) * 0.000125  # $0.125/1M tokens
        output_cost = (total_tokens_output / 1000) * 0.0005   # $0.5/1M tokens
        total_cost = input_cost + output_cost

        print(f"\n  💵 예상 비용 (Gemini 2.5 Pro):")
        print(f"    - Input cost: ${input_cost:.6f}")
        print(f"    - Output cost: ${output_cost:.6f}")
        print(f"    - Total cost: ${total_cost:.6f}")
        print(f"    - 참고: Flash 대비 약 8-10배 비쌈")

        return success_count > 0

    except Exception as e:
        print(f"\n  ❌ 테스트 실패: {e}")
        logger.error(f"Concurrent test failed: {e}", exc_info=True)
        return False


async def main():
    """메인 함수"""
    try:
        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)

        # 테스트 실행
        success = await test_concurrent_30_calls()

        print("\n" + "=" * 80)
        if success:
            print("  🎉 Gemini 2.5 Pro 테스트 완료!")
            print("  💡 Flash 모델 결과와 비교해보세요.")
        else:
            print("  ⚠️  테스트 실패. 로그를 확인하세요.")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        logger.error(f"Main error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
