#!/usr/bin/env python3
"""
Vertex AI + RAGAS 사용 예제
간단한 평가 시나리오 데모
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
    create_vertex_ai_litellm_model,
    test_vertex_ai_connection,
)

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from litellm import completion


def example_1_basic_setup():
    """예제 1: 기본 설정"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 설정 및 연결 테스트")
    print("=" * 60)

    # Vertex AI 설정
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0
    )

    print(f"✅ 설정 완료")
    print(f"   모델: {config.model_name}")
    print(f"   프로젝트: {config.project_id}")
    print(f"   위치: {config.location}")

    # 연결 테스트
    print("\n🔍 연결 테스트 중...")
    success = test_vertex_ai_connection(config)

    if success:
        print("✅ Vertex AI 연결 성공!")
    else:
        print("❌ 연결 실패. 환경 설정을 확인하세요.")


def example_2_answer_generation():
    """예제 2: LiteLLM으로 답변 생성"""
    print("\n" + "=" * 60)
    print("예제 2: LiteLLM으로 답변 생성")
    print("=" * 60)

    # Vertex AI 모델 생성
    model = create_vertex_ai_litellm_model(
        model_name="gemini-2.5-flash"
    )

    print(f"📤 모델: {model}")

    # 답변 생성
    question = "서울 지하철 1호선의 역사는 몇 개인가요?"

    print(f"❓ 질문: {question}")

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.1
    )

    answer = response.choices[0].message.content
    print(f"📝 답변: {answer}")


def example_3_ragas_evaluation():
    """예제 3: RAGAS 평가"""
    print("\n" + "=" * 60)
    print("예제 3: RAGAS 평가")
    print("=" * 60)

    # Vertex AI Evaluator 생성
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0
    )

    evaluator_llm = create_vertex_ai_evaluator_llm(config)
    evaluator_embeddings = create_vertex_ai_embeddings(config)

    print("✅ Evaluator LLM 생성 완료")
    print("✅ Evaluator Embeddings 생성 완료")

    # RAGAS 메트릭 설정
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    print(f"✅ 메트릭 설정 완료: {len(metrics)}개")

    # 평가 데이터셋
    test_data = [
        {
            "user_input": "지하철 요금은 얼마인가요?",
            "response": "서울 지하철 기본요금은 1,400원입니다.",
            "retrieved_contexts": [
                "서울시 대중교통 요금체계는 거리비례제를 적용하며 기본구간 요금이 적용된다."
            ],
        },
        {
            "user_input": "심야버스는 언제 운행하나요?",
            "response": "심야버스는 밤 11시 30분부터 새벽 4시 30분까지 운행합니다.",
            "retrieved_contexts": [
                "심야버스는 심야시간대 대중교통 이용을 위해 특별 운행된다."
            ],
        },
    ]

    dataset = EvaluationDataset.from_list(test_data)
    print(f"✅ 데이터셋 생성 완료: {len(test_data)}개 샘플")

    # RAGAS 평가 실행
    print("\n🚀 RAGAS 평가 실행 중...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm
    )

    print("✅ 평가 완료")

    # 결과 출력
    print("\n📊 평가 결과:")
    df = result.to_pandas()

    for metric in ["faithfulness", "answer_relevancy"]:
        if metric in df.columns:
            score = df[metric].mean()
            print(f"  {metric}: {score:.3f}")


def example_4_model_comparison():
    """예제 4: 여러 모델 답변 비교"""
    print("\n" + "=" * 60)
    print("예제 4: 여러 모델 답변 비교")
    print("=" * 60)

    question = "서울 지하철은 몇 개 노선이 있나요?"

    # 모델 설정
    models = [
        {
            "name": "Vertex AI Gemini 2.5",
            "model": create_vertex_ai_litellm_model("gemini-2.5-flash")
        },
        {
            "name": "GPT-4o-mini",
            "model": "gpt-4o-mini"
        },
    ]

    print(f"❓ 질문: {question}\n")

    # 각 모델로 답변 생성
    for model_config in models:
        try:
            print(f"🔍 {model_config['name']} 답변:")

            response = completion(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            print(f"  {answer}\n")

        except Exception as e:
            print(f"  ❌ 오류: {e}\n")


def main():
    """메인 함수"""
    print("\n🎯" * 30)
    print("Vertex AI + RAGAS 사용 예제")
    print("🎯" * 30)

    try:
        # 예제 1: 기본 설정
        example_1_basic_setup()

        # 예제 2: 답변 생성
        example_2_answer_generation()

        # 예제 3: RAGAS 평가
        example_3_ragas_evaluation()

        # 예제 4: 모델 비교
        example_4_model_comparison()

        print("\n" + "=" * 60)
        print("🎉 모든 예제 실행 완료!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
