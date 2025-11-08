#!/usr/bin/env python3
"""
RAGAS LLM 비교 평가 - Vertex AI Judge 버전
- Vertex AI Gemini를 평가 모델(Judge)로 사용
- 여러 LLM 모델들의 답변 비교
- 기존 test_ragas_compare_llms_litellm_v2.py를 Vertex AI용으로 변형
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import litellm
from litellm import completion

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
)

# RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# RAG 파이프라인
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage

# 타임스탬프
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 로그 설정
log_dir = Path("logs/llm_comparison")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            log_dir / f"vertex_ai_judge_{timestamp}.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LiteLLM 설정
litellm.set_verbose = False
litellm.drop_params = True


def print_important(message: str):
    """중요 메시지 출력"""
    print(message)
    logger.info(message)


def create_korean_testset() -> pd.DataFrame:
    """한국어 테스트셋 생성"""
    logger.info("한국어 테스트셋 생성 중...")

    test_data = [
        {
            "user_input": "지하철역에서 반려동물 용품을 살 수 있나요?",
            "reference": "네, 서울교통공사가 일부 지하철역에 반려동물 용품 자판기를 설치했습니다.",
            "reference_contexts": [
                "서울교통공사는 반려동물 키우는 지하철 이용고객을 위해 반려동물용품 자판기를 설치했다."
            ],
        },
        {
            "user_input": "서울시 대중교통 요금은 얼마인가요?",
            "reference": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다.",
            "reference_contexts": [
                "서울시 대중교통 요금체계는 거리비례제를 적용하며 기본구간 요금이 적용된다."
            ],
        },
        {
            "user_input": "택시 바가지요금 신고는 어떻게 하나요?",
            "reference": "택시 바가지요금은 서울시 120 다산콜센터나 택시 신고 앱을 통해 신고할 수 있습니다.",
            "reference_contexts": [
                "택시 불편사항은 120 다산콜센터를 통해 신고 가능하며, 차량정보와 증빙자료가 필요하다."
            ],
        },
        {
            "user_input": "심야버스 운행시간은 언제인가요?",
            "reference": "서울 심야버스(올빼미버스)는 밤 23:30부터 새벽 04:30까지 운행합니다.",
            "reference_contexts": [
                "심야버스는 심야시간대 대중교통 이용을 위해 특별 운행된다."
            ],
        },
        {
            "user_input": "교통카드 환불은 어디서 받나요?",
            "reference": "교통카드 환불은 지하철역 고객안내센터나 편의점에서 가능합니다.",
            "reference_contexts": [
                "교통카드 환불 정책에 따라 지정된 장소에서 환불이 가능하다."
            ],
        },
    ]

    df = pd.DataFrame(test_data)
    logger.info(f"테스트셋 생성 완료: {len(df)}개 샘플")
    return df


def test_model_connection(model_config: Dict[str, Any]) -> bool:
    """모델 연결 테스트"""
    logger.info(f"모델 연결 테스트: {model_config['name']}")

    try:
        response = completion(
            model=model_config["model"],
            messages=[{"role": "user", "content": "테스트"}],
            api_base=model_config.get("api_base"),
            max_tokens=10,
            temperature=0,
        )

        if response and response.choices:
            logger.info(f"연결 성공: {model_config['name']}")
            return True
        else:
            logger.error(f"응답 없음: {model_config['name']}")
            return False

    except Exception as e:
        logger.error(f"연결 실패 {model_config['name']}: {str(e)[:100]}")
        return False


def generate_answer_with_litellm(
    question: str, context: str, model_config: Dict[str, Any]
) -> str:
    """LiteLLM을 사용한 답변 생성"""

    prompt = f"""다음 컨텍스트를 사용하여 질문에 답하세요:

컨텍스트:
{context}

질문: {question}

답변:"""

    try:
        response = completion(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            api_base=model_config.get("api_base"),
            max_tokens=200,
            temperature=0.1,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"답변 생성 실패 ({model_config['name']}): {e}")
        return f"오류 발생: {str(e)[:50]}"


def evaluate_llm_with_rag(
    testset_df: pd.DataFrame, model_config: Dict[str, Any]
) -> List[Dict]:
    """특정 LLM으로 RAG 평가 실행"""
    logger.info(f"{model_config['name']} 평가 시작...")

    try:
        # Vector Store 로드
        logger.debug("벡터 스토어 로드 중...")
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store/seoul_traffic"))
        retriever = vector_store.as_retriever(k=4)

        # 각 질문에 대해 평가
        results = []
        for idx, row in testset_df.iterrows():
            question = row["user_input"]
            reference = row.get("reference", "")

            logger.debug(f"[{model_config['name']} - 평가 {idx + 1}/{len(testset_df)}]")

            # 컨텍스트 검색
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # LiteLLM으로 답변 생성
            answer = generate_answer_with_litellm(question, context, model_config)

            results.append(
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": [doc.page_content for doc in docs],
                    "reference": reference,
                    "llm": model_config["name"],
                }
            )

        return results

    except Exception as e:
        logger.error(f"{model_config['name']} 평가 실패: {e}")
        raise


def compare_llm_results(results_dict: Dict[str, List[Dict]], vertex_config: VertexAILLMConfig):
    """여러 LLM 결과 비교 및 RAGAS 평가 (Vertex AI Judge)"""
    logger.info("LLM 비교 평가 시작 (Judge: Vertex AI Gemini)...")

    # Vertex AI 평가 LLM 설정
    print_important(f"\n🔧 Judge 모델 설정: Vertex AI {vertex_config.model_name}")
    eval_llm = create_vertex_ai_evaluator_llm(vertex_config)
    eval_embeddings = create_vertex_ai_embeddings(vertex_config)

    # 메트릭 설정
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        AnswerCorrectness(llm=eval_llm),
    ]

    # 각 모델별 평가
    evaluation_results = {}

    for model_name, results in results_dict.items():
        logger.info(f"[{model_name} 평가]")
        dataset = EvaluationDataset.from_list(results)

        eval_result = evaluate(dataset=dataset, metrics=metrics, llm=eval_llm)

        evaluation_results[model_name] = eval_result

    return evaluation_results


def print_comparison_results(evaluation_results: Dict):
    """평가 결과 비교 출력"""
    print_important("\n" + "=" * 60)
    print_important("📊 LLM 성능 비교 결과 (Judge: Vertex AI Gemini)")
    print_important("=" * 60)

    # 결과를 DataFrame으로 변환
    comparison_data = []
    model_names = list(evaluation_results.keys())

    # 메트릭별로 비교
    metrics_to_compare = ["faithfulness", "answer_relevancy", "answer_correctness"]

    for metric in metrics_to_compare:
        row_data = {"Metric": metric}

        for model_name in model_names:
            eval_result = evaluation_results[model_name]

            if hasattr(eval_result, "to_pandas"):
                df = eval_result.to_pandas()
                if metric in df.columns:
                    score = df[metric].mean()
                    row_data[model_name] = f"{score:.3f}"

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # 최고 점수 모델 찾기
    print_important("\n🏆 메트릭별 최고 성능:")
    for row in comparison_data:
        metric = row["Metric"]
        scores = {k: float(v) for k, v in row.items() if k != "Metric"}
        if scores:
            best_model = max(scores, key=scores.get)
            best_score = scores[best_model]
            print_important(f"  {metric}: {best_model} ({best_score:.3f})")


def save_comparison_results(results_dict: Dict, evaluation_results: Dict, vertex_config: VertexAILLMConfig):
    """비교 결과 저장"""
    output_dir = Path("data/evaluation/llm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 평가 결과 직렬화
    serializable_eval_results = {}
    for model, eval_result in evaluation_results.items():
        if hasattr(eval_result, "to_pandas"):
            df = eval_result.to_pandas()
            serializable_eval_results[model] = {
                "metrics": df.to_dict("records"),
                "summary": {
                    "faithfulness": float(df["faithfulness"].mean())
                    if "faithfulness" in df.columns
                    else None,
                    "answer_relevancy": float(df["answer_relevancy"].mean())
                    if "answer_relevancy" in df.columns
                    else None,
                    "answer_correctness": float(df["answer_correctness"].mean())
                    if "answer_correctness" in df.columns
                    else None,
                },
            }

    # 전체 결과 저장
    all_results = {
        "timestamp": timestamp,
        "judge_model": {
            "provider": "Vertex AI",
            "model": vertex_config.model_name,
            "embedding": vertex_config.embedding_model,
            "project": vertex_config.project_id,
            "location": vertex_config.location,
        },
        "raw_results": results_dict,
        "evaluation_metrics": serializable_eval_results,
    }

    result_file = output_dir / f"vertex_ai_judge_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"결과 저장: {result_file}")
    print_important(f"\n💾 상세 결과 저장 완료: {result_file}")


def main():
    """메인 실행"""
    print_important("\n" + "=" * 60)
    print_important("🚀 LLM 비교 평가 (Judge: Vertex AI Gemini)".center(60))
    print_important("=" * 60)

    try:
        # 1. Vertex AI Judge 설정
        logger.info("[1단계] Vertex AI Judge 설정")
        print_important("\n[1단계] Vertex AI Judge 설정")

        try:
            vertex_config = VertexAILLMConfig(
                model_name="gemini-2.5-flash",
                temperature=0.0
            )
        except ValueError as e:
            print_important(f"\n❌ 설정 오류: {e}")
            print_important("\n📝 환경 변수 설정 방법:")
            print_important("  1. .env 파일에 다음 추가:")
            print_important("     VERTEXAI_PROJECT=your-project-id")
            print_important("     VERTEXAI_LOCATION=us-central1")
            print_important("     VERTEXAI_API_KEY=/path/to/service-account.json  # 선택사항")
            print_important("\n  2. 또는 터미널에서 직접 설정:")
            print_important("     export VERTEXAI_PROJECT=your-project-id")
            print_important("\n  3. 또는 gcloud 인증:")
            print_important("     gcloud auth application-default login")
            return

        print_important(f"  Judge 모델: {vertex_config.model_name}")
        print_important(f"  Embedding: {vertex_config.embedding_model}")
        print_important(f"  Project: {vertex_config.project_id}")
        print_important(f"  Location: {vertex_config.location}")

        # 2. 피평가 모델 설정
        logger.info("[2단계] 피평가 모델 설정")
        print_important("\n[2단계] 피평가 모델 설정")

        models = [
            {
                "name": "GPT-4o-mini",
                "model": "gpt-4o-mini",
                "api_base": None,
                "description": "OpenAI 경량 모델",
            },
            {
                "name": "EXAONE-3.5-7.8B",
                "model": "ollama/exaone3.5:7.8b",
                "api_base": "http://100.95.220.92:11434",
                "description": "한국어 주력",
            },
            {
                "name": "Gemma3-12B",
                "model": "ollama/gemma3:12b",
                "api_base": "http://100.95.220.92:11434",
                "description": "범용 fallback",
            },
        ]

        # 사용 가능한 모델 필터링
        available_models = []
        for model_config in models:
            print(f"  🔍 {model_config['name']} 연결 테스트 중...")
            if test_model_connection(model_config):
                available_models.append(model_config)
                print(f"  ✅ {model_config['name']} - {model_config['description']}")
            else:
                print(f"  ❌ {model_config['name']} 연결 실패")

        if len(available_models) < 1:
            print_important("❌ 최소 1개 이상의 모델이 필요합니다.")
            return

        print_important(f"\n✅ 사용 가능한 모델: {[m['name'] for m in available_models]}")

        # 3. 테스트셋 생성
        logger.info("[3단계] 테스트셋 생성")
        print_important("\n[3단계] 테스트셋 생성")
        testset_df = create_korean_testset()
        print(f"  ✅ 테스트셋: {len(testset_df)}개 샘플")

        # 4. 각 모델로 평가 실행
        logger.info("[4단계] RAG 평가 실행")
        print_important("\n[4단계] RAG 평가 실행")
        results_dict = {}

        for idx, model_config in enumerate(available_models, 1):
            print(f"  [{idx}/{len(available_models)}] {model_config['name']} 평가 중...")
            results = evaluate_llm_with_rag(testset_df, model_config)
            results_dict[model_config["name"]] = results
            print(f"  ✅ {model_config['name']} 완료")

        # 5. RAGAS 메트릭 비교 (Vertex AI Judge)
        logger.info("[5단계] RAGAS 메트릭 평가")
        print_important("\n[5단계] RAGAS 메트릭 평가 (Judge: Vertex AI)")
        evaluation_results = compare_llm_results(results_dict, vertex_config)

        # 6. 결과 출력
        print_comparison_results(evaluation_results)

        # 7. 결과 저장
        save_comparison_results(results_dict, evaluation_results, vertex_config)

        print_important("\n🎉 LLM 비교 평가 완료!")
        print_important(f"📁 상세 로그: logs/llm_comparison/vertex_ai_judge_{timestamp}.log")

    except KeyboardInterrupt:
        logger.warning("사용자 중단")
        print_important("\n⚠️ 사용자 중단")
    except Exception as e:
        logger.error(f"오류: {e}", exc_info=True)
        print_important(f"\n❌ 오류: {e}")


if __name__ == "__main__":
    main()
