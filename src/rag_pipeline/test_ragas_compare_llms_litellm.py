#!/usr/bin/env python3
"""
RAGAS LLM 비교 평가 - LiteLLM 통합 인터페이스 사용
- LiteLLM으로 Ollama와 OpenAI를 동일한 인터페이스로 호출
- RAGAS 메트릭으로 평가
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

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# LiteLLM 로깅 설정
litellm.set_verbose = False  # 디버그 모드 끄기

# LangChain imports (RAGAS용)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# 우리 RAG 파이프라인
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
from src.rag_pipeline.stages.stage_06_retrieval import RetrievalStage


def test_model_connection(model_config: Dict[str, Any]) -> bool:
    """모델 연결 테스트"""
    logger.info(f"🔍 모델 연결 테스트: {model_config['name']}")

    try:
        # LiteLLM completion API로 테스트
        response = completion(
            model=model_config['model'],
            messages=[{"role": "user", "content": "테스트"}],
            api_base=model_config.get('api_base'),
            max_tokens=10,
            temperature=0
        )

        if response and response.choices:
            logger.info(f"✅ 연결 성공: {model_config['name']}")
            return True
        else:
            logger.error(f"❌ 응답 없음: {model_config['name']}")
            return False

    except Exception as e:
        logger.error(f"❌ 연결 실패 {model_config['name']}: {str(e)[:100]}")
        return False


def create_korean_testset() -> pd.DataFrame:
    """한국어 테스트셋 생성"""
    logger.info("📝 한국어 테스트셋 생성 중...")

    test_data = [
        {
            "user_input": "지하철역에서 반려동물 용품을 살 수 있나요?",
            "reference": "네, 서울교통공사가 일부 지하철역에 반려동물 용품 자판기를 설치했습니다.",
            "reference_contexts": ["서울교통공사는 반려동물 키우는 지하철 이용고객을 위해 반려동물용품 자판기를 설치했다."]
        },
        {
            "user_input": "서울시 대중교통 요금은 얼마인가요?",
            "reference": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다.",
            "reference_contexts": ["서울시 대중교통 요금체계는 거리비례제를 적용하며 기본구간 요금이 적용된다."]
        },
        {
            "user_input": "택시 바가지요금 신고는 어떻게 하나요?",
            "reference": "택시 바가지요금은 서울시 120 다산콜센터나 택시 신고 앱을 통해 신고할 수 있습니다.",
            "reference_contexts": ["택시 불편사항은 120 다산콜센터를 통해 신고 가능하며, 차량정보와 증빙자료가 필요하다."]
        },
        {
            "user_input": "심야버스 운행시간은 언제인가요?",
            "reference": "서울 심야버스(올빼미버스)는 밤 23:30부터 새벽 04:30까지 운행합니다.",
            "reference_contexts": ["심야버스는 심야시간대 대중교통 이용을 위해 특별 운행된다."]
        },
        {
            "user_input": "교통카드 환불은 어디서 받나요?",
            "reference": "교통카드 환불은 지하철역 고객안내센터나 편의점에서 가능합니다.",
            "reference_contexts": ["교통카드 환불 정책에 따라 지정된 장소에서 환불이 가능하다."]
        }
    ]

    df = pd.DataFrame(test_data)
    logger.info(f"✅ 테스트셋 생성 완료: {len(df)}개 샘플")
    return df


def generate_answer_with_litellm(
    question: str,
    context: str,
    model_config: Dict[str, Any]
) -> str:
    """LiteLLM을 사용한 통합 답변 생성"""

    prompt = f"""다음 컨텍스트를 사용하여 질문에 답하세요:

컨텍스트:
{context}

질문: {question}

답변:"""

    try:
        response = completion(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            api_base=model_config.get('api_base'),
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"답변 생성 실패 ({model_config['name']}): {e}")
        return f"오류 발생: {str(e)[:50]}"


def evaluate_llm_with_rag(testset_df: pd.DataFrame, model_config: Dict[str, Any]) -> List[Dict]:
    """특정 LLM으로 RAG 평가 실행"""
    logger.info(f"\n🔍 {model_config['name']} 평가 시작...")

    try:
        # Vector Store 로드
        logger.info("  벡터 스토어 로드 중...")
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store/seoul_traffic"))
        retriever = vector_store.as_retriever(k=4)

        # 각 질문에 대해 평가
        results = []
        for idx, row in testset_df.iterrows():
            question = row['user_input']
            reference = row.get('reference', "")

            logger.info(f"\n[{model_config['name']} - 평가 {idx+1}/{len(testset_df)}]")
            logger.info(f"  질문: {question[:50]}...")

            # 컨텍스트 검색
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # LiteLLM으로 답변 생성
            answer = generate_answer_with_litellm(question, context, model_config)

            logger.info(f"  생성답변: {answer[:50]}...")
            logger.info(f"  검색문서: {len(docs)}개")

            results.append({
                'user_input': question,
                'response': answer,
                'retrieved_contexts': [doc.page_content for doc in docs],
                'reference': reference,
                'llm': model_config['name']
            })

        return results

    except Exception as e:
        logger.error(f"❌ {model_config['name']} 평가 실패: {e}")
        raise


def compare_llm_results(results_dict: Dict[str, List[Dict]]):
    """여러 LLM 결과 비교 및 RAGAS 평가"""
    logger.info("\n📊 LLM 비교 평가 시작...")

    # 평가 LLM 설정 (GPT-4o-mini를 judge로 사용)
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_llm_wrapper = LangchainLLMWrapper(eval_llm)

    # 메트릭 설정
    metrics = [
        Faithfulness(llm=eval_llm_wrapper),
        AnswerRelevancy(llm=eval_llm_wrapper),
        AnswerCorrectness(llm=eval_llm_wrapper),
    ]

    # 각 모델별 평가
    evaluation_results = {}

    for model_name, results in results_dict.items():
        logger.info(f"\n[{model_name} 평가]")
        dataset = EvaluationDataset.from_list(results)

        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm_wrapper
        )

        evaluation_results[model_name] = eval_result

    return evaluation_results


def print_comparison_results(evaluation_results: Dict):
    """평가 결과 비교 출력"""
    logger.info("\n" + "="*60)
    logger.info("📊 LLM 성능 비교 결과")
    logger.info("="*60)

    # 결과를 DataFrame으로 변환
    comparison_data = []
    model_names = list(evaluation_results.keys())

    # 메트릭별로 비교
    metrics_to_compare = ['faithfulness', 'answer_relevancy', 'answer_correctness']

    for metric in metrics_to_compare:
        row_data = {'Metric': metric}

        for model_name in model_names:
            eval_result = evaluation_results[model_name]

            if hasattr(eval_result, 'to_pandas'):
                df = eval_result.to_pandas()
                if metric in df.columns:
                    score = df[metric].mean()
                    row_data[model_name] = f"{score:.3f}"
            else:
                if metric in eval_result:
                    score = eval_result[metric]
                    row_data[model_name] = f"{score:.3f}"

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # 최고 점수 모델 찾기
    logger.info("\n🏆 메트릭별 최고 성능:")
    for row in comparison_data:
        metric = row['Metric']
        scores = {k: float(v) for k, v in row.items() if k != 'Metric' and v != 'N/A'}
        if scores:
            best_model = max(scores, key=scores.get)
            best_score = scores[best_model]
            logger.info(f"  {metric}: {best_model} ({best_score:.3f})")


def save_comparison_results(results_dict: Dict, evaluation_results: Dict):
    """비교 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/evaluation/llm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 결과 저장
    all_results = {
        "raw_results": results_dict,
        "evaluation_metrics": {
            model: dict(eval_result) if hasattr(eval_result, '__iter__') else eval_result
            for model, eval_result in evaluation_results.items()
        }
    }

    result_file = output_dir / f"litellm_comparison_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 결과 저장: {result_file}")


def main():
    """메인 실행"""
    print("\n" + "="*60)
    print("🚀 LiteLLM 통합 LLM 비교 평가".center(60))
    print("="*60)

    try:
        # 1. 모델 설정
        logger.info("\n[1단계] 모델 설정")

        # LiteLLM 모델 설정 (공식 문서 형식)
        models = [
            {
                "name": "GPT-4o",
                "model": "gpt-4o",
                "api_base": None  # OpenAI는 기본값 사용
            },
            {
                "name": "GPT-4o-mini",
                "model": "gpt-4o-mini",
                "api_base": None
            },
            {
                "name": "EXAONE-3.5",
                "model": "ollama/exaone3.5:7.8b",
                "api_base": "http://100.95.220.92:11434"
            },
            {
                "name": "Solar-Pro",
                "model": "ollama/solar-pro:latest",
                "api_base": "http://100.95.220.92:11434"
            },
            {
                "name": "Phi4-14B",
                "model": "ollama/phi4:latest",
                "api_base": "http://100.95.220.92:11434"
            }
        ]

        # 사용 가능한 모델 필터링
        available_models = []
        for model_config in models:
            if test_model_connection(model_config):
                available_models.append(model_config)
                if len(available_models) >= 3:  # 최대 3개 모델만 테스트
                    break

        if len(available_models) < 2:
            logger.error("❌ 비교를 위해 최소 2개 이상의 모델이 필요합니다.")
            return

        logger.info(f"\n✅ 사용 가능한 모델: {[m['name'] for m in available_models]}")

        # 2. 테스트셋 생성
        logger.info("\n[2단계] 테스트셋 생성")
        testset_df = create_korean_testset()

        # 3. 각 모델로 평가 실행
        logger.info("\n[3단계] RAG 평가 실행")
        results_dict = {}

        for model_config in available_models:
            results = evaluate_llm_with_rag(testset_df, model_config)
            results_dict[model_config['name']] = results

        # 4. RAGAS 메트릭 비교
        logger.info("\n[4단계] RAGAS 메트릭 평가")
        evaluation_results = compare_llm_results(results_dict)

        # 5. 결과 출력
        print_comparison_results(evaluation_results)

        # 6. 결과 저장
        save_comparison_results(results_dict, evaluation_results)

        logger.info("\n🎉 LLM 비교 평가 완료!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자 중단")
    except Exception as e:
        logger.error(f"\n❌ 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()