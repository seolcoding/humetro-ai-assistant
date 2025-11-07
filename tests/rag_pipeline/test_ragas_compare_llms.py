#!/usr/bin/env python3
"""
RAGAS LLM 비교 평가 - Ollama vs GPT-4o
- 동일한 컨텍스트로 두 LLM의 생성 결과를 비교
- RAGAS 메트릭으로 평가
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

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

# LangChain imports
from langchain_community.llms import Ollama
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


def check_ollama_endpoint(base_url: str, model_name: str):
    """원격 Ollama 엔드포인트 확인"""
    logger.info(f"🔍 Ollama 엔드포인트 확인: {base_url}")
    logger.info(f"   모델: {model_name}")

    import requests
    try:
        # 엔드포인트 연결 테스트
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "테스트",
                "stream": False,
                "options": {"num_predict": 1}
            },
            timeout=10
        )

        if response.status_code == 200:
            logger.info(f"✅ 엔드포인트 연결 성공: {model_name}")
            return True
        else:
            logger.error(f"❌ HTTP {response.status_code}: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 엔드포인트 연결 실패: {e}")
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


def evaluate_with_llm(testset_df: pd.DataFrame, llm_name: str, llm_model):
    """특정 LLM으로 RAG 평가 실행"""
    logger.info(f"\n🔍 {llm_name} 평가 시작...")

    try:
        # Vector Store 로드
        logger.info("  벡터 스토어 로드 중...")
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store/seoul_traffic"))

        # Retriever 설정
        retriever = vector_store.as_retriever(k=4)

        # RAG Chain 설정 (특정 LLM 사용)
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        prompt_template = """다음 컨텍스트를 사용하여 질문에 답하세요:

컨텍스트:
{context}

질문: {question}

답변:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # RAG 체인 생성
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm_model
            | StrOutputParser()
        )

        # 각 질문에 대해 평가
        results = []
        for idx, row in testset_df.iterrows():
            question = row['user_input']
            reference = row.get('reference', "")

            logger.info(f"\n[{llm_name} - 평가 {idx+1}/{len(testset_df)}]")
            logger.info(f"  질문: {question[:50]}...")

            # 컨텍스트 검색
            docs = retriever.get_relevant_documents(question)
            context_str = "\n\n".join([doc.page_content for doc in docs])

            # LLM으로 답변 생성
            answer = rag_chain.invoke(question)

            logger.info(f"  생성답변: {answer[:50]}...")
            logger.info(f"  검색문서: {len(docs)}개")

            results.append({
                'user_input': question,
                'response': answer,
                'retrieved_contexts': [doc.page_content for doc in docs],
                'reference': reference,
                'llm': llm_name
            })

        return results

    except Exception as e:
        logger.error(f"❌ {llm_name} 평가 실패: {e}")
        raise


def _format_docs(docs):
    """문서 포맷팅 헬퍼 함수"""
    return "\n\n".join([doc.page_content for doc in docs])


def compare_llm_results(ollama_results: List[Dict], gpt_results: List[Dict]):
    """두 LLM 결과 비교 및 RAGAS 평가"""
    logger.info("\n📊 LLM 비교 평가 시작...")

    # 평가 LLM 설정 (GPT-4o-mini를 judge로 사용)
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_llm_wrapper = LangchainLLMWrapper(eval_llm)

    # 메트릭 설정 (컨텍스트 고정이므로 Faithfulness, AnswerRelevancy만)
    metrics = [
        Faithfulness(llm=eval_llm_wrapper),
        AnswerRelevancy(llm=eval_llm_wrapper),
        AnswerCorrectness(llm=eval_llm_wrapper),
    ]

    # Ollama 평가
    logger.info("\n[Ollama 평가]")
    ollama_dataset = EvaluationDataset.from_list(ollama_results)
    ollama_eval = evaluate(
        dataset=ollama_dataset,
        metrics=metrics,
        llm=eval_llm_wrapper
    )

    # GPT-4o 평가
    logger.info("\n[GPT-4o 평가]")
    gpt_dataset = EvaluationDataset.from_list(gpt_results)
    gpt_eval = evaluate(
        dataset=gpt_dataset,
        metrics=metrics,
        llm=eval_llm_wrapper
    )

    return ollama_eval, gpt_eval


def print_comparison_results(ollama_eval, gpt_eval):
    """평가 결과 비교 출력"""
    logger.info("\n" + "="*60)
    logger.info("📊 LLM 성능 비교 결과")
    logger.info("="*60)

    # 결과를 DataFrame으로 변환
    comparison_data = []

    if hasattr(ollama_eval, 'to_pandas'):
        ollama_df = ollama_eval.to_pandas()
        gpt_df = gpt_eval.to_pandas()

        for metric in ['faithfulness', 'answer_relevancy', 'answer_correctness']:
            if metric in ollama_df.columns:
                ollama_score = ollama_df[metric].mean()
                gpt_score = gpt_df[metric].mean()
                diff = gpt_score - ollama_score

                comparison_data.append({
                    'Metric': metric,
                    'Ollama': f"{ollama_score:.3f}",
                    'GPT-4o': f"{gpt_score:.3f}",
                    'Difference': f"{diff:+.3f}"
                })
    else:
        for key in ollama_eval.keys():
            if isinstance(ollama_eval[key], (int, float)):
                ollama_score = ollama_eval[key]
                gpt_score = gpt_eval[key]
                diff = gpt_score - ollama_score

                comparison_data.append({
                    'Metric': key,
                    'Ollama': f"{ollama_score:.3f}",
                    'GPT-4o': f"{gpt_score:.3f}",
                    'Difference': f"{diff:+.3f}"
                })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # 승자 결정
    logger.info("\n🏆 평가 요약:")
    ollama_wins = 0
    gpt_wins = 0

    for row in comparison_data:
        diff = float(row['Difference'])
        if diff > 0:
            gpt_wins += 1
            logger.info(f"  {row['Metric']}: GPT-4o 승 ({row['GPT-4o']} vs {row['Ollama']})")
        elif diff < 0:
            ollama_wins += 1
            logger.info(f"  {row['Metric']}: Ollama 승 ({row['Ollama']} vs {row['GPT-4o']})")
        else:
            logger.info(f"  {row['Metric']}: 동점 ({row['Ollama']})")

    logger.info(f"\n최종 결과: GPT-4o {gpt_wins}승 / Ollama {ollama_wins}승")


def save_comparison_results(ollama_results, gpt_results, ollama_eval, gpt_eval):
    """비교 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/evaluation/llm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 결과 저장
    all_results = {
        "ollama_results": ollama_results,
        "gpt_results": gpt_results,
        "ollama_metrics": dict(ollama_eval) if hasattr(ollama_eval, '__iter__') else ollama_eval,
        "gpt_metrics": dict(gpt_eval) if hasattr(gpt_eval, '__iter__') else gpt_eval,
    }

    result_file = output_dir / f"comparison_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 결과 저장: {result_file}")


def main():
    """메인 실행"""
    print("\n" + "="*60)
    print("🚀 LLM 비교 평가 (Ollama vs GPT-4o)".center(60))
    print("="*60)

    try:
        # 1. Ollama 엔드포인트 설정
        OLLAMA_BASE_URL = "http://100.95.220.92:11434"

        # 사용 가능한 모델들 (우선순위 순)
        ollama_models = [
            {"name": "EXAONE-3.5-7.8B", "model": "exaone3.5:7.8b"},
            {"name": "Solar-Pro", "model": "solar-pro:latest"},
            {"name": "Phi4-14B", "model": "phi4:latest"},
            {"name": "Gemma3-12B", "model": "gemma3:12b"},
        ]

        # 첫 번째 사용 가능한 모델 찾기
        selected_model = None
        for model_info in ollama_models:
            if check_ollama_endpoint(OLLAMA_BASE_URL, model_info["model"]):
                selected_model = model_info
                break

        if not selected_model:
            logger.error("❌ 사용 가능한 Ollama 모델이 없습니다.")
            return

        # 2. 테스트셋 생성
        logger.info("\n[1단계] 테스트셋 생성")
        testset_df = create_korean_testset()

        # 3. LLM 모델 설정
        logger.info("\n[2단계] LLM 모델 설정")

        # Ollama 모델 (원격 엔드포인트)
        ollama_llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=selected_model["model"],
            temperature=0.1
        )
        logger.info(f"  ✓ Ollama: {selected_model['name']} ({selected_model['model']})")

        # GPT-4o 모델
        gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        logger.info(f"  ✓ GPT-4o")

        # 4. 각 LLM으로 평가 실행
        logger.info("\n[3단계] RAG 평가 실행")

        # 간단한 RAG 구현 (RetrievalStage 대신 직접 구현)
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store/seoul_traffic"))
        retriever = vector_store.as_retriever(k=4)

        # Ollama 평가
        ollama_results = []
        for idx, row in testset_df.iterrows():
            question = row['user_input']
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"
            answer = ollama_llm.invoke(prompt)

            ollama_results.append({
                'user_input': question,
                'response': answer,
                'retrieved_contexts': [doc.page_content for doc in docs],
                'reference': row['reference']
            })
            logger.info(f"  Ollama 평가 {idx+1}/{len(testset_df)} 완료")

        # GPT-4o 평가
        gpt_results = []
        for idx, row in testset_df.iterrows():
            question = row['user_input']
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"
            answer = gpt_llm.invoke(prompt).content

            gpt_results.append({
                'user_input': question,
                'response': answer,
                'retrieved_contexts': [doc.page_content for doc in docs],
                'reference': row['reference']
            })
            logger.info(f"  GPT-4o 평가 {idx+1}/{len(testset_df)} 완료")

        # 5. RAGAS 메트릭 비교
        logger.info("\n[4단계] RAGAS 메트릭 평가")
        ollama_eval, gpt_eval = compare_llm_results(ollama_results, gpt_results)

        # 6. 결과 출력
        print_comparison_results(ollama_eval, gpt_eval)

        # 7. 결과 저장
        save_comparison_results(ollama_results, gpt_results, ollama_eval, gpt_eval)

        logger.info("\n🎉 LLM 비교 평가 완료!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자 중단")
    except Exception as e:
        logger.error(f"\n❌ 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()