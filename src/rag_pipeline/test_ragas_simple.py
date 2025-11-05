#!/usr/bin/env python3
"""
RAGAS Simple Test - 올바른 포맷으로 구현
- OpenAI 클라이언트 직접 사용
- RAGAS 규격에 맞는 데이터 포맷
"""

import os
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

# OpenAI 및 LangChain imports
import openai
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# RAGAS imports - 공식 문서 방식
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution

# 우리 RAG 파이프라인
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
from src.rag_pipeline.stages.stage_06_retrieval import RetrievalStage


def load_simple_documents(doc_path: str, limit: int = 5) -> List[Document]:
    """간단한 문서 로드"""
    logger.info(f"📂 문서 로드 중: {doc_path}")

    documents = []
    doc_dir = Path(doc_path)

    # 마크다운 파일 읽기
    md_files = list(doc_dir.glob("*.md"))[:limit]

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            # 내용이 너무 길면 잘라내기 (토큰 절약)
            if len(content) > 2000:
                content = content[:2000]

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file.name),
                    "file_path": str(md_file)
                }
            )
            documents.append(doc)
            logger.debug(f"  ✓ {md_file.name}")
        except Exception as e:
            logger.warning(f"  ✗ {md_file.name}: {e}")

    logger.info(f"✅ {len(documents)}개 문서 로드 완료")
    return documents


def create_testset_with_ragas(documents: List[Document], size: int = 5):
    """RAGAS로 테스트셋 생성 - 공식 문서 방식"""
    logger.info("🔧 RAGAS TestsetGenerator 설정 중...")

    # LLM 설정 - LangchainLLMWrapper 사용
    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm_wrapper = LangchainLLMWrapper(generator_llm)

    # Embeddings - LangChain OpenAI Embeddings를 RAGAS wrapper로
    langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings_wrapper = LangchainEmbeddingsWrapper(langchain_embeddings)

    # TestsetGenerator 생성
    generator = TestsetGenerator(
        llm=llm_wrapper,
        embedding_model=embeddings_wrapper
    )

    logger.info("📝 테스트셋 생성 중... (1-2분 소요)")

    try:
        # Query distribution 설정 - generator_llm를 직접 전달
        query_dist = default_query_distribution(generator_llm)

        # 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=size,
            query_distribution=query_dist
        )

        # DataFrame 변환
        df = testset.to_pandas()

        logger.info(f"✅ 테스트셋 생성 완료: {len(df)}개 샘플")

        # 샘플 출력
        for idx, row in df.iterrows():
            logger.info(f"\n[샘플 {idx+1}]")
            # 컬럼 이름 확인하고 출력
            if 'user_input' in row:
                logger.info(f"  Q: {row['user_input'][:80]}...")
            elif 'question' in row:
                logger.info(f"  Q: {row['question'][:80]}...")

            if 'reference' in row:
                logger.info(f"  A: {row['reference'][:80]}...")
            elif 'ground_truth' in row:
                logger.info(f"  A: {row['ground_truth'][:80]}...")

        return df

    except Exception as e:
        logger.error(f"❌ 테스트셋 생성 실패: {e}")
        logger.info("💡 수동 테스트셋 생성으로 전환...")
        return create_manual_testset()


def create_manual_testset() -> pd.DataFrame:
    """수동으로 한국어 테스트셋 생성 (폴백 옵션)"""
    logger.info("📝 수동 테스트셋 생성 중...")

    test_data = [
        {
            "user_input": "지하철역에서 반려동물 용품을 살 수 있나요?",
            "reference": "네, 서울교통공사가 일부 지하철역에 반려동물 용품 자판기를 설치했습니다. 퇴근길에 편리하게 구매할 수 있습니다.",
            "reference_contexts": ["서울교통공사는 반려동물 키우는 지하철 이용고객을 위해 반려동물용품 자판기를 설치했다."]
        },
        {
            "user_input": "서울시 대중교통 요금은 얼마인가요?",
            "reference": "서울시 대중교통 기본요금은 지하철 1,400원, 버스 1,500원입니다. 교통카드 사용 시 환승할인이 적용됩니다.",
            "reference_contexts": ["서울시 대중교통 요금체계는 거리비례제를 적용하며 기본구간 요금이 적용된다."]
        },
        {
            "user_input": "택시 바가지요금 신고는 어떻게 하나요?",
            "reference": "택시 바가지요금은 서울시 120 다산콜센터나 택시 신고 앱을 통해 신고할 수 있습니다. 차량번호와 영수증을 준비하세요.",
            "reference_contexts": ["택시 불편사항은 120 다산콜센터를 통해 신고 가능하며, 차량정보와 증빙자료가 필요하다."]
        },
        {
            "user_input": "심야버스 운행시간은 언제인가요?",
            "reference": "서울 심야버스(올빼미버스)는 밤 23:30부터 새벽 04:30까지 운행합니다.",
            "reference_contexts": ["심야버스는 심야시간대 대중교통 이용을 위해 특별 운행된다."]
        },
        {
            "user_input": "교통카드 환불은 어디서 받나요?",
            "reference": "교통카드 환불은 지하철역 고객안내센터나 편의점에서 가능합니다. 카드 구매 영수증이 있으면 수수료가 면제됩니다.",
            "reference_contexts": ["교통카드 환불 정책에 따라 지정된 장소에서 환불이 가능하다."]
        }
    ]

    df = pd.DataFrame(test_data)
    logger.info(f"✅ 수동 테스트셋 생성 완료: {len(df)}개 샘플")
    return df


def evaluate_rag_with_testset(testset_df: pd.DataFrame):
    """테스트셋으로 RAG 평가"""
    logger.info("\n🔍 RAG 시스템 평가 시작...")

    try:
        # Vector Store 로드
        logger.info("  벡터 스토어 로드 중...")
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store/seoul_traffic"))

        # RAG Chain 설정
        retriever = vector_store.as_retriever(k=4)
        rag_stage = RetrievalStage(retriever=retriever)

        # 평가용 데이터 준비
        results = []

        # 컬럼 이름 확인
        question_col = 'user_input' if 'user_input' in testset_df.columns else 'question'
        reference_col = 'reference' if 'reference' in testset_df.columns else 'ground_truth'

        for idx, row in testset_df.iterrows():
            question = row[question_col]
            reference = row.get(reference_col, "")

            logger.info(f"\n[평가 {idx+1}/{len(testset_df)}]")
            logger.info(f"  질문: {question[:50]}...")

            # RAG 실행
            result = rag_stage.query_with_sources(question)

            # RAGAS 평가를 위한 포맷
            eval_data = {
                "user_input": question,
                "response": result['answer'],
                "retrieved_contexts": [doc.page_content for doc in result['sources']],
                "reference": reference
            }
            results.append(eval_data)

            logger.info(f"  생성답변: {result['answer'][:50]}...")
            logger.info(f"  검색문서: {len(result['sources'])}개")

        return results

    except Exception as e:
        logger.error(f"❌ RAG 평가 실패: {e}")
        return None


def run_ragas_evaluation(results: List[Dict]):
    """RAGAS 메트릭으로 평가"""
    logger.info("\n📊 RAGAS 메트릭 평가 시작...")

    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall
        )

        # 데이터셋 생성
        dataset = EvaluationDataset.from_list(results)

        # LLM for metrics
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_wrapper = LangchainLLMWrapper(eval_llm)

        # 메트릭 설정
        metrics = [
            Faithfulness(llm=llm_wrapper),
            AnswerRelevancy(llm=llm_wrapper),
            ContextPrecision(llm=llm_wrapper),
            ContextRecall(llm=llm_wrapper)
        ]

        # 평가 실행
        logger.info("  평가 실행 중...")
        evaluation = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_wrapper
        )

        # 결과 출력
        logger.info("\n✅ 평가 완료!")

        if hasattr(evaluation, 'to_pandas'):
            result_df = evaluation.to_pandas()
            # 평균 점수 계산
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                if metric in result_df.columns:
                    avg = result_df[metric].mean()
                    logger.info(f"  {metric}: {avg:.3f}")
        else:
            for key, value in evaluation.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.3f}")

        return evaluation

    except Exception as e:
        logger.error(f"❌ RAGAS 평가 실패: {e}")
        logger.info(f"💡 힌트: {str(e)}")
        return None


def save_results(testset_df, rag_results, eval_results):
    """결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/evaluation/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테스트셋 저장
    testset_file = output_dir / f"testset_{timestamp}.csv"
    testset_df.to_csv(testset_file, index=False, encoding='utf-8-sig')
    logger.info(f"  💾 테스트셋: {testset_file}")

    # RAG 결과 저장
    if rag_results:
        rag_file = output_dir / f"rag_results_{timestamp}.json"
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_results, f, ensure_ascii=False, indent=2)
        logger.info(f"  💾 RAG 결과: {rag_file}")

    logger.info(f"\n✅ 모든 결과가 {output_dir}에 저장됨")


def main():
    """메인 실행"""
    print("\n" + "="*60)
    print("🚀 RAGAS Simple Test - 올바른 구현".center(60))
    print("="*60)

    try:
        # 1. 문서 로드
        logger.info("\n[1단계] 문서 로드")
        doc_path = "data/crawled/seoul_traffic/markdown_deduplicated"
        documents = load_simple_documents(doc_path, limit=5)

        # 2. 테스트셋 생성 (RAGAS 또는 수동)
        logger.info("\n[2단계] 테스트셋 생성")
        testset_df = create_testset_with_ragas(documents, size=5)

        # 3. RAG 평가
        logger.info("\n[3단계] RAG 평가")
        rag_results = evaluate_rag_with_testset(testset_df)

        # 4. RAGAS 메트릭 평가
        eval_results = None
        if rag_results:
            logger.info("\n[4단계] RAGAS 메트릭")
            eval_results = run_ragas_evaluation(rag_results)

        # 5. 결과 저장
        logger.info("\n[5단계] 결과 저장")
        save_results(testset_df, rag_results, eval_results)

        logger.info("\n🎉 테스트 완료!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자 중단")
    except Exception as e:
        logger.error(f"\n❌ 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()