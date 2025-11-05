#!/usr/bin/env python3
"""
RAGAS Evaluation Test Flight v2
- RAGAS 0.3.1 버전에 맞춰 수정
- 5개 샘플만 생성하는 테스트 스크립트
- 상세한 로깅으로 진행 상황 모니터링
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정 - 매우 상세하게
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 필요한 라이브러리 임포트
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports - 0.3.1 버전
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 평가 메트릭
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

# 우리 RAG 파이프라인 임포트
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
from src.rag_pipeline.stages.stage_06_retrieval import RetrievalStage


def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def load_documents(doc_path: str, limit: int = 10) -> List[Document]:
    """마크다운 문서 로드 - 간단한 방법 사용"""
    print_section("1. 문서 로드")
    logger.info(f"문서 경로: {doc_path}")

    documents = []
    doc_dir = Path(doc_path)

    # 마크다운 파일 직접 읽기
    md_files = list(doc_dir.glob("*.md"))[:limit]  # limit 적용

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file.name),
                    "file_path": str(md_file)
                }
            )
            documents.append(doc)
            logger.debug(f"  ✓ {md_file.name} 로드 완료")
        except Exception as e:
            logger.warning(f"  ✗ {md_file.name} 로드 실패: {e}")

    logger.info(f"✅ {len(documents)}개 문서 로드 완료")

    # 샘플 출력
    if documents:
        logger.info(f"첫 번째 문서 미리보기:")
        logger.info(f"  - 파일: {documents[0].metadata['source']}")
        logger.info(f"  - 길이: {len(documents[0].page_content)} chars")
        logger.info(f"  - 내용 시작: {documents[0].page_content[:200]}...")

    return documents


def setup_testset_generator() -> TestsetGenerator:
    """TestsetGenerator 설정 - RAGAS 0.3.1 버전"""
    print_section("2. TestsetGenerator 설정")

    logger.info("LLM 모델 초기화...")

    # Generator LLM (저렴한 모델)
    generator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )
    logger.info("  ✓ Generator LLM: gpt-4o-mini")

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    logger.info("  ✓ Embeddings: text-embedding-3-small")

    # RAGAS 0.3.1을 위한 Wrapper
    generator_llm_wrapper = LangchainLLMWrapper(generator_llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    # TestsetGenerator 생성
    generator = TestsetGenerator(
        llm=generator_llm_wrapper,
        embedding_model=embeddings_wrapper
    )

    logger.info("✅ TestsetGenerator 설정 완료")
    return generator


def generate_testset(generator: TestsetGenerator, documents: List[Document], size: int = 5):
    """테스트셋 생성 - RAGAS 0.3.1 버전"""
    print_section("3. 테스트셋 생성")

    logger.info(f"테스트 샘플 수: {size}")
    logger.info("\n🔄 생성 시작... (약 2-3분 소요)")

    try:
        start_time = time.time()

        # Query distribution 설정 (0.3.1 버전)
        query_dist = default_query_distribution(generator.llm.langchain_llm)
        logger.info("Query Distribution 설정 완료")

        # 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=size,
            query_distribution=query_dist
        )

        generation_time = time.time() - start_time
        logger.info(f"✅ 테스트셋 생성 완료 ({generation_time:.2f}초)")

        # DataFrame으로 변환
        df = testset.to_pandas()

        # 결과 출력
        logger.info(f"\n생성된 테스트셋 정보:")
        logger.info(f"  - 총 샘플 수: {len(df)}")
        logger.info(f"  - 컬럼: {df.columns.tolist()}")

        # 각 샘플 상세 출력
        for idx, row in df.iterrows():
            logger.info(f"\n[샘플 {idx+1}]")
            logger.info(f"  질문: {row.get('user_input', row.get('question', 'N/A'))[:100]}...")
            if 'reference' in row:
                logger.info(f"  답변: {row['reference'][:100]}...")
            elif 'ground_truth' in row:
                logger.info(f"  답변: {row['ground_truth'][:100]}...")

        return testset, df

    except Exception as e:
        logger.error(f"❌ 테스트셋 생성 실패: {e}")
        raise


def evaluate_rag_pipeline(testset_df):
    """생성된 테스트셋으로 RAG 파이프라인 평가"""
    print_section("4. RAG 파이프라인 평가")

    try:
        # Vector Store 로드
        logger.info("Vector Store 로드 중...")
        vector_store = VectorStoreStage()
        vector_store.load_vector_store(Path("data/vector_store"))
        logger.info("  ✓ Vector Store 로드 완료")

        # Retrieval Stage 설정
        logger.info("RAG Chain 설정 중...")
        retriever = vector_store.as_retriever(k=4)
        rag_stage = RetrievalStage(retriever=retriever)
        logger.info("  ✓ RAG Chain 준비 완료")

        # 컬럼 이름 확인 및 조정
        question_col = 'user_input' if 'user_input' in testset_df.columns else 'question'
        reference_col = 'reference' if 'reference' in testset_df.columns else 'ground_truth'

        # 각 질문에 대해 RAG 실행
        results = []
        for idx, row in testset_df.iterrows():
            logger.info(f"\n[테스트 {idx+1}/{len(testset_df)}]")

            question = row[question_col]
            ground_truth = row.get(reference_col, "")

            # RAG로 답변 생성
            start_time = time.time()
            result = rag_stage.query_with_sources(question)
            elapsed = time.time() - start_time

            logger.info(f"  질문: {question[:50]}...")
            logger.info(f"  생성된 답변: {result['answer'][:50]}...")
            if ground_truth:
                logger.info(f"  실제 답변: {ground_truth[:50]}...")
            logger.info(f"  소요 시간: {elapsed:.2f}초")
            logger.info(f"  사용 문서 수: {len(result['sources'])}")

            results.append({
                'user_input': question,
                'response': result['answer'],
                'retrieved_contexts': [doc.page_content for doc in result['sources']],
                'reference': ground_truth
            })

        return results

    except Exception as e:
        logger.error(f"❌ RAG 평가 실패: {e}")
        raise


def run_ragas_metrics(results):
    """RAGAS 메트릭으로 평가 - 0.3.1 버전"""
    print_section("5. RAGAS 메트릭 평가")

    try:
        from ragas import EvaluationDataset

        # EvaluationDataset 생성 (0.3.1 버전 방식)
        dataset = EvaluationDataset.from_list(results)

        # LLM for evaluation
        eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        eval_llm_wrapper = LangchainLLMWrapper(eval_llm)

        # 평가 메트릭 초기화 (0.3.1 버전)
        metrics = [
            Faithfulness(llm=eval_llm_wrapper),
            AnswerRelevancy(llm=eval_llm_wrapper),
            ContextPrecision(llm=eval_llm_wrapper),
            ContextRecall(llm=eval_llm_wrapper),
            AnswerCorrectness(llm=eval_llm_wrapper),
        ]

        logger.info("평가 메트릭:")
        for metric in metrics:
            logger.info(f"  - {metric.name}")

        logger.info("\n🔄 평가 실행 중...")

        # RAGAS 평가 실행
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=eval_llm_wrapper
        )

        # 결과 출력
        logger.info("\n✅ 평가 완료!")
        logger.info("\n📊 평가 결과:")

        # 결과를 딕셔너리로 변환
        if hasattr(evaluation_result, 'to_pandas'):
            result_df = evaluation_result.to_pandas()
            # 평균 계산
            for col in result_df.columns:
                if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
                    avg_score = result_df[col].mean()
                    logger.info(f"  {col}: {avg_score:.4f}")
        else:
            # 결과가 딕셔너리인 경우
            for metric_name, score in evaluation_result.items():
                if isinstance(score, (int, float)):
                    logger.info(f"  {metric_name}: {score:.4f}")

        return evaluation_result

    except Exception as e:
        logger.error(f"❌ RAGAS 평가 실패: {e}")
        logger.error("힌트: 데이터 형식이나 메트릭 초기화를 확인하세요.")
        return None


def save_results(testset_df, rag_results, metrics_results):
    """결과 저장"""
    print_section("6. 결과 저장")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/evaluation/test_flight")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테스트셋 저장
    testset_file = output_dir / f"testset_{timestamp}.csv"
    testset_df.to_csv(testset_file, index=False)
    logger.info(f"  ✓ 테스트셋: {testset_file}")

    # RAG 결과 저장
    if rag_results:
        rag_file = output_dir / f"rag_results_{timestamp}.json"
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_results, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✓ RAG 결과: {rag_file}")

    # 메트릭 저장
    if metrics_results:
        try:
            metrics_file = output_dir / f"metrics_{timestamp}.json"
            if hasattr(metrics_results, 'to_pandas'):
                # DataFrame인 경우
                metrics_results.to_pandas().to_json(
                    metrics_file,
                    orient='records',
                    force_ascii=False,
                    indent=2
                )
            else:
                # 딕셔너리인 경우
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics_results, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✓ 메트릭: {metrics_file}")
        except Exception as e:
            logger.warning(f"  ⚠️ 메트릭 저장 실패: {e}")

    logger.info(f"\n✅ 결과가 {output_dir}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    print("\n" + "🚀 RAGAS Evaluation Test Flight v2 시작 🚀".center(80))
    print("="*80)

    try:
        # 1. 문서 로드
        doc_path = "data/crawled/seoul_traffic/markdown_deduplicated"
        documents = load_documents(doc_path, limit=10)  # 테스트용으로 10개만

        if not documents:
            logger.error("문서를 로드할 수 없습니다.")
            return

        # 2. TestsetGenerator 설정
        generator = setup_testset_generator()

        # 3. 테스트셋 생성 (5개 샘플)
        testset, testset_df = generate_testset(generator, documents, size=5)

        # 4. RAG 파이프라인 평가
        rag_results = evaluate_rag_pipeline(testset_df)

        # 5. RAGAS 메트릭 평가
        metrics_results = None
        if rag_results:
            metrics_results = run_ragas_metrics(rag_results)

        # 6. 결과 저장
        save_results(testset_df, rag_results, metrics_results)

        print_section("완료!")
        logger.info("🎉 테스트 플라이트가 성공적으로 완료되었습니다!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자가 중단했습니다.")
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()