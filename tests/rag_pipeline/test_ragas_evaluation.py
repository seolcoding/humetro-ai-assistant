#!/usr/bin/env python3
"""
RAGAS Evaluation Test Flight
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
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# RAGAS 및 필요한 라이브러리 임포트
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS imports
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    default_query_distribution
)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
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
    """마크다운 문서 로드"""
    print_section("1. 문서 로드")
    logger.info(f"문서 경로: {doc_path}")

    # DirectoryLoader 사용
    loader = DirectoryLoader(
        doc_path,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
        max_concurrency=4
    )

    start_time = time.time()
    documents = loader.load()
    load_time = time.time() - start_time

    # 제한된 수만 사용 (테스트용)
    if limit and len(documents) > limit:
        logger.info(f"전체 {len(documents)}개 중 {limit}개만 사용")
        documents = documents[:limit]

    logger.info(f"✅ {len(documents)}개 문서 로드 완료 ({load_time:.2f}초)")

    # 샘플 출력
    if documents:
        logger.debug(f"첫 번째 문서 미리보기:")
        logger.debug(f"  - 길이: {len(documents[0].page_content)} chars")
        logger.debug(f"  - 메타데이터: {documents[0].metadata}")
        logger.debug(f"  - 내용 시작: {documents[0].page_content[:200]}...")

    return documents


def setup_testset_generator() -> TestsetGenerator:
    """TestsetGenerator 설정"""
    print_section("2. TestsetGenerator 설정")

    logger.info("LLM 모델 초기화...")

    # Generator LLM (저렴한 모델)
    generator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        verbose=True
    )
    logger.info("  ✓ Generator LLM: gpt-4o-mini")

    # Critic LLM (고급 모델)
    critic_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        verbose=True
    )
    logger.info("  ✓ Critic LLM: gpt-4o")

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=True
    )
    logger.info("  ✓ Embeddings: text-embedding-3-small")

    # TestsetGenerator 생성
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings
    )

    logger.info("✅ TestsetGenerator 설정 완료")
    return generator


def generate_testset(generator: TestsetGenerator, documents: List[Document], size: int = 5):
    """테스트셋 생성"""
    print_section("3. 테스트셋 생성")

    # Evolution 분포 설정
    distributions = {
        simple: 0.5,        # 단순 질문
        multi_context: 0.3, # 여러 문서 참조
        reasoning: 0.2      # 추론 필요
    }

    logger.info(f"테스트 샘플 수: {size}")
    logger.info("Evolution 분포:")
    for evo_type, ratio in distributions.items():
        logger.info(f"  - {evo_type.__name__}: {ratio:.0%}")

    logger.info("\n🔄 생성 시작... (약 2-3분 소요)")

    try:
        start_time = time.time()

        # 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            test_size=size,
            distributions=distributions,
            with_debugging_logs=True,  # 디버깅 로그 활성화
            run_config={
                "max_retries": 3,
                "seed": 42
            }
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
            logger.info(f"  질문: {row['question'][:100]}...")
            logger.info(f"  답변: {row['ground_truth'][:100]}...")
            logger.info(f"  진화타입: {row.get('evolution_type', 'N/A')}")
            logger.info(f"  메타데이터: {row.get('metadata', {})}")

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

        # 각 질문에 대해 RAG 실행
        results = []
        for idx, row in testset_df.iterrows():
            logger.info(f"\n[테스트 {idx+1}/{len(testset_df)}]")
            question = row['question']
            ground_truth = row['ground_truth']

            # RAG로 답변 생성
            start_time = time.time()
            result = rag_stage.query_with_sources(question)
            elapsed = time.time() - start_time

            logger.info(f"  질문: {question[:50]}...")
            logger.info(f"  생성된 답변: {result['answer'][:50]}...")
            logger.info(f"  실제 답변: {ground_truth[:50]}...")
            logger.info(f"  소요 시간: {elapsed:.2f}초")
            logger.info(f"  사용 문서 수: {len(result['sources'])}")

            results.append({
                'question': question,
                'answer': result['answer'],
                'contexts': [doc.page_content for doc in result['sources']],
                'ground_truth': ground_truth
            })

        return results

    except Exception as e:
        logger.error(f"❌ RAG 평가 실패: {e}")
        raise


def run_ragas_metrics(results):
    """RAGAS 메트릭으로 평가"""
    print_section("5. RAGAS 메트릭 평가")

    try:
        from datasets import Dataset

        # Dataset 생성
        dataset = Dataset.from_list(results)

        # 평가 메트릭 선택
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]

        logger.info("평가 메트릭:")
        for metric in metrics:
            logger.info(f"  - {metric.name}")

        logger.info("\n🔄 평가 실행 중...")

        # RAGAS 평가 실행
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ChatOpenAI(model="gpt-4o-mini"),
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
        )

        # 결과 출력
        logger.info("\n✅ 평가 완료!")
        logger.info("\n📊 평가 결과:")
        for metric_name, score in evaluation_result.items():
            logger.info(f"  {metric_name}: {score:.4f}")

        return evaluation_result

    except Exception as e:
        logger.error(f"❌ RAGAS 평가 실패: {e}")
        logger.error("Hint: ground_truth가 없거나 형식이 맞지 않을 수 있습니다.")
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
    rag_file = output_dir / f"rag_results_{timestamp}.json"
    with open(rag_file, 'w', encoding='utf-8') as f:
        json.dump(rag_results, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ RAG 결과: {rag_file}")

    # 메트릭 저장
    if metrics_results:
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(dict(metrics_results), f, indent=2)
        logger.info(f"  ✓ 메트릭: {metrics_file}")

    logger.info(f"\n✅ 모든 결과가 {output_dir}에 저장되었습니다.")


def main():
    """메인 실행 함수"""
    print("\n" + "🚀 RAGAS Evaluation Test Flight 시작 🚀".center(80))
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