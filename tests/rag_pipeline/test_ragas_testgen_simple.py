#!/usr/bin/env python3
"""
RAGAS TestSet Generator 간단 테스트
- 5개 질문만 생성 시도
- 실패 원인 파악
"""

import sys
import logging
from pathlib import Path
from typing import List

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph


def load_sample_documents() -> List[Document]:
    """샘플 문서 로드 (markdown_deduplicated에서)"""
    logger.info("📄 샘플 문서 로드 중...")

    docs_dir = Path("data/crawled/seoul_traffic/markdown_deduplicated")

    if not docs_dir.exists():
        logger.error(f"문서 디렉토리 없음: {docs_dir}")
        return []

    # 처음 10개 파일 사용 (더 많은 문서로 클러스터링 시도)
    md_files = list(docs_dir.glob("*.md"))[:10]

    documents = []
    for md_file in md_files:
        content = md_file.read_text(encoding='utf-8')
        doc = Document(
            page_content=content,
            metadata={"source": str(md_file.name)}
        )
        documents.append(doc)

    logger.info(f"✅ 로드 완료: {len(documents)}개 문서")
    return documents


def test_testset_generation():
    """RAGAS TestSet 생성 테스트"""
    print("\n" + "="*60)
    print("🧪 RAGAS TestSet Generator 테스트".center(60))
    print("="*60)

    try:
        # 1. 문서 로드
        logger.info("\n[1단계] 문서 로드")
        documents = load_sample_documents()

        if not documents:
            logger.error("❌ 문서 로드 실패")
            return

        # 2. LLM & Embeddings 설정
        logger.info("\n[2단계] LLM & Embeddings 설정")

        # GPT-4o-mini 사용 (저렴하고 빠름)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        logger.info("  ✅ LLM: gpt-4o-mini")
        logger.info("  ✅ Embeddings: text-embedding-3-small")

        # RAGAS Wrapper (RAGAS 0.3.1 API)
        llm_wrapper = LangchainLLMWrapper(llm)
        embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

        # 3. TestSet Generator 생성
        logger.info("\n[3단계] TestSet Generator 초기화")

        generator = TestsetGenerator(
            llm=llm_wrapper,
            embedding_model=embeddings_wrapper,
        )

        logger.info("  ✅ Generator 초기화 완료")

        # 4. TestSet 생성 (5개만)
        logger.info("\n[4단계] TestSet 생성 시도 (5개 질문)")

        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=5,
            raise_exceptions=True,  # 오류 발생시 예외 발생
        )

        logger.info(f"✅ TestSet 생성 성공!")
        logger.info(f"  생성된 질문 수: {len(testset)}")

        # 5. 결과 출력
        logger.info("\n[5단계] 생성된 질문 확인")

        df = testset.to_pandas()
        print("\n생성된 질문:")
        print("-" * 60)
        for idx, row in df.iterrows():
            print(f"\n[질문 {idx+1}]")
            print(f"Q: {row.get('user_input', 'N/A')}")
            print(f"A: {row.get('reference', 'N/A')[:100]}...")

        # 6. 저장
        output_path = Path("data/evaluation/testsets/ragas_generated_5q.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"\n💾 저장 완료: {output_path}")
        logger.info("\n🎉 테스트 성공!")

    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}", exc_info=True)

        # 상세 오류 분석
        if "knowledge graph" in str(e).lower():
            logger.error("\n🔍 Knowledge Graph 오류 감지")
            logger.error("  원인: 문서에서 충분한 엔티티/관계 추출 실패")
            logger.error("  해결: 더 많은 문서 또는 더 상세한 문서 필요")

        elif "cluster" in str(e).lower():
            logger.error("\n🔍 클러스터링 오류 감지")
            logger.error("  원인: 문서 내용이 너무 짧거나 유사도 낮음")
            logger.error("  해결: 문서 개수 증가 또는 다양한 주제 포함")


if __name__ == "__main__":
    test_testset_generation()
