#!/usr/bin/env python3
"""
Balanced TestSet Generator: Single-hop 25개 + Multi-hop 25개
===========================================================

50개 질문을 균형있게 생성하고 벤치마크용 캐시로 고정

RAGAS 0.3.1 Synthesizers:
- SingleHopSpecificQuerySynthesizer
- MultiHopAbstractQuerySynthesizer
- MultiHopSpecificQuerySynthesizer

Author: Claude Code
Date: 2025-11-06
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import logging
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_documents(source_dir: str, num_docs: int = 30) -> list[Document]:
    """문서 로드"""
    docs_path = Path(source_dir)
    md_files = list(docs_path.glob("*.md"))[:num_docs]

    logger.info(f"📄 문서 로드: {len(md_files)}개")

    documents = []
    for md_file in md_files:
        content = md_file.read_text(encoding='utf-8')
        doc = Document(
            page_content=content,
            metadata={"source": str(md_file.name)}
        )
        documents.append(doc)

    return documents


def main():
    """균형잡힌 50개 질문셋 생성"""

    logger.info("="*70)
    logger.info("  Balanced TestSet: 25 Single-hop + 25 Multi-hop".center(70))
    logger.info("="*70)

    # 1. 문서 로드
    documents = load_documents(
        "data/crawled/seoul_traffic/markdown_deduplicated",
        num_docs=30
    )

    # 2. LLM & Embeddings 설정
    logger.info("\n[모델 설정]")
    logger.info("  LLM: gpt-4o-mini")
    logger.info("  Embeddings: text-embedding-3-large")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    llm_wrapper = LangchainLLMWrapper(llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    # 3. Custom Query Distribution (50% Single-hop, 50% Multi-hop specific)
    # Note: MultiHopAbstract 제외 (knowledge graph 클러스터 부족 문제)
    logger.info("\n[Query Distribution 설정]")
    logger.info("  SingleHopSpecific: 50%")
    logger.info("  MultiHopSpecific: 50%")

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.5),
        (MultiHopSpecificQuerySynthesizer(llm=llm_wrapper), 0.5),
    ]

    # 4. Generator 생성
    generator = TestsetGenerator(
        llm=llm_wrapper,
        embedding_model=embeddings_wrapper,
    )

    # 5. TestSet 생성 (100개 요청하여 충분한 여유 확보)
    logger.info("\n[TestSet 생성: 100개 요청 (custom distribution 적용)]")

    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=100,
        query_distribution=query_distribution,  # 여기에 전달!
        raise_exceptions=True,
    )

    testset_df = testset.to_pandas()
    logger.info(f"✅ 생성 완료: {len(testset_df)}개 질문")

    # 6. synthesizer_name 기반 분류
    logger.info("\n[Synthesizer 분포]")
    if 'synthesizer_name' in testset_df.columns:
        synth_counts = testset_df['synthesizer_name'].value_counts()
        for synth, count in synth_counts.items():
            logger.info(f"  {synth}: {count}개")

        # Single-hop과 Multi-hop 분리
        single_hop_df = testset_df[
            testset_df['synthesizer_name'].str.contains('single_hop', case=False, na=False)
        ]
        multi_hop_df = testset_df[
            testset_df['synthesizer_name'].str.contains('multi_hop', case=False, na=False)
        ]

        logger.info(f"\n[필터링 전]")
        logger.info(f"  Single-hop: {len(single_hop_df)}개")
        logger.info(f"  Multi-hop: {len(multi_hop_df)}개")

        # 충분한지 확인
        if len(single_hop_df) < 25 or len(multi_hop_df) < 25:
            logger.error(f"\n❌ 부족한 질문!")
            logger.error(f"   Single-hop: {len(single_hop_df)}/25")
            logger.error(f"   Multi-hop: {len(multi_hop_df)}/25")
            logger.error(f"   다시 실행하거나 distribution 조정 필요")
            return

        # 정확히 25개씩 선택 (랜덤 샘플링으로 다양성 확보)
        single_hop_25 = single_hop_df.sample(n=25, random_state=42)
        multi_hop_25 = multi_hop_df.sample(n=25, random_state=42)

        # 합치고 섞기
        balanced_df = pd.concat([single_hop_25, multi_hop_25]).sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"\n✅ 균형 조정 완료: {len(balanced_df)}개 질문")
        logger.info(f"  Single-hop: {len(single_hop_25)}개")
        logger.info(f"  Multi-hop: {len(multi_hop_25)}개")

        # 7. 저장 (벤치마크용 캐시)
        cache_dir = Path("data/evaluation/testsets")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        cache_file = cache_dir / f"testset_balanced_50_{timestamp}.csv"
        json_file = cache_file.with_suffix('.json')

        balanced_df.to_csv(cache_file, index=False, encoding='utf-8')
        balanced_df.to_json(json_file, orient='records', force_ascii=False, indent=2)

        logger.info(f"\n💾 저장 완료:")
        logger.info(f"  CSV: {cache_file.name}")
        logger.info(f"  JSON: {json_file.name}")

        # 8. 최종 확인
        logger.info(f"\n[최종 검증]")
        final_single = balanced_df[balanced_df['synthesizer_name'].str.contains('single_hop', case=False, na=False)]
        final_multi = balanced_df[balanced_df['synthesizer_name'].str.contains('multi_hop', case=False, na=False)]
        logger.info(f"  ✅ Single-hop: {len(final_single)}개")
        logger.info(f"  ✅ Multi-hop: {len(final_multi)}개")
        logger.info(f"  ✅ Total: {len(balanced_df)}개")

    else:
        logger.error("\n❌ synthesizer_name 컬럼 없음!")
        logger.info(f"   컬럼: {testset_df.columns.tolist()}")

    logger.info("\n" + "="*70)
    logger.info("✅ 완료: 이 파일을 벤치마크 설정에 사용하세요")
    logger.info("="*70)


if __name__ == "__main__":
    main()
