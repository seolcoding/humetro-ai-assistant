#!/usr/bin/env python3
"""
Test vector store with various user queries.

This script loads the saved vector store and tests it with
realistic user queries to verify RAG retrieval quality.
"""

from pathlib import Path
from src.rag_pipeline.stages import create_embedding_stage
from src.common.logger import RAGLogger


def main():
    """Test vector store with realistic queries."""
    print("=" * 80)
    print("🔍 Vector Store Query Testing")
    print("=" * 80)

    # Initialize
    logger = RAGLogger(experiment_name="test_queries")
    embedding_stage = create_embedding_stage(logger=logger)

    vector_store_path = Path("./data/vector_store/seoul_traffic")

    # Load vector store
    print(f"\n📂 Loading vector store from: {vector_store_path}")

    try:
        vector_store = embedding_stage.load_vector_store(vector_store_path)
        print(f"✅ Vector store loaded: {vector_store.index.ntotal:,} vectors")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return

    # Test queries - realistic user questions
    test_queries = [
        {
            "category": "반려동물",
            "query": "지하철역에서 강아지 사료 살 수 있는 곳이 어디인가요?"
        },
        {
            "category": "택시",
            "query": "공항에서 택시 탈 때 바가지요금 당하면 어떻게 신고하나요?"
        },
        {
            "category": "주차",
            "query": "공영주차장에 과태료 체납 차량 세워두면 단속되나요?"
        },
        {
            "category": "지하철",
            "query": "지하철 의자가 미끄러워서 불편한데 개선 계획 있나요?"
        },
        {
            "category": "교통",
            "query": "서울시 대중교통 요금은 얼마인가요?"
        }
    ]

    # Run tests
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'#'*80}")
        print(f"🙋 테스트 #{i} - {test_case['category']}")
        print(f"{'#'*80}")
        print(f"💬 질문: {test_case['query']}")
        print(f"{'#'*80}")

        # Search
        results = vector_store.similarity_search(test_case['query'], k=3)

        print(f"\n📊 검색 결과: {len(results)}개 관련 문서 발견\n")

        for j, doc in enumerate(results, 1):
            print(f"{'='*80}")
            print(f"🔍 검색 결과 #{j}")
            print(f"{'='*80}")
            print(f"📄 출처: {Path(doc.metadata.get('source', 'unknown')).name}")
            print(f"🔢 청크 ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"📏 청크 길이: {doc.metadata.get('chunk_length', len(doc.page_content))} chars")

            print(f"\n📝 내용:")
            print(f"{'-'*80}")
            print(doc.page_content)
            print(f"{'-'*80}\n")

        print(f"{'#'*80}\n")

    # Summary
    print("=" * 80)
    print("✅ Query Testing Complete!")
    print("=" * 80)
    print(f"📊 Tested {len(test_queries)} different query categories")
    print(f"💾 Vector store: {vector_store_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
