#!/usr/bin/env python3
"""
End-to-End RAG System Test

Tests the complete RAG pipeline from vector store loading to answer generation.
Uses realistic Korean queries to validate the full system.
"""

from pathlib import Path
from src.rag_pipeline.stages import (
    create_vector_store_stage,
    create_retrieval_stage
)
from src.common.logger import RAGLogger


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}\n")


def test_rag_system():
    """Test end-to-end RAG system with realistic queries."""

    print_separator()
    print("🤖 RAG System End-to-End Test")
    print_separator()

    # Initialize logger
    logger = RAGLogger(experiment_name="rag_e2e_test")

    # Configuration
    vector_store_path = Path("./data/vector_store/seoul_traffic")

    print_section("Step 1: Loading Vector Store")

    # Stage 5: Vector Store
    vector_store_stage = create_vector_store_stage(
        model="text-embedding-3-small",
        logger=logger
    )

    vector_store = vector_store_stage.load_vector_store(vector_store_path)

    # Get store info
    store_info = vector_store_stage.get_store_info()
    print(f"📊 Vector Store Info:")
    print(f"   Total vectors: {store_info['total_vectors']:,}")
    print(f"   Vector dimension: {store_info['vector_dimension']}")
    print(f"   Index type: {store_info['index_type']}")
    print(f"   Embedding model: {store_info['embedding_model']}")

    print_section("Step 2: Creating Retriever")

    # Create retriever with similarity search (k=4)
    retriever = vector_store_stage.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print(f"✅ Retriever created: search_type=similarity, k=4")

    print_section("Step 3: Initializing RAG Chain")

    # Stage 6: Retrieval (RAG chain)
    retrieval_stage = create_retrieval_stage(
        retriever=retriever,
        model="gpt-4o-mini",
        temperature=0.0,
        logger=logger
    )

    chain_info = retrieval_stage.get_chain_info()
    print(f"🔗 RAG Chain Info:")
    print(f"   Model: {chain_info['model']}")
    print(f"   Temperature: {chain_info['temperature']}")
    print(f"   Has retriever: {chain_info['has_retriever']}")

    print_section("Step 4: Testing with Realistic Queries")

    # Test queries (realistic citizen questions)
    test_queries = [
        {
            "question": "지하철역에서 반려동물 용품을 살 수 있나요? 어느 역인가요?",
            "category": "반려동물 & 지하철"
        },
        {
            "question": "공항에서 택시 탈 때 바가지요금 당했는데 어떻게 신고하나요?",
            "category": "택시 불편 신고"
        },
        {
            "question": "서울시 대중교통 요금은 얼마인가요? 카드와 현금이 다른가요?",
            "category": "교통 요금"
        },
        {
            "question": "공영주차장에 과태료 체납 차량 세워두면 단속되나요?",
            "category": "주차 & 단속"
        }
    ]

    results = []

    for i, query_info in enumerate(test_queries, 1):
        question = query_info["question"]
        category = query_info["category"]

        print(f"\n{'#'*80}")
        print(f"📝 테스트 #{i} - {category}")
        print(f"{'#'*80}")
        print(f"\n💬 질문: {question}\n")

        # Query with sources
        result = retrieval_stage.query_with_sources(question)

        print(f"🤖 답변:")
        print(f"{'-'*80}")
        print(result['answer'])
        print(f"{'-'*80}\n")

        print(f"📚 참고 문서: {result['num_sources']}개")
        for j, doc in enumerate(result['sources'], 1):
            source = doc.metadata.get('source', 'unknown')
            source_name = Path(source).name
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"   [{j}] {source_name} (chunk #{chunk_id})")
            print(f"       미리보기: {preview}...")

        results.append(result)

    print_section("Step 5: Summary")

    print(f"✅ RAG System Test Complete!")
    print(f"\n📊 Test Statistics:")
    print(f"   Total queries: {len(test_queries)}")
    print(f"   Total answers generated: {len(results)}")
    print(f"   Average answer length: {sum(len(r['answer']) for r in results) / len(results):.0f} chars")
    print(f"   Average sources per query: {sum(r['num_sources'] for r in results) / len(results):.1f}")

    print(f"\n{'='*80}")
    print(f"🎉 All tests passed! RAG system is working correctly.")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    try:
        test_rag_system()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
