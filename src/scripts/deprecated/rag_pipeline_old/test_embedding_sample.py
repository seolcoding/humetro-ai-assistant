#!/usr/bin/env python3
"""
Test embedding with small sample of documents.

Tests Stage 1, 2, 3 pipeline with a few documents before running on full dataset.
"""

from pathlib import Path
from src.rag_pipeline.stages import (
    create_data_collection_stage,
    create_chunking_stage,
    create_embedding_stage
)
from src.common.logger import RAGLogger


def main():
    """Test embedding pipeline with small sample."""
    print("=" * 60)
    print("🧪 Testing Embedding Pipeline (Sample)")
    print("=" * 60)

    # Initialize logger
    logger = RAGLogger(experiment_name="test_embedding")

    # Stage 1: Data Collection (will load ALL files, then we sample)
    print("\n📂 Stage 1: Loading documents...")
    data_stage = create_data_collection_stage(logger=logger)
    all_documents = data_stage.load_documents()

    # Sample first 5 documents for testing
    sample_docs = all_documents[:5]
    print(f"✅ Sampled {len(sample_docs)} documents from {len(all_documents)} total")

    for i, doc in enumerate(sample_docs, 1):
        print(f"   {i}. {doc.metadata.get('source', 'unknown')[:50]}... "
              f"({len(doc.page_content)} chars)")

    # Stage 2: Chunking
    print("\n✂️  Stage 2: Chunking documents...")
    chunking_stage = create_chunking_stage(logger=logger)
    chunks = chunking_stage.chunk_documents(sample_docs)

    stats = chunking_stage.get_chunking_stats(sample_docs, chunks)
    print(f"✅ Created {stats['total_chunks']} chunks")
    print(f"   Avg chunks/doc: {stats['chunks_per_document']:.1f}")
    print(f"   Avg chunk length: {stats['avg_chunk_length']:.0f} chars")

    # Stage 3: Embedding
    print("\n🔢 Stage 3: Creating embeddings & FAISS index...")
    embedding_stage = create_embedding_stage(logger=logger)

    # Estimate cost before embedding
    est_stats = embedding_stage.get_embedding_stats(chunks)
    print(f"   Estimated tokens: {est_stats['estimated_tokens']:,}")
    print(f"   Estimated cost: ${est_stats['estimated_cost_usd']:.4f}")

    # Create vector store
    print("\n⚡ Generating embeddings (this may take a moment)...")
    vector_store = embedding_stage.create_vector_store(chunks)

    print(f"✅ FAISS vector store created")
    print(f"   Index size: {vector_store.index.ntotal} vectors")

    # Test similarity search with realistic user queries
    print("\n🔍 Testing similarity search with realistic user queries...")

    # 실제 민원인이 물어볼만한 구체적인 질문 3개
    test_queries = [
        "지하철역에서 강아지 사료 살 수 있는 곳이 어디인가요?",
        "공항에서 택시 탈 때 바가지요금 당하면 어떻게 신고하나요?",
        "공영주차장에 과태료 체납 차량 세워두면 단속되나요?"
    ]

    for query_num, query in enumerate(test_queries, 1):
        print(f"\n{'#'*80}")
        print(f"🙋 민원인 질문 #{query_num}")
        print(f"{'#'*80}")
        print(f"💬 질문: '{query}'")
        print(f"{'#'*80}")

        results = vector_store.similarity_search(query, k=3)

        print(f"\n📊 검색 결과: {len(results)}개 관련 문서 발견")

        for i, doc in enumerate(results, 1):
            print(f"\n{'='*80}")
            print(f"🔍 검색 결과 #{i}")
            print(f"{'='*80}")
            print(f"📄 출처: {doc.metadata.get('source', 'unknown')}")
            print(f"🔢 청크 ID: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"📏 청크 길이: {doc.metadata.get('chunk_length', len(doc.page_content))} chars")
            print(f"\n📝 전체 내용:")
            print(f"{'-'*80}")
            print(doc.page_content)
            print(f"{'-'*80}")
            print(f"\n💡 메타데이터:")
            for key, value in doc.metadata.items():
                print(f"  - {key}: {value}")
            print(f"{'='*80}\n")

        print(f"\n{'#'*80}\n")

    # Save to temp location
    temp_path = Path("./data/temp_vector_store")
    print(f"\n💾 Saving vector store to: {temp_path}")
    embedding_stage.save_vector_store(vector_store, temp_path)

    print(f"\n✅ Test complete!")
    print(f"   Vector store saved at: {temp_path}")
    print(f"   Ready for full dataset embedding (awaiting your approval)")

    print("=" * 60)


if __name__ == "__main__":
    main()
