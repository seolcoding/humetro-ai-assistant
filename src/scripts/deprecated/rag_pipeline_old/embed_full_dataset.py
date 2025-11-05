#!/usr/bin/env python3
"""
Embed full dataset (1,969 documents) and create FAISS vector store.

This script:
1. Loads all 1,969 markdown documents
2. Chunks them into ~512 token pieces
3. Generates embeddings using OpenAI
4. Saves FAISS vector store to disk

Estimated cost: ~$0.08 (3.13M tokens)
Estimated time: 2-3 minutes
"""

from pathlib import Path
from datetime import datetime
from src.rag_pipeline.stages import (
    create_data_collection_stage,
    create_chunking_stage,
    create_embedding_stage
)
from src.common.logger import RAGLogger


def main():
    """Embed full dataset and save vector store."""
    print("=" * 80)
    print("🚀 Full Dataset Embedding Pipeline")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Initialize logger
    logger = RAGLogger(experiment_name="full_dataset_embedding")

    # Stage 1: Data Collection
    print("\n📂 Stage 1: Loading all documents...")
    data_stage = create_data_collection_stage(logger=logger)

    if not data_stage.check_cache_exists():
        print("❌ Error: Cache not found. Please run crawler first.")
        return

    all_documents = data_stage.load_documents()
    print(f"✅ Loaded {len(all_documents)} documents")

    collection_stats = data_stage.get_collection_stats(all_documents)
    print(f"\n📊 Collection Statistics:")
    print(f"   Total documents: {collection_stats['total_documents']:,}")
    print(f"   Total characters: {collection_stats['total_characters']:,}")
    print(f"   Avg document length: {collection_stats['avg_document_length']:,} chars")
    print(f"   Min document length: {collection_stats['min_document_length']:,} chars")
    print(f"   Max document length: {collection_stats['max_document_length']:,} chars")

    # Stage 2: Chunking
    print("\n✂️  Stage 2: Chunking documents...")
    chunking_stage = create_chunking_stage(logger=logger)
    chunks = chunking_stage.chunk_documents(all_documents)

    chunking_stats = chunking_stage.get_chunking_stats(all_documents, chunks)
    print(f"✅ Created {chunking_stats['total_chunks']:,} chunks")
    print(f"\n📊 Chunking Statistics:")
    print(f"   Total chunks: {chunking_stats['total_chunks']:,}")
    print(f"   Chunks per document: {chunking_stats['chunks_per_document']:.1f}")
    print(f"   Avg chunk length: {chunking_stats['avg_chunk_length']:.0f} chars")
    print(f"   Min chunk length: {chunking_stats['min_chunk_length']:,} chars")
    print(f"   Max chunk length: {chunking_stats['max_chunk_length']:,} chars")

    # Stage 3: Embedding - Cost Estimation
    print("\n🔢 Stage 3: Preparing embeddings...")
    embedding_stage = create_embedding_stage(logger=logger)

    embedding_stats = embedding_stage.get_embedding_stats(chunks)
    print(f"\n💰 Cost Estimation:")
    print(f"   Total chunks: {embedding_stats['total_chunks']:,}")
    print(f"   Total characters: {embedding_stats['total_characters']:,}")
    print(f"   Estimated tokens: {embedding_stats['estimated_tokens']:,}")
    print(f"   Estimated cost: ${embedding_stats['estimated_cost_usd']:.4f}")

    # User confirmation
    print(f"\n{'='*80}")
    print("⚠️  WARNING: This will call OpenAI API and incur costs!")
    print(f"   Estimated cost: ${embedding_stats['estimated_cost_usd']:.4f}")
    print(f"   Estimated time: 2-3 minutes")
    print(f"{'='*80}")

    response = input("\n🤔 Do you want to proceed? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("\n❌ Embedding cancelled by user.")
        return

    # Create embeddings
    print("\n⚡ Generating embeddings (this will take 2-3 minutes)...")
    print("   Please wait... ☕")

    start_time = datetime.now()
    vector_store = embedding_stage.create_vector_store(chunks)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✅ FAISS vector store created!")
    print(f"   Index size: {vector_store.index.ntotal:,} vectors")
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Save vector store
    save_path = Path("./data/vector_store/seoul_traffic")
    print(f"\n💾 Saving vector store to: {save_path}")
    embedding_stage.save_vector_store(vector_store, save_path)

    print(f"\n{'='*80}")
    print("✅ Full Dataset Embedding Complete!")
    print(f"{'='*80}")
    print(f"📊 Final Statistics:")
    print(f"   Documents processed: {len(all_documents):,}")
    print(f"   Chunks created: {len(chunks):,}")
    print(f"   Vectors embedded: {vector_store.index.ntotal:,}")
    print(f"   Total time: {duration:.1f} seconds")
    print(f"   Vector store saved at: {save_path}")
    print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print(f"\n💡 Next Steps:")
    print(f"   1. Run verification: uv run python src/scripts/verify_vector_store.py")
    print(f"   2. Test queries: uv run python src/scripts/test_vector_store_queries.py")


if __name__ == "__main__":
    main()
