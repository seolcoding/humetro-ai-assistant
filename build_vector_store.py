#!/usr/bin/env python3
"""
Build FAISS Vector Store from Filtered Documents
=================================================

Runs complete RAG pipeline to create FAISS vector store:
1. Stage 1: Load markdown documents (filtered 2021-2025)
2. Stage 2: Chunk documents (512 tokens, 64 overlap)
3. Stage 3: Generate embeddings (text-embedding-3-large, 3072D)
4. Save FAISS vector store

Uses text-embedding-3-large (3072D) to match Neo4j KG embeddings
for fair Naive RAG vs KG RAG comparison.

Usage:
    python build_vector_store.py
    python build_vector_store.py --output data/vector_store/custom_path
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.common.logger import RAGLogger
from src.rag_pipeline.stages.stage_01_data_collection import DataCollectionStage
from src.rag_pipeline.stages.stage_02_chunking import ChunkingStage
from src.rag_pipeline.stages.stage_03_embedding import EmbeddingStage


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector store")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vector_store/seoul_traffic"),
        help="Output path for vector store"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model (default: text-embedding-3-large for KG compatibility)"
    )
    args = parser.parse_args()

    # Initialize logger
    logger = RAGLogger("VectorStoreBuild")

    print("\n" + "="*70)
    print("FAISS Vector Store Builder".center(70))
    print("="*70)
    print(f"\n📁 Output: {args.output}")
    print(f"🔧 Embedding Model: {args.embedding_model}")
    print(f"📐 Dimensions: 3072D (matches Neo4j KG)")
    print(f"📊 Source: data/crawled/seoul_traffic/markdown_filtered (874 files)")

    try:
        # Stage 1: Data Collection
        print("\n" + "="*70)
        print("[Stage 1/3] Data Collection".center(70))
        print("="*70)

        stage1 = DataCollectionStage(logger=logger)
        documents = stage1.load_documents()

        logger.info(f"✅ Loaded {len(documents)} documents")

        # Stage 2: Chunking
        print("\n" + "="*70)
        print("[Stage 2/3] Chunking".center(70))
        print("="*70)

        stage2 = ChunkingStage(
            chunk_size=512,
            chunk_overlap=64,
            logger=logger
        )
        chunks = stage2.chunk_documents(documents)

        stats = stage2.get_chunking_stats(documents, chunks)
        logger.info(f"✅ Created {stats['total_chunks']} chunks")
        logger.info(f"   Avg: {stats['avg_chunk_length']:.0f} chars/chunk")
        logger.info(f"   Chunks per doc: {stats['chunks_per_document']:.1f}")

        # Stage 3: Embedding + Vector Store
        print("\n" + "="*70)
        print("[Stage 3/3] Embedding & Vector Store Creation".center(70))
        print("="*70)

        stage3 = EmbeddingStage(
            model=args.embedding_model,
            batch_size=100,
            logger=logger
        )

        # Get cost estimate first
        cost_stats = stage3.get_embedding_stats(chunks)
        logger.info(f"📊 Estimated cost: ${cost_stats['estimated_cost_usd']:.2f}")
        logger.info(f"   Tokens: {cost_stats['estimated_tokens']:,}")
        logger.info(f"   Chunks: {cost_stats['total_chunks']:,}")

        # Create vector store
        logger.info("Creating FAISS vector store (this may take a few minutes)...")
        vector_store = stage3.create_vector_store(chunks)

        # Verify dimensions
        dimension = vector_store.index.d
        total_vectors = vector_store.index.ntotal
        index_type = vector_store.index.__class__.__name__

        logger.info(f"✅ Vector store created")
        logger.info(f"   Vectors: {total_vectors:,}")
        logger.info(f"   Dimensions: {dimension}D")
        logger.info(f"   Index type: {index_type}")
        logger.info(f"   Similarity: Cosine (normalized IP)")

        # Save to disk
        print("\n" + "-"*70)
        print("Saving Vector Store")
        print("-"*70)

        stage3.save_vector_store(vector_store, args.output)

        # Verify saved files
        index_file = args.output / "index.faiss"
        pkl_file = args.output / "index.pkl"

        index_size = index_file.stat().st_size / 1024 / 1024  # MB
        pkl_size = pkl_file.stat().st_size / 1024 / 1024  # MB

        logger.info(f"✅ Saved successfully")
        logger.info(f"   index.faiss: {index_size:.2f} MB")
        logger.info(f"   index.pkl: {pkl_size:.2f} MB")

        # Final summary
        print("\n" + "="*70)
        print("✅ Build Complete!".center(70))
        print("="*70)
        print(f"\n📁 Vector Store: {args.output}")
        print(f"📊 Total Vectors: {total_vectors:,}")
        print(f"📐 Dimensions: {dimension}D (matches Neo4j KG)")
        print(f"💰 Actual Cost: ${cost_stats['estimated_cost_usd']:.2f}")
        print("\n" + "="*70)

        # Next steps
        print("\n🚀 Next Steps:")
        print(f"   1. Verify: python src/rag_pipeline/dump_faiss_vectorstore.py")
        print(f"   2. Test retrieval: python run_benchmark.py --config config/healthcheck.json")
        print(f"   3. Compare: Naive RAG vs KG RAG benchmarks")

    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
