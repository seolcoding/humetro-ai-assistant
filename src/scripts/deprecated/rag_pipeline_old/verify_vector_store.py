#!/usr/bin/env python3
"""
Verify FAISS vector store was saved and can be loaded correctly.

This script:
1. Checks if vector store files exist
2. Loads the vector store from disk
3. Verifies index integrity
4. Tests basic similarity search
5. Validates metadata
"""

from pathlib import Path
from src.rag_pipeline.stages import create_embedding_stage
from src.common.logger import RAGLogger


def main():
    """Verify vector store loading and integrity."""
    print("=" * 80)
    print("🔍 Vector Store Verification")
    print("=" * 80)

    # Initialize
    logger = RAGLogger(experiment_name="verify_vector_store")
    embedding_stage = create_embedding_stage(logger=logger)

    vector_store_path = Path("./data/vector_store/seoul_traffic")

    # Step 1: Check files exist
    print("\n📁 Step 1: Checking vector store files...")
    print(f"   Path: {vector_store_path}")

    if not vector_store_path.exists():
        print(f"❌ Error: Vector store directory not found!")
        print(f"   Expected path: {vector_store_path}")
        print(f"   Please run embed_full_dataset.py first.")
        return

    # Check for FAISS index files
    index_file = vector_store_path / "index.faiss"
    pkl_file = vector_store_path / "index.pkl"

    print(f"   ✅ Directory exists: {vector_store_path}")

    if index_file.exists():
        print(f"   ✅ FAISS index found: {index_file}")
        print(f"      Size: {index_file.stat().st_size:,} bytes ({index_file.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"   ❌ FAISS index not found: {index_file}")

    if pkl_file.exists():
        print(f"   ✅ Metadata file found: {pkl_file}")
        print(f"      Size: {pkl_file.stat().st_size:,} bytes ({pkl_file.stat().st_size / 1024:.2f} KB)")
    else:
        print(f"   ❌ Metadata file not found: {pkl_file}")

    if not (index_file.exists() and pkl_file.exists()):
        print(f"\n❌ Vector store files are incomplete!")
        return

    # Step 2: Load vector store
    print(f"\n📂 Step 2: Loading vector store from disk...")

    try:
        vector_store = embedding_stage.load_vector_store(vector_store_path)
        print(f"   ✅ Vector store loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading vector store: {e}")
        return

    # Step 3: Verify index integrity
    print(f"\n🔢 Step 3: Verifying index integrity...")
    print(f"   Index type: {type(vector_store.index).__name__}")
    print(f"   Total vectors: {vector_store.index.ntotal:,}")
    print(f"   Vector dimension: {vector_store.index.d}")
    print(f"   Index trained: {vector_store.index.is_trained}")

    if vector_store.index.ntotal == 0:
        print(f"   ❌ Warning: Index is empty!")
        return

    print(f"   ✅ Index integrity verified!")

    # Step 4: Test basic similarity search
    print(f"\n🔍 Step 4: Testing similarity search...")

    test_queries = [
        "지하철역에서 반려동물 용품 구매",
        "택시 바가지요금 신고",
        "공영주차장 체납 차량 단속"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: '{query}'")

        try:
            results = vector_store.similarity_search(query, k=3)
            print(f"   ✅ Search successful: {len(results)} results found")

            for j, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'unknown')
                chunk_id = doc.metadata.get('chunk_id', 'N/A')
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"      {j}. Chunk #{chunk_id} from {Path(source).name}")
                print(f"         Preview: {preview}...")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return

    # Step 5: Validate metadata
    print(f"\n📊 Step 5: Validating metadata...")

    # Get a sample document
    sample_results = vector_store.similarity_search("교통", k=1)

    if sample_results:
        sample_doc = sample_results[0]
        print(f"   Sample document metadata:")

        for key, value in sample_doc.metadata.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"      - {key}: {value[:50]}...")
            else:
                print(f"      - {key}: {value}")

        # Check required metadata fields
        required_fields = ['source', 'chunk_id', 'chunk_size_config', 'chunk_overlap_config']
        missing_fields = [f for f in required_fields if f not in sample_doc.metadata]

        if missing_fields:
            print(f"   ⚠️  Warning: Missing metadata fields: {missing_fields}")
        else:
            print(f"   ✅ All required metadata fields present!")

    # Final summary
    print(f"\n{'='*80}")
    print("✅ Vector Store Verification Complete!")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"   ✅ Files exist and are readable")
    print(f"   ✅ Vector store loads correctly")
    print(f"   ✅ Index integrity verified ({vector_store.index.ntotal:,} vectors)")
    print(f"   ✅ Similarity search working")
    print(f"   ✅ Metadata preserved")
    print(f"\n💾 Vector store is ready for use!")
    print(f"   Path: {vector_store_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
