#!/usr/bin/env python3
"""
FAISS Vector Store Dump Script
================================

Exports local FAISS vector store metadata and sample data including:
- Vector statistics (count, dimensions, index type)
- Document metadata and sources
- Sample documents and embeddings
- Similarity search tests
- Full export options (JSON)

Usage:
    python src/rag_pipeline/dump_faiss_vectorstore.py
    python src/rag_pipeline/dump_faiss_vectorstore.py --full  # Export all documents
    python src/rag_pipeline/dump_faiss_vectorstore.py --store data/temp_vector_store  # Custom path
"""

import json
import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_vector_store_statistics(vector_store: FAISS, store_path: Path) -> Dict[str, Any]:
    """Get comprehensive vector store statistics"""

    print("\n" + "="*70)
    print("FAISS Vector Store Statistics".center(70))
    print("="*70)

    stats = {
        'store_path': str(store_path),
        'timestamp': datetime.now().isoformat()
    }

    # Index information
    index = vector_store.index
    stats['total_vectors'] = int(index.ntotal)
    stats['vector_dimension'] = int(index.d)
    stats['index_type'] = index.__class__.__name__

    print(f"\n📊 Total Vectors: {stats['total_vectors']:,}")
    print(f"📐 Vector Dimension: {stats['vector_dimension']}")
    print(f"🔧 Index Type: {stats['index_type']}")

    # Determine similarity metric
    if 'IP' in stats['index_type']:
        stats['similarity_metric'] = 'Cosine Similarity (Inner Product with normalized vectors)'
    elif 'L2' in stats['index_type']:
        stats['similarity_metric'] = 'L2 Distance (Euclidean)'
    else:
        stats['similarity_metric'] = 'Unknown'

    print(f"📏 Similarity Metric: {stats['similarity_metric']}")

    # Document store information
    docstore = vector_store.docstore
    stats['total_documents'] = len(docstore._dict) if hasattr(docstore, '_dict') else 'N/A'

    print(f"📄 Total Documents: {stats['total_documents']}")

    # Index to docstore mapping
    index_to_docstore = vector_store.index_to_docstore_id
    stats['index_mapping_size'] = len(index_to_docstore)

    print(f"🔗 Index Mappings: {stats['index_mapping_size']}")

    # File sizes
    index_file = store_path / "index.faiss"
    pkl_file = store_path / "index.pkl"

    stats['file_sizes'] = {
        'index_faiss': index_file.stat().st_size if index_file.exists() else 0,
        'index_pkl': pkl_file.stat().st_size if pkl_file.exists() else 0
    }

    print(f"\n💾 File Sizes:")
    print(f"   index.faiss: {stats['file_sizes']['index_faiss'] / 1024 / 1024:.2f} MB")
    print(f"   index.pkl: {stats['file_sizes']['index_pkl'] / 1024 / 1024:.2f} MB")

    return stats


def get_document_metadata_summary(vector_store: FAISS) -> Dict[str, Any]:
    """Analyze document metadata"""

    print("\n" + "-"*70)
    print("Document Metadata Summary:")
    print("-"*70)

    metadata_summary = {
        'sources': {},
        'metadata_keys': set(),
        'sample_metadata': []
    }

    docstore = vector_store.docstore
    if not hasattr(docstore, '_dict'):
        print("  ⚠️ Cannot access docstore documents")
        return metadata_summary

    docs = list(docstore._dict.values())
    print(f"\n  Total Documents: {len(docs)}")

    # Analyze metadata
    for doc in docs:
        # Track sources
        source = doc.metadata.get('source', 'unknown')
        metadata_summary['sources'][source] = metadata_summary['sources'].get(source, 0) + 1

        # Track metadata keys
        metadata_summary['metadata_keys'].update(doc.metadata.keys())

    # Convert set to list for JSON serialization
    metadata_summary['metadata_keys'] = sorted(list(metadata_summary['metadata_keys']))

    print(f"\n  Unique Sources: {len(metadata_summary['sources'])}")
    print(f"  Metadata Keys: {', '.join(metadata_summary['metadata_keys'])}")

    # Show top 10 sources
    print(f"\n  Top 10 Sources by Document Count:")
    sorted_sources = sorted(
        metadata_summary['sources'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for source, count in sorted_sources:
        source_name = Path(source).name if source != 'unknown' else 'unknown'
        print(f"    {source_name}: {count} docs")

    # Sample metadata
    if len(docs) > 0:
        metadata_summary['sample_metadata'] = [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'metadata_keys': list(doc.metadata.keys()),
                'content_length': len(doc.page_content)
            }
            for doc in docs[:5]
        ]

    return metadata_summary


def get_sample_documents(vector_store: FAISS, n: int = 5) -> List[Dict[str, Any]]:
    """Get sample documents"""

    print("\n" + "-"*70)
    print(f"Sample Documents (First {n}):")
    print("-"*70)

    samples = []

    docstore = vector_store.docstore
    if not hasattr(docstore, '_dict'):
        print("  ⚠️ Cannot access docstore documents")
        return samples

    docs = list(docstore._dict.values())[:n]

    for i, doc in enumerate(docs, 1):
        sample = {
            'index': i,
            'source': doc.metadata.get('source', 'unknown'),
            'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
            'content_length': len(doc.page_content),
            'metadata': doc.metadata
        }
        samples.append(sample)

        source_name = Path(sample['source']).name if sample['source'] != 'unknown' else 'unknown'
        print(f"\n  Document {i}:")
        print(f"    Source: {source_name}")
        print(f"    Length: {sample['content_length']} chars")
        print(f"    Preview: {sample['content_preview'][:100]}...")

    return samples


def test_similarity_search(vector_store: FAISS) -> Dict[str, Any]:
    """Test similarity search functionality"""

    print("\n" + "-"*70)
    print("Similarity Search Test:")
    print("-"*70)

    test_queries = [
        "서울 지하철 1호선",
        "버스 노선 정보",
        "교통 카드 사용법"
    ]

    search_results = {}

    for query in test_queries:
        print(f"\n  Query: '{query}'")

        results = vector_store.similarity_search_with_score(query, k=3)

        query_results = []
        for j, (doc, score) in enumerate(results, 1):
            source_name = Path(doc.metadata.get('source', 'unknown')).name
            result = {
                'rank': j,
                'score': float(score),
                'source': source_name,
                'content_preview': doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
            }
            query_results.append(result)

            print(f"    [{j}] Score: {score:.4f} | Source: {source_name}")
            print(f"        {result['content_preview'][:80]}...")

        search_results[query] = query_results

    return search_results


def export_full_documents(
    vector_store: FAISS,
    output_dir: Path,
    include_embeddings: bool = False
) -> Path:
    """Export all documents to JSON"""

    print("\n" + "="*70)
    print("Exporting Full Documents".center(70))
    print("="*70)

    docstore = vector_store.docstore
    if not hasattr(docstore, '_dict'):
        print("  ⚠️ Cannot access docstore documents")
        return None

    docs = list(docstore._dict.values())
    print(f"\n📦 Exporting {len(docs)} documents...")

    documents_export = []

    for doc in docs:
        doc_data = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'content_length': len(doc.page_content)
        }
        documents_export.append(doc_data)

    # Save documents
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"faiss_documents_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents_export, f, ensure_ascii=False, indent=2, default=str)

    print(f"   ✅ Saved to: {output_file}")

    # Optionally export embeddings (as separate file due to size)
    if include_embeddings:
        print(f"\n📊 Exporting embeddings...")

        # Get all embeddings from index
        embeddings = []
        for i in range(vector_store.index.ntotal):
            vec = vector_store.index.reconstruct(i)
            embeddings.append(vec.tolist())

        embeddings_file = output_dir / f"faiss_embeddings_{timestamp}.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2)

        print(f"   ✅ Saved to: {embeddings_file}")
        print(f"   📐 Shape: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Dump FAISS vector store data")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("data/vector_store/seoul_traffic"),
        help="Path to FAISS vector store directory"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export full documents to JSON"
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Include embeddings in full export (large file)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/faiss_dumps"),
        help="Output directory for exports (default: data/faiss_dumps)"
    )
    args = parser.parse_args()

    try:
        # Check if store exists
        if not args.store.exists():
            print(f"\n❌ Vector store not found: {args.store}")
            print("\nAvailable vector stores:")
            for vs_path in Path("data").rglob("index.faiss"):
                print(f"  - {vs_path.parent}")
            sys.exit(1)

        # Initialize embeddings (must match the model used to create the store)
        print("\n🔧 Initializing embeddings model...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Load vector store
        print(f"📂 Loading vector store from: {args.store}")
        vector_store = FAISS.load_local(
            str(args.store),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True
        )
        print("   ✅ Vector store loaded successfully")

        # Get statistics
        stats = get_vector_store_statistics(vector_store, args.store)

        # Get metadata summary
        metadata_summary = get_document_metadata_summary(vector_store)

        # Get sample documents
        samples = get_sample_documents(vector_store, n=5)

        # Test similarity search
        search_results = test_similarity_search(vector_store)

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'metadata_summary': metadata_summary,
            'sample_documents': samples,
            'search_test_results': search_results
        }

        # Save summary
        summary_dir = args.output_dir
        summary_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"faiss_summary_{timestamp}.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "="*70)
        print("Summary Saved".center(70))
        print("="*70)
        print(f"\n📄 {summary_file}")

        # Full export if requested
        if args.full:
            exported_file = export_full_documents(
                vector_store,
                args.output_dir,
                include_embeddings=args.embeddings
            )

            if exported_file:
                print("\n" + "="*70)
                print("Export Complete".center(70))
                print("="*70)
                print(f"\n📁 Documents: {exported_file}")

        print("\n" + "="*70)
        print("✅ Dump Complete!".center(70))
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
