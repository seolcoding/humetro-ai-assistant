#!/usr/bin/env python3
"""
Build Dasan Call Center Vector Store
=====================================

Creates a FAISS vector store from Dasan call center markdown documents.
Uses text-embedding-3-large for high-quality semantic search.

Usage:
    # Build full vector store (all 10,757 documents)
    python scripts/build_dasan_vector_store.py

    # Build with specific category
    python scripts/build_dasan_vector_store.py --category 교통

    # Quick test (100 documents)
    python scripts/build_dasan_vector_store.py --max-docs 100 --output data/vector_store/dasan_test
"""

import argparse
import logging
from pathlib import Path
from typing import List
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_markdown_documents(
    base_path: Path,
    max_docs: int = None
) -> List[Document]:
    """
    Load markdown documents from Dasan organized directory.

    Args:
        base_path: Base directory containing markdown files
        max_docs: Maximum number of documents to load (None = all)

    Returns:
        List of LangChain Documents
    """
    logger.info(f"📂 Loading documents from: {base_path}")

    # Find all markdown files (excluding index.md and README.md)
    md_files = [
        f for f in base_path.rglob("*.md")
        if f.name not in ["index.md", "README.md"]
    ]

    logger.info(f"  Found {len(md_files)} markdown files")

    if max_docs:
        md_files = md_files[:max_docs]
        logger.info(f"  Limited to {max_docs} files for testing")

    documents = []
    skipped = 0

    for md_file in md_files:
        try:
            # Read markdown content
            content = md_file.read_text(encoding='utf-8')

            # Skip empty files
            if len(content.strip()) < 100:
                skipped += 1
                continue

            # Extract metadata from path
            relative_path = md_file.relative_to(base_path)
            category = str(relative_path.parent.parent) if relative_path.parent.parent != Path(".") else "기타"

            # Parse YAML frontmatter if exists
            metadata = {
                "source": str(md_file),
                "category": category,
                "filename": md_file.name,
                "doc_length": len(content)
            }

            # Extract YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    yaml_content = parts[1]
                    # Simple parsing (could use PyYAML for robust parsing)
                    for line in yaml_content.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip()] = value.strip()

                    # Use markdown content after frontmatter
                    content = parts[2].strip()

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        except Exception as e:
            logger.warning(f"  ⚠️ Failed to load {md_file.name}: {e}")
            skipped += 1

    logger.info(f"  ✅ Loaded {len(documents)} documents")
    if skipped > 0:
        logger.info(f"  ⚠️ Skipped {skipped} files")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of documents to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    logger.info(f"✂️ Chunking documents (size={chunk_size}, overlap={chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(f"  ✅ Created {len(chunks)} chunks from {len(documents)} documents")
    logger.info(f"  📊 Average: {len(chunks) / len(documents):.1f} chunks per document")

    return chunks


def build_vector_store(
    chunks: List[Document],
    output_path: Path,
    model: str = "text-embedding-3-large"
) -> FAISS:
    """
    Build FAISS vector store with embeddings.

    Args:
        chunks: List of document chunks
        output_path: Path to save vector store
        model: OpenAI embedding model

    Returns:
        FAISS vector store
    """
    logger.info(f"🔨 Building vector store with {model}")
    logger.info(f"  📊 Total chunks: {len(chunks)}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=model)

    # Build vector store (with batching for large datasets)
    # OpenAI API limit: 300k tokens/request
    # Safe batch size: ~200 chunks (avg 1000 tokens/chunk = 200k tokens/batch)
    batch_size = 200
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_store is None:
            # Create initial vector store
            vector_store = FAISS.from_documents(
                batch,
                embeddings,
                normalize_L2=True  # Use cosine similarity
            )
        else:
            # Add to existing vector store
            batch_store = FAISS.from_documents(
                batch,
                embeddings,
                normalize_L2=True
            )
            vector_store.merge_from(batch_store)

    logger.info(f"  ✅ Vector store created: {vector_store.index.ntotal:,} vectors")

    # Save to disk
    output_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_path))

    logger.info(f"  💾 Saved to: {output_path}")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    import json
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model": model,
        "total_vectors": vector_store.index.ntotal,
        "total_documents": len(chunks),
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "similarity_metric": "cosine"
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"  📝 Metadata saved: {metadata_file}")

    return vector_store


def main():
    parser = argparse.ArgumentParser(
        description="Build Dasan call center vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/AI_HUB_DASAN_QA/03_markdown_full"),
        help="Input directory with markdown files"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/AI_HUB_DASAN_QA/07_vector_stores/full"),
        help="Output directory for vector store"
    )

    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="OpenAI embedding model"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks"
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum number of documents (for testing)"
    )

    parser.add_argument(
        "--category",
        help="Specific category to process (e.g., 교통)"
    )

    args = parser.parse_args()

    # Determine input path
    input_path = args.input_dir
    if args.category:
        input_path = input_path / args.category
        if not input_path.exists():
            logger.error(f"❌ Category path not found: {input_path}")
            return

    # Validate input
    if not input_path.exists():
        logger.error(f"❌ Input directory not found: {input_path}")
        logger.info("💡 Expected structure: data/dasan_call/organized_markdown_full/")
        return

    logger.info("="*70)
    logger.info(" "*15 + "Dasan Vector Store Builder")
    logger.info("="*70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Chunk: {args.chunk_size} chars (overlap: {args.chunk_overlap})")
    if args.max_docs:
        logger.info(f"Limit: {args.max_docs} documents (TEST MODE)")
    logger.info("")

    try:
        # Step 1: Load documents
        documents = load_markdown_documents(
            base_path=input_path,
            max_docs=args.max_docs
        )

        if not documents:
            logger.error("❌ No documents loaded!")
            return

        # Step 2: Chunk documents
        chunks = chunk_documents(
            documents=documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        # Step 3: Build vector store
        vector_store = build_vector_store(
            chunks=chunks,
            output_path=args.output,
            model=args.model
        )

        logger.info("="*70)
        logger.info("✅ Vector store build completed!")
        logger.info("="*70)
        logger.info(f"📊 Statistics:")
        logger.info(f"  - Documents: {len(documents):,}")
        logger.info(f"  - Chunks: {len(chunks):,}")
        logger.info(f"  - Vectors: {vector_store.index.ntotal:,}")
        logger.info(f"  - Location: {args.output}")
        logger.info("")
        logger.info("🚀 Ready to use with unified_benchmark_v4_dasan.py!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Build interrupted by user")
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
