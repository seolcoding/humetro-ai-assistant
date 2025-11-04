"""
Stage 2: Chunking

Splits cleaned documents into optimally-sized chunks for embedding and retrieval.
Optimized for Korean text with 8B local models.
"""

from typing import List, Optional
from langchain_core.documents import Document

from src.core.chunking import TextChunker
from src.common.logger import RAGLogger


class ChunkingStage:
    """
    Chunking Stage for RAG Pipeline.

    Splits documents into chunks optimized for:
    - Korean language processing
    - 8B parameter local models (e.g., Llama3.1-8B)
    - Multilingual embedding models (max 512 tokens)

    Configuration based on research:
    - 512 tokens: Optimal for 8B models and embedding model limits
    - 64 token overlap (12.5%): Maintains context between chunks
    - Korean-optimized separators: \n\n, \n, ". ", " ", ""
    """

    # Optimal settings for Korean RAG with 8B models
    DEFAULT_CHUNK_SIZE = 512      # tokens (~768 chars for Korean)
    DEFAULT_OVERLAP = 64          # 12.5% overlap
    KOREAN_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_OVERLAP,
        separators: List[str] = None,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Chunking Stage.

        Args:
            chunk_size: Maximum chunk size in tokens (default: 512)
            chunk_overlap: Overlap between chunks in tokens (default: 64)
            separators: Custom separators (default: Korean-optimized)
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.KOREAN_SEPARATORS
        self.logger = logger

        # Create chunker instance
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators
        )

        if self.logger:
            self.logger.info(
                f"ChunkingStage initialized: chunk_size={chunk_size}, "
                f"overlap={chunk_overlap}"
            )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of cleaned documents from Stage 1

        Returns:
            List of chunked documents with metadata

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("No documents provided for chunking")

        if self.logger:
            self.logger.info(f"Chunking {len(documents)} documents...")

        # Chunk all documents
        chunks = self.chunker.chunk_documents(documents)

        # Add chunk metadata
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            # Enhance metadata
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size_config": self.chunk_size,
                "chunk_overlap_config": self.chunk_overlap,
                "chunk_length": len(chunk.page_content)
            })
            enriched_chunks.append(chunk)

        if self.logger:
            stats = self.get_chunking_stats(documents, enriched_chunks)
            self.logger.info(
                f"Chunking complete: {stats['total_chunks']} chunks from "
                f"{stats['total_documents']} documents "
                f"(avg {stats['chunks_per_document']:.1f} chunks/doc)"
            )

        return enriched_chunks

    def get_chunking_stats(
        self,
        original_documents: List[Document],
        chunks: List[Document]
    ) -> dict:
        """
        Get statistics about chunking operation.

        Args:
            original_documents: Original documents before chunking
            chunks: Chunked documents

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                "total_documents": len(original_documents),
                "total_chunks": 0,
                "chunks_per_document": 0,
                "avg_chunk_length": 0,
                "min_chunk_length": 0,
                "max_chunk_length": 0
            }

        chunk_lengths = [len(c.page_content) for c in chunks]

        return {
            "total_documents": len(original_documents),
            "total_chunks": len(chunks),
            "chunks_per_document": len(chunks) / len(original_documents) if original_documents else 0,
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths)
        }

    def estimate_embedding_cost(
        self,
        chunks: List[Document],
        cost_per_million_tokens: float = 0.50
    ) -> dict:
        """
        Estimate embedding cost for chunks.

        Args:
            chunks: Chunked documents
            cost_per_million_tokens: Cost per 1M tokens (default: $0.50)

        Returns:
            Dictionary with cost estimates
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "estimated_tokens": 0,
                "estimated_cost_usd": 0.0
            }

        # Rough estimate: avg 1.5 chars per token for Korean
        total_chars = sum(len(c.page_content) for c in chunks)
        estimated_tokens = total_chars / 1.5

        return {
            "total_chunks": len(chunks),
            "estimated_tokens": int(estimated_tokens),
            "estimated_cost_usd": (estimated_tokens / 1_000_000) * cost_per_million_tokens
        }


def create_chunking_stage(
    chunk_size: int = ChunkingStage.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = ChunkingStage.DEFAULT_OVERLAP,
    logger: Optional[RAGLogger] = None
) -> ChunkingStage:
    """
    Factory function to create ChunkingStage.

    Args:
        chunk_size: Maximum chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        logger: Optional logger instance

    Returns:
        Configured ChunkingStage instance
    """
    return ChunkingStage(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        logger=logger
    )
