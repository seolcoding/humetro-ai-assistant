"""
Text chunking module for RAG pipeline.

This module provides configurable text chunking functionality
for splitting documents into manageable chunks for embedding and retrieval.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextChunker:
    """
    Configurable text chunking for RAG pipelines.

    Uses RecursiveCharacterTextSplitter with Korean-optimized separators.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        separators: List[str] = None,
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to use for splitting (default: Korean-optimized)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Korean-optimized separators
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects
        """
        chunks = []

        for doc in documents:
            doc_chunks = self.splitter.split_documents([doc])
            chunks.extend(doc_chunks)

        return chunks

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into chunks.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunked Document objects
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.splitter.split_documents([doc])

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks.

        Args:
            chunks: List of chunked documents

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
            }

        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
        }


def create_chunker(chunk_size: int = 1024, chunk_overlap: int = 256) -> TextChunker:
    """
    Create a text chunker with default settings.

    Args:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        TextChunker instance
    """
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
