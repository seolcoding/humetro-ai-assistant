"""
Stage 1 Extension: Dasan Call Knowledge Base Loader

This module provides a data loader for Dasan Call knowledge documents
stored in JSONL format, compatible with the existing RAG pipeline.

Usage:
    from src.rag_pipeline.stages.stage_01_dasan_loader import DasanKnowledgeLoader

    loader = DasanKnowledgeLoader(
        jsonl_path=Path("data/dasan_call/knowledge_docs.jsonl")
    )
    documents = loader.load_documents()
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from src.common.logger import RAGLogger


class DasanKnowledgeLoader:
    """
    Load Dasan knowledge documents from JSONL.

    Compatible with existing Stage 1 interface (DataCollectionStage).
    """

    def __init__(
        self,
        jsonl_path: Path = Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl"),
        logger: Optional[RAGLogger] = None,
        min_validation_score: Optional[float] = None,
        validated_only: bool = False
    ):
        """
        Initialize Dasan knowledge loader.

        Args:
            jsonl_path: Path to consolidated JSONL file
            logger: Optional RAG logger instance
            min_validation_score: Minimum validation score to include (if available)
            validated_only: Only load documents that passed validation
        """
        self.jsonl_path = jsonl_path
        self.logger = logger or self._create_default_logger()
        self.min_validation_score = min_validation_score
        self.validated_only = validated_only

        self.stats = {
            "total_lines": 0,
            "documents_loaded": 0,
            "documents_filtered": 0,
            "parse_errors": 0
        }

    def load_documents(self) -> List[Document]:
        """
        Load knowledge documents into LangChain Document format.

        Returns:
            List[Document] with rich metadata for RAG pipeline

        Raises:
            FileNotFoundError: If JSONL file doesn't exist
        """
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self.logger.info(f"Loading Dasan knowledge documents from {self.jsonl_path}")

        documents = []

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total_lines'] += 1

                try:
                    data = json.loads(line)

                    # Apply filters
                    if not self._should_include_document(data):
                        self.stats['documents_filtered'] += 1
                        continue

                    # Convert to LangChain Document
                    doc = self._create_langchain_document(data)
                    documents.append(doc)

                    self.stats['documents_loaded'] += 1

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    self.stats['parse_errors'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process line {line_num}: {e}")
                    self.stats['parse_errors'] += 1

        # Log summary
        self._log_summary()

        return documents

    def _should_include_document(self, data: Dict[str, Any]) -> bool:
        """
        Check if document should be included based on filters.

        Args:
            data: Document data from JSONL

        Returns:
            True if document should be included
        """
        metadata = data.get('metadata', {})

        # Filter by validation status
        if self.validated_only:
            if not metadata.get('validated', False):
                return False

        # Filter by validation score (if available)
        if self.min_validation_score is not None:
            score = metadata.get('validation_score')
            if score is None or score < self.min_validation_score:
                return False

        return True

    def _create_langchain_document(self, data: Dict[str, Any]) -> Document:
        """
        Convert JSONL data to LangChain Document.

        Args:
            data: Document data from JSONL

        Returns:
            LangChain Document with metadata
        """
        # Extract Markdown document
        doc_content = data.get('document', '')

        # Build LangChain metadata
        doc_metadata = data.get('metadata', {})

        metadata = {
            # Required fields for RAG pipeline
            "source": f"dasan_call_{data.get('dialogue_id', 'unknown')}",
            "dialogue_id": data.get('dialogue_id', 'unknown'),

            # Topic hierarchy (for filtering/routing)
            "topic_path": data.get('topic_path', ''),
            "primary_topic": data.get('primary_topic', ''),
            "secondary_topics": data.get('secondary_topics', []),

            # RAG optimization
            "doc_length": doc_metadata.get('doc_length_chars', len(doc_content)),
            "estimated_chunks": doc_metadata.get('estimated_chunks', 1),

            # Quality filtering
            "validated": doc_metadata.get('validated', False),
            "validation_issues": doc_metadata.get('validation_issues', []),

            # Original Q&A for evaluation
            "original_question": data.get('original_question', ''),
            "original_answer": data.get('original_answer', ''),

            # Categorization
            "category": doc_metadata.get('category', ''),
            "domain": doc_metadata.get('domain', ''),

            # Extraction metadata (for debugging/tracing)
            "extraction_model": doc_metadata.get('extraction_model', 'unknown'),
            "extraction_date": doc_metadata.get('extraction_date', 'unknown'),
        }

        return Document(
            page_content=doc_content,
            metadata=metadata
        )

    def _create_default_logger(self) -> RAGLogger:
        """Create default logger if none provided."""
        logger = RAGLogger(
            experiment_name="dasan_knowledge_loader"
        )
        return logger

    def _log_summary(self):
        """Log loading summary."""
        self.logger.info("=" * 60)
        self.logger.info("DASAN KNOWLEDGE LOADER SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total lines in JSONL: {self.stats['total_lines']}")
        self.logger.info(f"Documents loaded: {self.stats['documents_loaded']}")
        self.logger.info(f"Documents filtered: {self.stats['documents_filtered']}")
        self.logger.info(f"Parse errors: {self.stats['parse_errors']}")
        self.logger.info("=" * 60)

    def get_stats(self) -> Dict[str, int]:
        """
        Get loading statistics.

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()


def load_dasan_knowledge(
    jsonl_path: Path = Path("data/dasan_call/knowledge_docs.jsonl"),
    validated_only: bool = False
) -> List[Document]:
    """
    Convenience function to load Dasan knowledge documents.

    Args:
        jsonl_path: Path to JSONL file
        validated_only: Only load validated documents

    Returns:
        List of LangChain Documents
    """
    loader = DasanKnowledgeLoader(
        jsonl_path=jsonl_path,
        validated_only=validated_only
    )
    return loader.load_documents()


if __name__ == '__main__':
    # Example usage
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_path = Path("data/dasan_call/knowledge_docs.jsonl")

    print(f"Loading documents from: {jsonl_path}")

    loader = DasanKnowledgeLoader(jsonl_path=jsonl_path)
    documents = loader.load_documents()

    print(f"\n✓ Loaded {len(documents)} documents")

    if documents:
        print("\nFirst document preview:")
        print(f"  Source: {documents[0].metadata['source']}")
        print(f"  Topic: {documents[0].metadata['primary_topic']}")
        print(f"  Length: {documents[0].metadata['doc_length']} chars")
        print(f"  Validated: {documents[0].metadata['validated']}")
        print(f"  Content preview: {documents[0].page_content[:200]}...")
