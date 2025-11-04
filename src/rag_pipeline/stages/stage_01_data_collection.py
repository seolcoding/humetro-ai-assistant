"""
Stage 1: Data Collection

Loads crawled markdown documents from cache and applies cleaning operations
to remove social sharing links and UI artifacts.
"""

import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from src.common.logger import RAGLogger


class DataCollectionStage:
    """
    Data Collection Stage for RAG Pipeline.

    Loads markdown documents from crawled cache and applies cleaning
    operations to remove social sharing links and template artifacts.

    Uses "transportation" terminology in code while actual cache paths
    use "traffic" for URL compatibility.
    """

    # Regex patterns for social sharing links and UI artifacts
    SOCIAL_LINK_PATTERNS = [
        # Links like [인쇄](url) or [스크랩](url)
        r'\[인쇄\]\([^)]+\)',
        r'\[스크랩\]\([^)]+\)',

        # Social buttons like [ 카카오톡 ], [ 페이스북 ], [ X ], etc.
        r'\[\s*카카오톡\s*\]\([^)]+\)',
        r'\[\s*페이스북\s*\]\([^)]+\)',
        r'\[\s*X\s*\]\([^)]+\)',
        r'\[\s*네이버블로그\s*\]\([^)]+\)',
        r'\[\s*URL\s*복사\s*\]\([^)]+\)',

        # Action links like [**네이버blog**], [**메일전송**], [**소스복사**]
        r'\[\*\*네이버blog\*\*\]\([^)]+\)',
        r'\[\*\*메일전송\*\*\]\([^)]+\)',
        r'\[\*\*소스복사\*\*\]\([^)]+\)',

        # Generic pattern for markdown links in sharing sections
        r'\[\s*\*\*[^\]]+\*\*\s*\]\([^)]+(?:"새창열림"|"소스복사")[^)]*\)',
    ]

    def __init__(
        self,
        transportation_cache_dir: Path = Path("./data/crawled/seoul_traffic"),
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Data Collection Stage.

        Args:
            transportation_cache_dir: Base cache directory (uses "traffic" in path)
            logger: Optional logger instance
        """
        self.transportation_cache_dir = Path(transportation_cache_dir)
        self.markdown_dir = self.transportation_cache_dir / "markdown_deduplicated"
        self.logger = logger

        # Compile regex patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.MULTILINE | re.UNICODE)
            for pattern in self.SOCIAL_LINK_PATTERNS
        ]

        if self.logger:
            self.logger.info(
                f"DataCollectionStage initialized with cache: {self.transportation_cache_dir}"
            )

    def check_cache_exists(self) -> bool:
        """
        Check if cache directory and markdown files exist.

        Returns:
            True if cache exists with markdown files
        """
        if not self.transportation_cache_dir.exists():
            if self.logger:
                self.logger.warning(
                    f"Cache directory not found: {self.transportation_cache_dir}"
                )
            return False

        if not self.markdown_dir.exists():
            if self.logger:
                self.logger.warning(
                    f"Markdown directory not found: {self.markdown_dir}"
                )
            return False

        # Check for markdown files
        md_files = list(self.markdown_dir.glob("*.md"))
        if not md_files:
            if self.logger:
                self.logger.warning(
                    f"No markdown files found in: {self.markdown_dir}"
                )
            return False

        if self.logger:
            self.logger.info(
                f"Cache exists with {len(md_files)} markdown files"
            )

        return True

    def clean_markdown_content(self, content: str) -> str:
        """
        Remove social sharing links and UI artifacts from markdown content.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned markdown content
        """
        cleaned = content

        # Apply all regex patterns
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)

        # Remove excessive blank lines (more than 2 consecutive)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def load_documents(self) -> List[Document]:
        """
        Load and clean markdown documents from cache.

        Returns:
            List of cleaned LangChain Document objects

        Raises:
            FileNotFoundError: If cache directory doesn't exist
            ValueError: If no markdown files found
        """
        # Verify cache exists
        if not self.check_cache_exists():
            raise FileNotFoundError(
                f"Cache not found at {self.transportation_cache_dir}. "
                f"Please run crawler first."
            )

        if self.logger:
            self.logger.info(f"Loading markdown files from: {self.markdown_dir}")

        # Load all markdown files
        loader = DirectoryLoader(
            str(self.markdown_dir),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True if self.logger and self.logger.verbose else False
        )

        documents = loader.load()

        if not documents:
            raise ValueError(
                f"No documents loaded from {self.markdown_dir}"
            )

        if self.logger:
            self.logger.info(
                f"Loaded {len(documents)} documents, applying cleaning..."
            )

        # Clean each document's content
        cleaned_documents = []
        for doc in documents:
            cleaned_content = self.clean_markdown_content(doc.page_content)

            # Create new document with cleaned content
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata={
                    **doc.metadata,
                    "cleaned": True,
                    "original_length": len(doc.page_content),
                    "cleaned_length": len(cleaned_content)
                }
            )
            cleaned_documents.append(cleaned_doc)

        if self.logger:
            total_original = sum(d.metadata["original_length"] for d in cleaned_documents)
            total_cleaned = sum(d.metadata["cleaned_length"] for d in cleaned_documents)
            reduction = ((total_original - total_cleaned) / total_original * 100) if total_original > 0 else 0

            self.logger.info(
                f"Cleaning complete: {len(cleaned_documents)} documents, "
                f"content reduced by {reduction:.1f}%"
            )

        return cleaned_documents

    def get_collection_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about collected documents.

        Args:
            documents: List of documents

        Returns:
            Dictionary with collection statistics
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "avg_document_length": 0,
                "min_document_length": 0,
                "max_document_length": 0
            }

        lengths = [len(doc.page_content) for doc in documents]

        return {
            "total_documents": len(documents),
            "total_characters": sum(lengths),
            "avg_document_length": sum(lengths) // len(lengths),
            "min_document_length": min(lengths),
            "max_document_length": max(lengths)
        }


def create_data_collection_stage(
    transportation_cache_dir: Path = Path("./data/crawled/seoul_traffic"),
    logger: Optional[RAGLogger] = None
) -> DataCollectionStage:
    """
    Factory function to create DataCollectionStage.

    Args:
        transportation_cache_dir: Base cache directory
        logger: Optional logger instance

    Returns:
        Configured DataCollectionStage instance
    """
    return DataCollectionStage(
        transportation_cache_dir=transportation_cache_dir,
        logger=logger
    )
