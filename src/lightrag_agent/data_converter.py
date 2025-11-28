#!/usr/bin/env python3
"""
Data Converter for LightRAG
============================

Convert AI_HUB_DASAN_QA JSONL format to LightRAG-compatible text documents.

Input Format (knowledge_docs_full.jsonl):
{
    "dialogue_id": "...",
    "original_question": "질문",
    "original_answer": "답변",
    "document": "전체 문서 텍스트",
    "metadata": {...}
}

Output Format:
Plain text documents for LightRAG indexing, preserving:
- Full context from document field
- Dialogue ID for traceability
- Category metadata for filtering
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DasanDataConverter:
    """Convert Dasan QA JSONL to LightRAG text documents"""

    def __init__(self, jsonl_path: Path):
        """
        Initialize converter

        Args:
            jsonl_path: Path to knowledge_docs_full.jsonl
        """
        self.jsonl_path = Path(jsonl_path)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

    def load_jsonl(self) -> Iterator[Dict[str, Any]]:
        """
        Load JSONL file line by line (memory efficient)

        Yields:
            Dict containing dialogue data
        """
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def convert_to_text_documents(self, max_docs: int = None) -> List[str]:
        """
        Convert JSONL to plain text documents for LightRAG

        Args:
            max_docs: Maximum number of documents to convert (None = all)

        Returns:
            List of text documents
        """
        documents = []

        logger.info(f"Converting {self.jsonl_path.name} to LightRAG format...")

        for i, entry in enumerate(tqdm(self.load_jsonl(), desc="Converting")):
            if max_docs and i >= max_docs:
                break

            # Extract fields
            dialogue_id = entry.get("dialogue_id", f"unknown_{i}")
            question = entry.get("original_question", "")
            answer = entry.get("original_answer", "")
            document = entry.get("document", "")
            metadata = entry.get("metadata", {})
            category = metadata.get("category", "unknown")

            # Format as plain text document
            # Strategy: Use full document text for maximum context
            # Include Q&A for relevance
            text_doc = f"""[Document ID: {dialogue_id}]
[Category: {category}]

Question: {question}

Answer: {answer}

Context:
{document}
"""

            documents.append(text_doc)

        logger.info(f"✅ Converted {len(documents)} documents")
        return documents

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics

        Returns:
            Dict with counts and categories
        """
        total = 0
        categories = {}

        for entry in self.load_jsonl():
            total += 1
            category = entry.get("metadata", {}).get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

        return {
            "total_documents": total,
            "categories": categories
        }

    def save_as_text(self, output_path: Path, max_docs: int = None):
        """
        Save converted documents as single text file

        Args:
            output_path: Output text file path
            max_docs: Maximum documents to convert
        """
        documents = self.convert_to_text_documents(max_docs=max_docs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc)
                f.write("\n" + "="*80 + "\n")  # Document separator

        logger.info(f"💾 Saved {len(documents)} documents to {output_path}")


def main():
    """CLI test"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert Dasan JSONL to LightRAG format")
    parser.add_argument(
        "--input",
        default="data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        default="data/AI_HUB_DASAN_QA/09_lightrag/documents.txt",
        help="Output text file"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum documents to convert (for testing)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    converter = DasanDataConverter(args.input)

    # Show statistics
    stats = converter.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total: {stats['total_documents']} documents")
    logger.info(f"  Categories: {stats['categories']}")

    # Convert and save
    converter.save_as_text(args.output, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
