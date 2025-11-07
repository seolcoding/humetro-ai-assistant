#!/usr/bin/env python3
"""
Document Organizer
==================

Organize extracted knowledge documents into topic-based structure.

Usage:
    uv run python src/knowledge_extraction/document_organizer.py \
        --input data/evaluation/knowledge_extraction_poc/extracted_documents.json \
        --output-dir data/processed/knowledge_docs
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentOrganizer:
    """
    Organize knowledge documents by topic and save as Markdown files.
    """

    def __init__(self, min_doc_size: int = 500, merge_small_docs: bool = True):
        """
        Args:
            min_doc_size: Minimum document size in tokens (approximate)
            merge_small_docs: Whether to merge small documents
        """
        self.min_doc_size = min_doc_size
        self.merge_small_docs = merge_small_docs

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough approximation: ~1 token per 4 characters for Korean
        return len(text) // 4

    def organize_documents(
        self,
        documents: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Organize documents into topic-based directory structure.

        Args:
            documents: List of extracted documents
            output_dir: Output directory

        Returns:
            Organization statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("ORGANIZING DOCUMENTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total documents: {len(documents)}")
        logger.info(f"Output directory: {output_dir}")

        # Group by topic_path
        grouped = defaultdict(list)
        for doc in documents:
            topic_path = doc.get('topic_path', 'uncategorized')
            grouped[topic_path].append(doc)

        logger.info(f"Unique topics: {len(grouped)}")

        # Create directory structure and save documents
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {
            "total_documents": len(documents),
            "unique_topics": len(grouped),
            "files_created": 0,
            "small_docs_merged": 0,
            "documents_by_category": defaultdict(int)
        }

        for topic_path, docs in grouped.items():
            # Parse topic path
            parts = topic_path.split('/')
            if len(parts) >= 1:
                category = parts[0]
                subcategory = parts[1] if len(parts) >= 2 else ""
                filename = parts[-1] if len(parts) >= 2 else topic_path

                # Create category directory
                category_dir = output_dir / category
                if subcategory:
                    category_dir = category_dir / subcategory
                category_dir.mkdir(parents=True, exist_ok=True)

                # Merge documents with same topic
                if len(docs) == 1:
                    merged_doc = docs[0]['document']
                else:
                    # Multiple docs with same topic - merge them
                    merged_doc = self._merge_documents(docs)
                    stats["small_docs_merged"] += len(docs) - 1

                # Estimate size
                token_count = self.estimate_token_count(merged_doc)

                # Save file
                file_path = category_dir / f"{filename}.md"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(merged_doc)

                stats["files_created"] += 1
                stats["documents_by_category"][category] += 1

                logger.info(f"  ✓ {topic_path} → {file_path.relative_to(output_dir)} ({token_count} tokens)")

        logger.info(f"\n{'='*60}")
        logger.info("ORGANIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Files created: {stats['files_created']}")
        logger.info(f"Documents merged: {stats['small_docs_merged']}")
        logger.info(f"\nDocuments by category:")
        for category, count in stats["documents_by_category"].items():
            logger.info(f"  - {category}: {count} files")

        return stats

    def _merge_documents(self, docs: List[Dict[str, Any]]) -> str:
        """
        Merge multiple documents with the same topic.

        Args:
            docs: Documents to merge

        Returns:
            Merged Markdown content
        """
        if len(docs) == 1:
            return docs[0]['document']

        # Extract front matter from first doc
        first_doc = docs[0]['document']
        if first_doc.startswith('---'):
            fm_end = first_doc.find('---', 3)
            front_matter = first_doc[:fm_end + 3]
            content_parts = []

            # Extract content from all docs
            for doc in docs:
                doc_content = doc['document']
                if doc_content.startswith('---'):
                    content_start = doc_content.find('---', 3) + 3
                    content = doc_content[content_start:].strip()
                else:
                    content = doc_content.strip()
                content_parts.append(content)

            # Merge
            merged_content = '\n\n'.join(content_parts)
            return f"{front_matter}\n\n{merged_content}"
        else:
            # No front matter - just concatenate
            return '\n\n---\n\n'.join(doc['document'] for doc in docs)


def main():
    parser = argparse.ArgumentParser(description="Organize extracted knowledge documents")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to extracted documents JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for organized Markdown files"
    )
    parser.add_argument(
        "--min-doc-size",
        type=int,
        default=500,
        help="Minimum document size in tokens"
    )
    parser.add_argument(
        "--merge-small",
        action="store_true",
        help="Merge small documents"
    )
    args = parser.parse_args()

    # Load documents
    with open(args.input, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Organize
    organizer = DocumentOrganizer(
        min_doc_size=args.min_doc_size,
        merge_small_docs=args.merge_small
    )

    stats = organizer.organize_documents(documents, Path(args.output_dir))

    # Save stats
    stats_path = Path(args.output_dir) / "organization_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\nStatistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
